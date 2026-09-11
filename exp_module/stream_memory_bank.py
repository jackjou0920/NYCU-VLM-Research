import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Importance scoring：統一介面 (tokens, question_embed=None, norm_stats=None) -> [N]
#
# norm_stats: 可選的 dict {"mean": scalar tensor, "std": scalar tensor}
#   - 若提供，signal 項會用「跨 tile 的全域統計量」做 z-score 正規化，
#     讓不同 tile 各自算出來的分數在同一個尺度上可比較。
#   - 若不提供（None），fallback 回舊版「單一呼叫內的相對排名」（維持
#     StreamingMemoryBank 原本行為不變，因為它的 buffer 本來就橫跨多個 tile）。
# ──────────────────────────────────────────────────────────────────────────────
def score_l2_norm(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    # **_ 吞掉 norm_stats 等只有 info_density 用得到的 kwarg，讓 finalize()/merge
    # 的重算路徑對三個 score_fn 都能用同一個呼叫介面（l2_norm / random 忽略之）。
    return tokens.float().norm(dim=-1)

def score_random(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    return torch.rand(tokens.shape[0], device=tokens.device)

def score_information_density(
    tokens, question_embed,
    alpha=0.3, beta=0.3, gamma=0.4,
    knn_k=3,
    use_novelty: bool = False,   # 新增開關
    norm_stats: dict | None = None,
    # ↑ 修正跨 tile 分數不可比問題：
    #   若提供 {"mean":..., "std":...}（來自 finalize() 時對「所有已收集 tile」
    #   算出的全域統計量），signal 改用全域 z-score → sigmoid 正規化。
    #   若為 None（例如舊呼叫路徑 / add_tile() 內的即時估算），fallback 回
    #   原本「單一 tile 內部 max 正規化」的行為，語意不變。
):
    """
    Question-aware Information Density Score
    Score = alpha*S + beta*N + gamma*R，三項各自正規化到 [0,1] 後再加權，
    避免任一項的原始尺度（尤其 novelty 曾經可能 >1）扭曲權重的實際意義。

    tokens:         [N, D]  (已經過 mlp1，跟 LLM embedding space 同維度)
    question_embed: [D]     (question 文字的 mean-pooled LLM embedding，同一個 space)
    """
    x = tokens.float()
    N = x.shape[0]

    # ---- Signal Strength：用相對排名而非除以 max，避免單一離群值壓縮整批分數 ----
    raw_norm = x.norm(dim=-1)

    if norm_stats is not None:
        # 全域 z-score：raw_norm 相對於「所有已收集 tile」的 mean/std，
        # 再用 sigmoid 壓進 (0,1)。同一張圖不同 tile 算出來的 signal
        # 現在才是同一把尺，可以直接跨 tile 比大小（見 finalize() 的呼叫處）。
        z = (raw_norm - norm_stats["mean"]) / norm_stats["std"]
        signal = torch.sigmoid(z)
    elif N > 1:
        # fallback：舊行為，tile 內部 max 正規化。
        # 注意：這個分支算出來的分數「只在同一次呼叫（同一個 tile）內」
        # 彼此可比，不能拿去跟其他 tile 的分數直接比較大小。
        signal = raw_norm / (raw_norm.max() + 1e-9)
    else:
        signal = torch.zeros(N, device=x.device)

    # ---- Novelty：top-k 最相似的平均相似度，而非只看單一最像的（1-NN 對雜訊太敏感）----
    x_norm = F.normalize(x, dim=-1)
    sim = x_norm @ x_norm.T
    sim.fill_diagonal_(-1.0)
    k = min(knn_k, max(N - 1, 1))
    topk_sim = sim.topk(k, dim=-1).values.mean(dim=-1) if N > 1 else torch.zeros(N, device=x.device)
    
    if use_novelty:
        novelty = ((1 - topk_sim) / 2).clamp(0.0, 1.0)
    else:
        novelty = torch.zeros(N, device=x.device)
        beta = 0.0

    # ---- Question Relevance ----
    if question_embed is None or (isinstance(question_embed, str)):
        # 沒有有效 question_embed 時，退化成不考慮 relevance（權重併入其他兩項）
        relevance = torch.zeros(N, device=x.device)
        gamma_eff = 0.0
    else:
        q = F.normalize(question_embed.float().reshape(-1), dim=-1)
        relevance = ((x_norm @ q) + 1) / 2   # [-1,1] -> [0,1]
        gamma_eff = gamma

    total = alpha + beta + gamma_eff
    a, b, g = alpha / total, beta / total, gamma_eff / total
    score = a * signal + b * novelty + g * relevance
    return score


SCORE_FUNCS = {
    "l2_norm": score_l2_norm,
    "info_density": score_information_density,
    "random": score_random,
}


class TileStreamingMemoryBank:
    """
    以 Tile (256 tokens) 為單位。
    add_tile() 只負責累積 + 算分,真正的淘汰決策延後到 finalize(),
    並用「空間分箱 + 箱內取最高分」保證全圖覆蓋不留空洞。
    """
    def __init__(
        self,
        capacity: int,
        dim: int, device, dtype,
        score_fn: str = "l2_norm",
        mode: str = "evict",
        num_image_token: int = 256,
        question_embed=None,
        tile_agg: str = "mean",
        # ↑ 一個 tile 有 num_image_token 個 token 分數，要壓成單一 tile 分數才能
        #   進 eviction 排序。"mean" = 原本行為（完全不變）。其餘選項都對「分數
        #   分佈的高分端」更敏感，用來救「整片背景 + 一小塊高資訊區（小字 / 招牌
        #   / 人臉）」這種 tile 被平均值稀釋、在壓縮時被誤丟的情況：
        #     "max"       單一最顯著 token
        #     "topk_mean" 分數前 agg_topk_frac 比例的 token 取平均
        #     "mean_std"  mean + agg_std_lambda * std
        #     "quantile"  第 agg_quantile 分位數
        #   paper 就是固定其他設定、只掃這個參數來畫「聚合方式 vs accuracy / 壓縮率」。
        agg_topk_frac: float = 0.1,
        agg_quantile: float = 0.9,
        agg_std_lambda: float = 1.0,
        # ── mode="merge" / "merge_spatial" 專用（其餘 mode 完全忽略）───────────
        #   兩者「選誰留下來」都跟 evict 一模一樣（全域 Top-K），差別只在多一步
        #   把被淘汰 tile 的 token 折進存活 token；保留的 tile 數、每個 tile 的
        #   token 數、final_size 都跟 evict 相同，方便乾淨 A/B。
        #     "merge"         折進全域 cosine 最近的存活 token（可能跨到很遠的 tile）
        #     "merge_spatial" 每個被淘汰 tile 先綁定 raster 距離最近的存活 tile，
        #                     token 只在那一個相鄰 tile 內找位置折 → 折合範圍受空間
        #                     約束，對應「空間連續保留比獨立 token 好」的實驗結論
        merge_weight: str = "score",      # "score"=用 token importance 加權；"uniform"=等權（score_fn=random 時用這個）
        #   merge_sim_thresh：best cosine < thresh 的被淘汰 token 改成硬丟，其餘照折。
        #   -1.0 = 一律 merge（純 merge，evict-vs-merge A/B 用這個）。
        #   刻意不開成 CLI flag；要掃「merge→evict 之間」時直接改這行的預設值
        #   （0.2~0.4 會濾掉「跟任何存活 token 都不像」的雜訊 token，越高越接近 evict）。
        merge_sim_thresh: float = -1.0,
        # ── 存活 tile 的「選擇策略」，跟 mode（evict / merge / merge_spatial）正交 ──
        #   "topk"        : 全域 Top-K，純看分數（原本行為）
        #   "spatial_bin" : 把 tile 依原始空間順序切成 work_budget 個連續 bin，每 bin
        #                   取最高分 → 存活 tile 保證鋪滿全圖、不留大空洞。搭配
        #                   merge_spatial 特別有意義：每個被淘汰 tile 才真的有近鄰
        #                   可折。fifo / n<=capacity 時此參數無作用。
        select: str = "topk",
        protected_tiles: int = 1,
        protect_position: str = "last",  # 請依 load_image_tiles 實際順序確認
        max_newline_tokens: int = 0,
        # ↑ grid-aware finalize（例如 LLaVA-OneVision 的 image_newline）會在攤平時
        #   每一列多插入 1 個 token，這裡先從 capacity 扣掉最壞情況（列數上限）的
        #   token 數，確保 finalize() 之後的實際長度仍然 <= capacity。
        #   跟 InternVL 一樣用 flat 模式（grid_shape=None）的話這個參數不用管，
        #   預設 0 完全不影響原本行為。
    ):
        self.capacity_tiles = max(1, (capacity - max_newline_tokens) // num_image_token)
        self.num_image_token = num_image_token
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.protected_tiles = min(protected_tiles, self.capacity_tiles)
        self.protect_position = protect_position
 
        self.tiles = []        # list of [256, D]，依原始空間(raster)順序累積
        self.tile_scores = []  # list of float
        self.score_fn = SCORE_FUNCS[score_fn]
        self.mode = mode
        self.question_embed = question_embed
        self.tile_agg = tile_agg
        self.agg_topk_frac = agg_topk_frac
        self.agg_quantile = agg_quantile
        self.agg_std_lambda = agg_std_lambda
        self.merge_weight = merge_weight
        self.merge_sim_thresh = merge_sim_thresh
        self.select = select

        self.total_seen_tiles = 0
        self.total_dropped_tiles = 0
        self.size_history = []

    def _aggregate_tile_score(self, token_scores: torch.Tensor) -> float:
        """把一個 tile 內 [num_image_token] 個 token 分數壓成單一 tile 分數。
        由 self.tile_agg 決定；"mean" 完全等於舊行為，其餘見 __init__ 的說明。
        add_tile()（即時估算）與 finalize()（帶全域 norm_stats 重算）都走這裡，
        確保兩處聚合方式一致。"""
        s = token_scores.float().reshape(-1)
        n = s.numel()
        if n == 0:
            return 0.0
        mode = self.tile_agg
        if mode == "mean":
            return s.mean().item()
        if mode == "max":
            return s.max().item()
        if mode == "topk_mean":
            k = max(1, int(round(self.agg_topk_frac * n)))
            return s.topk(k).values.mean().item()
        if mode == "mean_std":
            # unbiased=False：n==1 時回傳 0 而不是 NaN
            return (s.mean() + self.agg_std_lambda * s.std(unbiased=False)).item()
        if mode == "quantile":
            return s.quantile(self.agg_quantile).item()
        raise ValueError(f"unknown tile_agg={mode!r}")

    def add_tile(self, tile_tokens: torch.Tensor):
        token_scores = self.score_fn(tile_tokens, self.question_embed)
        tile_score = self._aggregate_tile_score(token_scores)

        self.tiles.append(tile_tokens)
        self.tile_scores.append(tile_score)
        self.total_seen_tiles += 1
        # 不在這裡做任何淘汰，只記錄「如果現在 finalize 會保留幾個」方便觀察
        self.size_history.append(min(len(self.tiles), self.capacity_tiles) * self.num_image_token)

    def _compute_global_norm_stats(self):
        """把目前 self.tiles 裡所有 token 的 raw L2 norm 攤平算一次
        全域 mean/std，給 score_information_density 的 signal 項做
        z-score 正規化用。只在 finalize() 真的要做 eviction 排序時呼叫，
        避免沒有淘汰需求（n <= capacity 或 fifo）時做多餘的計算。"""
        all_norms = torch.cat([t.float().norm(dim=-1) for t in self.tiles])
        return {"mean": all_norms.mean(), "std": all_norms.std() + 1e-9}

    def _protected_indices(self, n):
        if self.protected_tiles <= 0:
            return set()
        if self.protect_position == "first":
            return set(range(self.protected_tiles))
        return set(range(n - self.protected_tiles, n))

    def _assemble_flat(self, keep_idx, image_newline):
        """原本的行為：純粹依 raster 順序把保留下來的 tile concat 起來。
        image_newline 不是 None 的話，在最後面補一個 newline token
        （對應 LLaVA-OneVision 官方 pack_image_features 在「只有 1 個 patch」
        時的分支：整段 image_feature 後面直接接一個 newline）。"""
        pieces = [self.tiles[i] for i in keep_idx]
        if image_newline is not None:
            newline_tok = image_newline.reshape(1, -1).to(device=self.device, dtype=self.dtype)
            pieces.append(newline_tok)
        return torch.cat(pieces, dim=0)

    def _assemble_grid(self, keep_idx, grid_shape, image_newline):
        """
        Grid-aware 重組：假設 self.tiles[0] 是被保護的 base image（整圖縮圖），
        self.tiles[1:] 依 row-major 順序對應到 grid_shape=(num_patch_height,
        num_patch_width) 這個網格（跟 LLaVA-OneVision anyres 前處理吐出來的
        patch 順序一致）。

        跟官方 pack_image_features 最大的不同：官方是先把「完整的」網格 unpad +
        （必要時）雙線性內插到 anyres_max_9 預算，再逐一 token-row 插入
        newline；這裡因為 eviction 是在 tile（=patch）粒度做的，某一列裡的
        patch 可能只保留一部分、甚至整列被丟光，所以改成「這一列只要還有
        任何一個 patch 存活，就把存活的 patch 依原本 column 順序接起來，
        接完這一列才補一個 newline」。粒度比官方粗（官方是每個 token-row 一個
        newline，這裡是每個 patch-row 一個），但至少讓 LLM 看得到「這裡換行了」
        這個訊號，而不是完全沒有列的概念。

        整列被淘汰的情況：直接跳過該列（不補 newline），效果上跟官方 unpad
        把整條 padding-only 的列裁掉是類似的（尤其搭配呼叫端把純 padding 列
        的分數設成 -inf，讓它們最優先被淘汰時）。
        """
        num_patch_height, num_patch_width = grid_shape
        keep_set = set(keep_idx)

        pieces = []
        if 0 in keep_set:
            pieces.append(self.tiles[0])   # base image，不接 newline，直接放最前面

        newline_tok = None
        if image_newline is not None:
            newline_tok = image_newline.reshape(1, -1).to(device=self.device, dtype=self.dtype)

        for r in range(num_patch_height):
            row_pieces = [
                self.tiles[1 + r * num_patch_width + c]
                for c in range(num_patch_width)
                if (1 + r * num_patch_width + c) in keep_set
            ]
            if row_pieces:
                pieces.extend(row_pieces)
                if newline_tok is not None:
                    pieces.append(newline_tok)

        if not pieces:
            return torch.empty(0, self.dim, device=self.device, dtype=self.dtype)
        return torch.cat(pieces, dim=0)

    def _fold_group(self, src_tiles, tgt_tiles, norm_stats, info):
        """把 src_tiles 的每個 token，依 cosine 折進 tgt_tiles 的 token 裡最相近的
        那個（加權平均），就地改寫 self.tiles[tgt]。src / tgt 的 token 數、tile
        集合都不變，只有 tgt token 的「值」被動過。累計統計進 info（in-place）。"""
        if not src_tiles or not tgt_tiles:
            return
        tgt_tok = torch.cat([self.tiles[i] for i in tgt_tiles], dim=0).float()   # [T, D]
        src_tok = torch.cat([self.tiles[i] for i in src_tiles], dim=0).float()   # [S, D]

        sim = F.normalize(src_tok, dim=-1) @ F.normalize(tgt_tok, dim=-1).T      # [S, T]
        best_sim, tgt = sim.max(dim=-1)                                          # [S]

        if self.merge_sim_thresh > -1.0:
            m = best_sim >= self.merge_sim_thresh
        else:
            m = torch.ones_like(best_sim, dtype=torch.bool)
        info["tokens_hard_dropped"] += int((~m).sum())
        src_tok, tgt, best_sim = src_tok[m], tgt[m], best_sim[m]
        if src_tok.shape[0] == 0:
            return

        if self.merge_weight == "score":
            w_t = torch.cat([
                self.score_fn(self.tiles[i], self.question_embed, norm_stats=norm_stats).reshape(-1)
                for i in tgt_tiles
            ]).float().clamp(min=1e-6)
            w_s = torch.cat([
                self.score_fn(self.tiles[i], self.question_embed, norm_stats=norm_stats).reshape(-1)
                for i in src_tiles
            ]).float()[m].clamp(min=1e-6)
        else:
            w_t = torch.ones(tgt_tok.shape[0], device=tgt_tok.device)
            w_s = torch.ones(src_tok.shape[0], device=src_tok.device)

        # new_t = (w_t·t + Σ_{s→t} w_s·s) / (w_t + Σ_{s→t} w_s)
        num = tgt_tok * w_t.unsqueeze(-1)
        num.index_add_(0, tgt, src_tok * w_s.unsqueeze(-1))
        den = w_t.clone()
        den.index_add_(0, tgt, w_s)
        merged = (num / den.unsqueeze(-1)).to(self.dtype)

        off = 0
        for i in tgt_tiles:
            L = self.tiles[i].shape[0]
            self.tiles[i] = merged[off:off + L].clone()
            off += L

        info["tokens_merged"] += int(m.sum())
        info["_sim_sum"] += float(best_sim.sum())
        info["_sim_cnt"] += int(m.sum())

    def _merge_evicted_tokens(self, keep_idx, prot_idx, norm_stats):
        """把被淘汰 tile 的 token 折進存活 token（就地改寫 self.tiles[target]）。
        保留的 tile 集合、每個 tile 的 token 數、final_size 都跟 evict 完全一樣，
        後面 assemble(flat/grid) 直接沿用 evict 的同一條路徑 → 乾淨的 A/B。

        mode="merge"        : 全域 cosine。每個被淘汰 token → 所有存活(非 protected)
                              token 裡最相近的那個。目標可能落在很遠的 tile，會
                              打散空間結構。
        mode="merge_spatial": 空間約束。每個被淘汰「tile」先指派給 raster 距離最近
                              的存活 tile 當 target，token 只在那一個相鄰 tile 內
                              找 cosine 最近的位置折進去。折合範圍被鎖在單一相鄰
                              tile，符合「空間連續保留 > 獨立 token」。
                              （flat 模式沒有 2D 網格資訊，用 |index 差| 當 2D
                               相鄰的一維近似；raster 同列相鄰準，跨列邊界會偏。）

        回傳 dict：tokens_merged / tokens_hard_dropped（best cosine < merge_sim_thresh
        被硬丟的數量）/ mean_merge_sim（折進去的 token 對其 target 的平均 cosine：
        高 → 被丟內容本來就冗餘、merge≈evict；低 → merge 真的在補會消失的東西）。
        """
        info = {"tokens_merged": 0, "tokens_hard_dropped": 0, "mean_merge_sim": 0.0,
                "_sim_sum": 0.0, "_sim_cnt": 0}
        keep_set = set(keep_idx)
        drop_idx = [i for i in range(len(self.tiles)) if i not in keep_set]
        if not drop_idx:
            return {k: info[k] for k in ("tokens_merged", "tokens_hard_dropped", "mean_merge_sim")}

        # target = 存活但非 protected 的 tile。protected 是全域縮圖 / base image，
        # 不希望被高解析 crop 的 token 污染；真的全部都是 protected 才退回用全部 keep。
        target_pool = [i for i in keep_idx if i not in set(prot_idx)] or list(keep_idx)

        if self.mode == "merge_spatial":
            # 每個被淘汰 tile → raster 距離最近的存活 tile（tie 取較小 index）。
            # 連續一整段被淘汰的 tile 會折進同一個附近的存活 tile。
            assign: dict[int, list[int]] = {}
            for e in drop_idx:
                t = min(target_pool, key=lambda k: (abs(k - e), k))
                assign.setdefault(t, []).append(e)
            for t, evs in assign.items():
                self._fold_group(evs, [t], norm_stats, info)
        else:
            # mode == "merge"：所有存活 tile 當一個大 target 池，一次折完
            self._fold_group(drop_idx, target_pool, norm_stats, info)

        if info["_sim_cnt"] > 0:
            info["mean_merge_sim"] = info["_sim_sum"] / info["_sim_cnt"]
        return {k: info[k] for k in ("tokens_merged", "tokens_hard_dropped", "mean_merge_sim")}

    def finalize(self, grid_shape=None, image_newline=None):
        """
        grid_shape   : None → 跟原本一模一樣的 flat 模式（InternVL 用這個）。
                       (num_patch_height, num_patch_width) → grid-aware 模式，
                       finalize 時依原始 2D 座標重組並在列邊界補 image_newline
                       （LLaVA-OneVision 用這個）。
        image_newline: None → 不插入任何 newline token。
                       Tensor[D] → 該模型的 image_newline embedding，flat 模式下
                       只在最後補一個（對應單一 patch 的情況），grid 模式下每一
                       列補一個。
        """
        n = len(self.tiles)
        if n == 0:
            return torch.empty(0, self.dim, device=self.device, dtype=self.dtype), {
                "final_size": 0, "total_seen": 0, "total_dropped": 0,
                "compression_ratio": 0.0, "size_history": self.size_history,
                "work_budget": 0, "grid_patches_kept": 0, "rows_kept": 0,
            }

        merge_stats = {"tokens_merged": 0, "tokens_hard_dropped": 0, "mean_merge_sim": 0.0}

        if n <= self.capacity_tiles or self.mode == "fifo":
            work_budget = None   # 沒有真的做 eviction，跟原本一樣
            keep_idx = (
                list(range(max(0, n - self.capacity_tiles), n)) 
                if self.mode == "fifo" and n > self.capacity_tiles else list(range(n))
            )
        else:
            # 【修正】add_tile() 當下用 score_fn(..., norm_stats=None) 算出的 tile_scores，
            # 對 info_density 而言是「tile 內部 max 正規化」，只在單一 tile 內部可比，不能拿來跨 tile 排序（原本的 bug）。
            # 這裡先用目前收集到的所有 tile 算一次全域統計量，重新計算 tile_scores 再進入淘汰排序，
            # 確保比的是「tile 之間」誰的訊號真的比較強，而不是各自 tile 內部的相對分佈形狀。
            # l2_norm / random 這兩個 score_fn 忽略 norm_stats，重算等於白算一次但結果不變，維持介面一致不用特判。
            norm_stats = self._compute_global_norm_stats()
            self.tile_scores = [
                self._aggregate_tile_score(
                    self.score_fn(self.tiles[i], self.question_embed, norm_stats=norm_stats)
                )
                for i in range(n)
            ]


            prot_idx = self._protected_indices(n)
            work_budget = self.capacity_tiles - len(prot_idx)
            work_idx = [i for i in range(n) if i not in prot_idx]
 
            # 存活 tile 的選擇策略，由 self.select 決定（跟 mode 正交）
            if self.select != "spatial_bin":
                # "topk"：全域 Top-K，純看分數；budget 增加時保留的 tile 是嚴格超集。
                work_idx.sort(key=lambda i: self.tile_scores[i], reverse=True)
                selected = work_idx[:work_budget]
            else:
                # "spatial_bin"：把 work_idx 依「原始空間順序」切成 work_budget 個連續
                #    bin，每個 bin 內挑分數最高的 tile。bin 邊界保證全圖覆蓋不留空洞，
                #    bin 內挑分數保留「哪裡資訊量高就多留一點細節」的能力。
                bin_edges = torch.linspace(0, len(work_idx), steps=work_budget + 1).long().tolist()
                selected = []
                for k in range(work_budget):
                    lo, hi = bin_edges[k], bin_edges[k + 1]
                    if lo >= hi:
                        # bin 太小（budget 比 tile 數還接近），退化成往前找最近的可用 index
                        hi = min(lo + 1, len(work_idx))
                    bin_local = work_idx[lo:hi]
                    best_local = max(bin_local, key=lambda i: self.tile_scores[i])
                    selected.append(best_local)
 
            keep_idx = sorted(set(prot_idx) | set(selected))
            self.total_dropped_tiles += (n - len(keep_idx))

            if self.mode in ("merge", "merge_spatial"):
                # 被淘汰 tile 的 token 折進存活 token（就地改寫 self.tiles[target]）。
                # keep_idx 不變，所以下面 assemble 的路徑跟 evict 一模一樣。
                merge_stats = self._merge_evicted_tokens(keep_idx, prot_idx, norm_stats)

        if grid_shape is not None:
            final_tokens = self._assemble_grid(keep_idx, grid_shape, image_newline)
            num_patch_height, num_patch_width = grid_shape
            keep_set = set(keep_idx)
            grid_patches_kept = sum(1 for i in keep_idx if i != 0)
            rows_kept = sum(
                1 for r in range(num_patch_height)
                if any((1 + r * num_patch_width + c) in keep_set for c in range(num_patch_width))
            )
            # (row, col) 座標，debug 用：看被留下來的 crop 是不是集中在同一小塊區域
            # （evict 模式在極端壓縮比下的已知副作用），而不是分散在全圖。
            grid_positions_kept = sorted(
                ((i - 1) // num_patch_width, (i - 1) % num_patch_width)
                for i in keep_idx if i != 0
            )
        else:
            final_tokens = self._assemble_flat(keep_idx, image_newline)
            # 注意：這裡不能再假設「一定有 1 個 protected base tile 在 index 0」
            # （LLaVA-OV 的 row-eviction 用法是 protected_tiles=0，keep_idx 裡
            # 每一個都是實際被留下的 tile，不用扣掉任何東西）。
            grid_patches_kept, rows_kept, grid_positions_kept = len(keep_idx), None, None
 
        stats = {
            "final_size": final_tokens.shape[0],
            "total_seen": self.total_seen_tiles * self.num_image_token,
            "total_dropped": self.total_dropped_tiles * self.num_image_token,
            "compression_ratio": self.total_seen_tiles / max(len(keep_idx), 1),
            "size_history": self.size_history,
            # ↓ 新增：debug 用，特別是拿來查「grid 模式下 work_budget 是不是被壓成 0」，
            #   以及 evict 模式選中的 crop 是不是全部擠在同一區域。
            "work_budget": work_budget,
            "grid_patches_kept": grid_patches_kept,   # 不含 base image
            "rows_kept": rows_kept,                    # None 表示 flat 模式，沒有列的概念
            "grid_positions_kept": grid_positions_kept,
            # ↓ mode="merge" 才有意義（其餘 mode 全 0）：
            "tokens_merged": merge_stats["tokens_merged"],
            "tokens_hard_dropped": merge_stats["tokens_hard_dropped"],
            "mean_merge_sim": round(merge_stats["mean_merge_sim"], 4),
        }
        return final_tokens, stats
