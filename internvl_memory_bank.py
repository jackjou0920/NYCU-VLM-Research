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
def score_l2_norm(tokens: torch.Tensor, question_embed=None) -> torch.Tensor:
    return tokens.float().norm(dim=-1)

def score_random(tokens: torch.Tensor, question_embed=None) -> torch.Tensor:
    return torch.rand(tokens.shape[0], device=tokens.device)

def score_information_density(
    tokens, question_embed,
    alpha=0.3, beta=0.3, gamma=0.4,
    knn_k=3,
    use_novelty: bool = False,   # 新增開關
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

    # 避免 mean() 為常數，改用對數或其他非線性轉換，或者直接依照 batch 統計量
    # 這裡採用簡單的 Max 正規化 (考慮到線上推論特性)
    if N > 1:
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
            protected_tiles: int = 1,
            protect_position: str = "last",  # 請依 load_image_tiles 實際順序確認
    ):
        self.capacity_tiles = max(1, capacity // num_image_token)
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

        self.total_seen_tiles = 0
        self.total_dropped_tiles = 0
        self.size_history = []

    def add_tile(self, tile_tokens: torch.Tensor):
        token_scores = self.score_fn(tile_tokens, self.question_embed)
        tile_score = token_scores.mean().item()

        self.tiles.append(tile_tokens)
        self.tile_scores.append(tile_score)
        self.total_seen_tiles += 1
        # 不在這裡做任何淘汰，只記錄「如果現在 finalize 會保留幾個」方便觀察
        self.size_history.append(min(len(self.tiles), self.capacity_tiles) * self.num_image_token)

    def _protected_indices(self, n):
        if self.protected_tiles <= 0:
            return set()
        if self.protect_position == "first":
            return set(range(self.protected_tiles))
        return set(range(n - self.protected_tiles, n))

    def finalize(self):
        n = len(self.tiles)
        if n == 0:
            return torch.empty(0, self.dim, device=self.device, dtype=self.dtype), {
                "final_size": 0, "total_seen": 0, "total_dropped": 0,
                "compression_ratio": 0.0, "size_history": self.size_history,
            }

        if n <= self.capacity_tiles or self.mode == "fifo":
            keep_idx = (
                list(range(max(0, n - self.capacity_tiles), n)) 
                if self.mode == "fifo" and n > self.capacity_tiles else list(range(n))
            )
        else:
            prot_idx = self._protected_indices(n)
            work_budget = self.capacity_tiles - len(prot_idx)
            work_idx = [i for i in range(n) if i not in prot_idx]

            # 依據 self.mode 決定淘汰策略，修復原本忽略 mode 的問題
            if self.mode in ["evict"]:
                # 策略：全域 Top-K，確保 budget 增加時保留的 Tile 是嚴格超集
                work_idx.sort(key=lambda i: self.tile_scores[i], reverse=True)
                selected = work_idx[:work_budget]
            else:
                # ── 關鍵：把 work_idx 依「原始空間順序」切成 work_budget 個連續 bin，
                #    每個 bin 內挑分數最高的 tile。bin 邊界保證了全圖覆蓋不留空洞，
                #    bin 內挑分數保留了「哪裡資訊量高就多留一點細節」的能力。
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

        final_tokens = torch.cat([self.tiles[i] for i in keep_idx], dim=0)
        stats = {
            "final_size": final_tokens.shape[0],
            "total_seen": self.total_seen_tiles * self.num_image_token,
            "total_dropped": self.total_dropped_tiles * self.num_image_token,
            "compression_ratio": self.total_seen_tiles / max(len(keep_idx), 1),
            "size_history": self.size_history,
        }
        return final_tokens, stats
