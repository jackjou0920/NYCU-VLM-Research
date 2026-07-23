import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Importance scoring：統一介面 (tokens, question_embed=None, norm_stats=None) -> [N]
# 這樣不管哪個 score_fn 都能被 StreamingMemoryBank / TileStreamingMemoryBank 一致呼叫
#
# norm_stats: 可選的 dict {"mean": scalar tensor, "std": scalar tensor}
#   - 若提供，signal 項會用「跨 tile 的全域統計量」做 z-score 正規化，
#     讓不同 tile 各自算出來的分數在同一個尺度上可比較。
#   - 若不提供（None），fallback 回舊版「單一呼叫內的相對排名」（維持
#     StreamingMemoryBank 原本行為不變，因為它的 buffer 本來就橫跨多個 tile）。
# ──────────────────────────────────────────────────────────────────────────────
def score_l2_norm(tokens: torch.Tensor, question_embed=None) -> torch.Tensor:
    return tokens.float().norm(dim=-1)

def score_attention_entropy(tokens: torch.Tensor, question_embed=None) -> torch.Tensor:
    normed = F.normalize(tokens.float(), dim=-1)
    sim = normed @ normed.t()
    sim.fill_diagonal_(-1e9)
    attn = F.softmax(sim, dim=-1)
    entropy = -(attn * attn.clamp_min(1e-9).log()).sum(dim=-1)
    return -entropy

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
    if N > 1:
        signal = raw_norm.argsort().argsort().float() / (N - 1)
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
    "information_density": score_information_density,
    "attn_entropy": score_attention_entropy,
    "random": score_random,
}


# ──────────────────────────────────────────────────────────────────────────────
# Diversity-aware Greedy Selection
# 用來取代「evict」模式裡的一次性 topk：topk 容易一次選中/丟掉整群語意相近的 token，
# 造成小 budget 時方差很大。greedy 每選一個就懲罰跟它相似的候選，強迫多樣性。
# ──────────────────────────────────────────────────────────────────────────────
def select_by_budget_greedy(
    tokens: torch.Tensor,
    scores: torch.Tensor,
    budget: int,
    diversity_weight: float = 0.4,
) -> torch.Tensor:
    """
    tokens: [N, D]
    scores: [N]  (importance score，越高越重要)
    budget: 要保留幾個
    return: keep_idx（相對於輸入 tokens 的 index，未排序，呼叫端自行 .sort()）
    """
    N = tokens.shape[0]
    device = tokens.device
    if N <= budget:
        return torch.arange(N, device=device)

    x_norm = F.normalize(tokens.float(), dim=-1)
    combined = scores.clone().float()
    remaining = torch.ones(N, dtype=torch.bool, device=device)
    selected = torch.zeros(budget, dtype=torch.long, device=device)

    for i in range(budget):
        masked = combined.masked_fill(~remaining, float("-inf"))
        idx = masked.argmax()
        selected[i] = idx
        remaining[idx] = False
        if not remaining.any():
            selected = selected[: i + 1]
            break
        # 懲罰跟剛選中的 token 相似的候選，強迫下一輪選出不同語意區域的 token
        sim_to_new = (x_norm @ x_norm[idx]).clamp(min=0)
        combined = combined - diversity_weight * sim_to_new

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# Progressive Merge （原樣保留，未改動）
# ──────────────────────────────────────────────────────────────────────────────
def bipartite_merge(tokens, scores, sizes, r):
    n = tokens.shape[0]
    r = min(r, n // 2)
    if r <= 0:
        return tokens, scores, sizes

    idx = torch.arange(n, device=tokens.device)
    a_idx, b_idx = idx[0::2], idx[1::2]

    normed = F.normalize(tokens.float(), dim=-1)
    A, B = normed[a_idx], normed[b_idx]
    sim = A @ B.t()
    best_sim, best_j = sim.max(dim=-1)
    order = torch.argsort(best_sim, descending=True)

    used_b = torch.zeros(B.shape[0], dtype=torch.bool, device=tokens.device)
    merge_pairs = []
    for a_local in order.tolist():
        b_local = best_j[a_local].item()
        if used_b[b_local]:
            continue
        used_b[b_local] = True
        merge_pairs.append((a_local, b_local))
        if len(merge_pairs) >= r:
            break

    merged_a_local = {p[0] for p in merge_pairs}
    merged_b_local = {p[1] for p in merge_pairs}

    retained_items = []
    for i in range(A.shape[0]):
        if i not in merged_a_local:
            gi = a_idx[i].item()
            retained_items.append((gi, tokens[gi], scores[gi], sizes[gi]))
    for i in range(B.shape[0]):
        if i not in merged_b_local:
            gj = b_idx[i].item()
            retained_items.append((gj, tokens[gj], scores[gj], sizes[gj]))

    for a_local, b_local in merge_pairs:
        gi, gj = a_idx[a_local].item(), b_idx[b_local].item()
        wi, wj = sizes[gi], sizes[gj]
        w_sum = wi + wj
        if scores[gi] >= scores[gj]:
            new_token = tokens[gi] * 0.8 + tokens[gj] * 0.2
        else:
            new_token = tokens[gi] * 0.2 + tokens[gj] * 0.8
        new_score = max(scores[gi], scores[gj])
        target_idx = min(gi, gj)
        retained_items.append((target_idx, new_token, new_score, w_sum))

    retained_items.sort(key=lambda x: x[0])
    result_tokens = torch.stack([x[1] for x in retained_items], dim=0)
    result_scores = torch.stack([x[2] for x in retained_items], dim=0)
    result_sizes = torch.stack([x[3] for x in retained_items], dim=0)
    return result_tokens, result_scores, result_sizes


# ──────────────────────────────────────────────────────────────────────────────
# Streaming Memory Bank
# ──────────────────────────────────────────────────────────────────────────────
class StreamingMemoryBank:
    """
    mode="fifo"        : 超過 budget 就丟最舊的 local tile。
    mode="evict"        : 用 select_by_budget_greedy 做 diversity-aware 淘汰
                          （取代原本的一次性 topk，降低小 budget 時的高方差）。
    mode="evict_topk"   : 保留原本單純 topk 版本，方便做 ablation 比較。
    mode="merge"        : progressive merge，逼近極小 n 時退化成 evict。
    """
 
    def __init__(
        self,
        capacity: int,
        dim: int, device, dtype,
        score_fn: str = "l2_norm",
        mode: str = "evict",
        question_embed=None,
        protected_len: int = 256,
        diversity_weight: float = 0.4,
    ):
        self.capacity = capacity
        self.protected_len = protected_len
        self.diversity_weight = diversity_weight

        self.tokens = torch.empty(0, dim, device=device, dtype=dtype)
        self.scores = torch.empty(0, device=device, dtype=torch.float32)
        self.sizes = torch.empty(0, device=device, dtype=torch.float32)
        self.score_fn = SCORE_FUNCS[score_fn]
        self.mode = mode
        self.question_embed = question_embed  # 現在應該是一個 [D] tensor，不是字串

        self.total_seen = 0
        self.total_dropped = 0
        self.size_history = []

    def add_tile(self, tile_tokens: torch.Tensor):
        """tile_tokens: [K, D]，一個 tile（或一次 vit_batch）算出來的 vision token。"""
        new_scores = self.score_fn(tile_tokens, self.question_embed)
        new_sizes = torch.ones(tile_tokens.shape[0], device=tile_tokens.device)

        self.tokens = torch.cat([self.tokens, tile_tokens], dim=0)
        self.scores = torch.cat([self.scores, new_scores], dim=0)
        self.sizes = torch.cat([self.sizes, new_sizes], dim=0)
        self.total_seen += tile_tokens.shape[0]
 
        if self.tokens.shape[0] > self.capacity:
            prot_tokens = self.tokens[: self.protected_len]
            prot_scores = self.scores[: self.protected_len]
            prot_sizes = self.sizes[: self.protected_len]

            work_tokens = self.tokens[self.protected_len:]
            work_scores = self.scores[self.protected_len:]
            work_sizes = self.sizes[self.protected_len:]

            work_capacity = self.capacity - self.protected_len
            n_over = work_tokens.shape[0] - work_capacity

            if n_over > 0:
                if self.mode == "fifo":
                    work_tokens = work_tokens[n_over:]
                    work_scores = work_scores[n_over:]
                    work_sizes = work_sizes[n_over:]
                    self.total_dropped += n_over

                elif self.mode == "evict_topk":
                    keep_idx = torch.topk(work_scores, work_capacity, largest=True).indices
                    keep_idx, _ = keep_idx.sort()
                    self.total_dropped += n_over
                    work_tokens = work_tokens[keep_idx]
                    work_scores = work_scores[keep_idx]
                    work_sizes = work_sizes[keep_idx]

                elif self.mode == "evict":
                    keep_idx = select_by_budget_greedy(
                        work_tokens, work_scores, work_capacity,
                        diversity_weight=self.diversity_weight,
                    )
                    keep_idx, _ = keep_idx.sort()
                    self.total_dropped += n_over
                    work_tokens = work_tokens[keep_idx]
                    work_scores = work_scores[keep_idx]
                    work_sizes = work_sizes[keep_idx]

                elif self.mode == "merge":
                    work_tokens, work_scores, work_sizes = bipartite_merge(
                        work_tokens, work_scores, work_sizes, r=n_over
                    )
                    self.total_dropped += n_over
                    if work_tokens.shape[0] > work_capacity:
                        extra = work_tokens.shape[0] - work_capacity
                        keep_idx = select_by_budget_greedy(
                            work_tokens, work_scores, work_capacity,
                            diversity_weight=self.diversity_weight,
                        )
                        keep_idx, _ = keep_idx.sort()
                        work_tokens = work_tokens[keep_idx]
                        work_scores = work_scores[keep_idx]
                        work_sizes = work_sizes[keep_idx]
                        self.total_dropped += extra

            self.tokens = torch.cat([prot_tokens, work_tokens], dim=0)
            self.scores = torch.cat([prot_scores, work_scores], dim=0)
            self.sizes = torch.cat([prot_sizes, work_sizes], dim=0)

        self.size_history.append(self.tokens.shape[0])

    def finalize(self):
        stats = {
            "final_size": self.tokens.shape[0],
            "total_seen": self.total_seen,
            "total_dropped": self.total_dropped,
            "compression_ratio": self.total_seen / max(self.tokens.shape[0], 1),
            "size_history": self.size_history,
        }
        return self.tokens, stats
 

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
            keep_idx = list(range(max(0, n - self.capacity_tiles), n)) if self.mode == "fifo" and n > self.capacity_tiles \
                       else list(range(n))
        else:
            prot_idx = self._protected_indices(n)
            work_budget = self.capacity_tiles - len(prot_idx)
            work_idx = [i for i in range(n) if i not in prot_idx]

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
