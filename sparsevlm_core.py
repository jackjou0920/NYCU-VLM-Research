"""
SparseVLM「忠於論文設計」版本。

參考對象：
  - 論文：Zhang et al., "SparseVLM: Visual Token Sparsification for Efficient
    Vision-Language Model Inference", ICML 2025 (arXiv:2410.04417)
  - 官方原始碼：https://github.com/Gumpest/SparseVLMs
    （讀了 llava/model/language_model/{modelling_sparse_llama,score,utils}.py
    理解演算法後，用自己的變數命名/函式切法重新實作，不是照抄；下面每個函式
    的 docstring 都寫了對應官方碼的哪個步驟，方便你回頭對照原始碼驗證正確性）。

跟 sparsevlm_baseline.py（近似版）最大的不同，是這個檔案把「真正決定
SparseVLM 表現的兩個機制」都做出來了：

  1. 重要性訊號 = LLM decoder 內部真實的「文字 rater token → 視覺 token」
     cross-attention（不是 question embedding 的 cosine 相似度近似）。
  2. 淘汰是在 LLM 的『特定幾層』forward 過程中『漸進式』進行（預設對應
     32 層模型的第 2 / 6 / 15 層，比例上等同深度 6.25% / 18.75% / 46.9%，
     會依你實際模型層數等比縮放），而不是進 LLM 前一次性決定。

演算法四步驟（跟官方碼逐一對應）：
  A. select_text_raters()      對應 score.py 呼叫前、modelling_sparse_llama.py
                                "Select Text Raters, from SparseVLM 3.2" 那段：
                                視覺 token 對文字 token 的相關度（softmax 後
                                取平均），高於平均值的文字 token 被選為
                                「rater」，只在第一個 pruning stage 選一次，
                                之後每個 stage 重複使用同一組 rater。
  B. extract_text_visual_relation() / score_visual_by_attention()
                                對應 score.py 的 attn_postprocess_topk()：
                                把這一層的 attention weights 取多頭平均，
                                抓出 [rater text tokens] 對 [視覺 token] 的
                                那個子矩陣 P（論文 Eq.7 的 P 就是這個矩陣），
                                對 rater 維度取平均，得到每個視覺 token 的
                                重要性分數。
  B'. rank_based_stage_removal() 對應論文 Section 3.2 Eq.(7)（不是 score.py！
                                見下方「論文 vs. 官方程式碼的落差」說明）：
                                N_remove = λ * (L_v − rank(P))，用矩陣 P 的
                                數值秩(SVD)決定這一層要丟幾個視覺 token，
                                秩低（attention pattern 高度線性相關/冗餘）
                                就多丟，秩高就少丟。
  C. dpc_knn_merge()            對應 utils.py 的 cluster_and_merge()：
                                Density-Peak/KNN 密度聚類（Rodriguez & Laio
                                式的 density-peak 概念，用 KNN 距離估局部
                                密度），選密度峰值當群心，其餘 token 依最近
                                群心分群，群內依重要性加權平均merge成一個
                                代表 token。
  D. sparsevlm_prune_step()     對應 modelling_sparse_llama.py 這一段：
                                B' 算出要丟幾個 → Top-K 硬保留其餘 → 丟棄
                                候選中依分數篩前 30%（score.py 的 0.3）當
                                merge 候選、丟掉後 70%（視為純雜訊）→
                                merge 候選聚類成 N/10+1 群（score.py 的
                                /10 + 1）merge 回來 → 保留 token + 群心
                                token 接回原序列。

────────────────────────────────────────────────────────────────────────────
論文 vs. 官方程式碼的落差（實作這個檔案時發現，如實記錄）：

論文 Section 3.2 / Eq.(7) 描述的是上面 B' 這個「用 attention 矩陣的數值秩
決定每層剔除數量」的一般化、資料自適應公式，理論上不需要為特定 token 數/
特定 budget 校準。但官方 repo 的 score.py 實際上**沒有實作這條公式**，改用
一份寫死的查表（`sparse_token_dict`），只覆蓋「LLaVA-1.5、初始 576 個視覺
token、目標剛好是 192/128/96/64」這四種設定，論文正文/附錄也沒公布 Eq.7
裡 λ 的實際數值。

這個檔案選擇忠於**論文正文的 Eq.7**（不是官方 repo 的查表），因為 InternVL
的初始視覺 token 數跟你 CLI 的 `--budget` 都是任意值，套官方那份四選一的
查表沒有意義；代價是 λ 沒有「官方答案」可以抄，需要你自己在實驗裡調（下面
rank_based_stage_removal() 的 docstring 有說明怎麼調、調大調小分別代表什麼）。
────────────────────────────────────────────────────────────────────────────

檔案最下面的 SparseVLMPruner 把 A~D 串成多層版本；再下面的「Integration
Notes」說明要接進本專案 InternVL pipeline 實際要動哪裡、代價是什麼
（需要在 pruning 層強制用 eager attention 取得 output_attentions，且要
改到 chunked prefill 迴圈本身），目前還沒有接進 internvl_adapter.py，
是可以直接單獨測試/驗證邏輯的 standalone 模組。
"""
from __future__ import annotations

import torch


# ──────────────────────────────────────────────────────────────────────────
# A. 文字 rater 選擇（只在第一個 pruning stage 做一次）
# ──────────────────────────────────────────────────────────────────────────
def select_text_raters(vis_tokens: torch.Tensor, text_tokens: torch.Tensor) -> torch.Tensor:
    """
    vis_tokens  : [Nv, D]  目前視覺 token 的 hidden state
    text_tokens : [Nt, D]  問題文字 token 的 hidden state（image 之後那段）
    回傳: rater_idx [R]，是 text_tokens 裡「值得拿來當重要性訊號來源」的
    文字 token 索引（R <= Nt，通常遠小於 Nt）。

    做法：Nv 個視覺 token 各自對 Nt 個文字 token 算一個 dot-product 相關度、
    在文字維度做 softmax（每個視覺 token 對所有文字 token 的相對關注度和為
    1），再對 Nv 個視覺 token 取平均 → 每個文字 token 得到一個「平均而言被
    多少視覺內容關注」的分數。高於平均值的文字 token 被視為對這張圖有實際
    指涉意義的關鍵詞（例如問題裡的名詞），其餘（介詞、標點等）被濾掉。
    """
    logits = vis_tokens.float() @ text_tokens.float().T        # [Nv, Nt]
    attn_proxy = logits.softmax(dim=-1)                         # 每個視覺 token 對文字的相對關注
    text_relevance = attn_proxy.mean(dim=0)                     # [Nt]
    rater_idx = (text_relevance > text_relevance.mean()).nonzero(as_tuple=True)[0]
    if rater_idx.numel() == 0:
        # 極端情況（例如問題只有 1~2 個字，分數幾乎相同）：至少保留分數最高的 1 個，
        # 避免後面的 attention 抓取拿到空集合。
        rater_idx = text_relevance.topk(1).indices
    return rater_idx


# ──────────────────────────────────────────────────────────────────────────
# B. 用真實 LLM attention 對視覺 token 打分（同時是 Eq.7 的 P 矩陣來源）
# ──────────────────────────────────────────────────────────────────────────
def extract_text_visual_relation(
    attn_weights: torch.Tensor,   # [H, L, L]（單一序列；輸出 attentions 拿到的原始 weights）
    vis_start: int, vis_len: int, text_start: int,
    rater_idx: torch.Tensor,      # 來自 select_text_raters()，相對於 text 區段的索引
    row_offset: int = 0,
) -> torch.Tensor:
    """回傳 [R, vis_len] 的「rater 文字 token → 視覺 token」attention 子矩陣。

    這個矩陣被兩處共用（同一次 attention 抽取，不用算兩次）：
      - score_visual_by_attention() 對 R 取平均 → 每個視覺 token 的重要性分數
      - rank_based_stage_removal() 直接對這個矩陣算數值秩 → 論文 Eq.7 的 P

    attn_weights 必須是「這一層 forward 時、output_attentions=True 拿到的
    原始 attention」——用 SDPA / FlashAttention2 跑的話預設不會吐出這個張量，
    要嘛把這幾層強制切成 eager attn_implementation，要嘛自己在 hook 裡重算
    一次 QK^T softmax（見檔案最下面 Integration Notes 的第 1 點）。

    row_offset：attn_weights 第 0 列對應到完整序列裡的第幾個絕對 query 位置。
    預設 0，代表 attn_weights 是官方作法的完整 [H, L, L] 矩陣（第 0 列就是
    絕對位置 0）。如果呼叫端為了省記憶體，只算了『文字 token query』那幾列
    （例如只有 [H, L_t, L]，第 0 列對應絕對位置 text_start——見
    sparsevlm_internvl.py 的 sparsevlm_prefill()，那裡刻意不算全部 L 個
    query 的 attention 以避免 O(L^2) 記憶體），把 text_start 傳進來當
    row_offset，這裡會自動把 rater_idx 換算成「attn_weights 裡真正的列
    index」，語意不變、只是省了視覺 token 自己那些用不到的 query rows。
    """
    attn = attn_weights.mean(dim=0)                             # 多頭平均 [H, L, L] -> [L, L]（或 [L_t, L]）
    rater_row = rater_idx + text_start - row_offset
    vis_slice = slice(vis_start, vis_start + vis_len)
    return attn[rater_row][:, vis_slice]                        # [R, vis_len]


def score_visual_by_attention(
    attn_weights: torch.Tensor, vis_start: int, vis_len: int, text_start: int, rater_idx: torch.Tensor,
    row_offset: int = 0,
) -> torch.Tensor:
    """回傳每個視覺 token 的重要性分數 [vis_len]（P 對 rater 維度取平均）。"""
    relation = extract_text_visual_relation(attn_weights, vis_start, vis_len, text_start, rater_idx, row_offset)
    return relation.mean(dim=0)                                 # [vis_len]


def extract_full_question_relation(
    attn_weights: torch.Tensor, vis_start: int, vis_len: int, text_start: int,
    row_offset: int = 0,
) -> torch.Tensor:
    """Eq.7 用的 P：**全部**問題文字 token（不限 rater 子集）對視覺 token 的
    attention 矩陣，[L_t, L_v]。

    這裡刻意跟 extract_text_visual_relation() 分開：後者只取 rater 子集
    （通常只有個位數幾個 token）做重要性『排序』平均；如果拿同一個 rater
    子集矩陣去算 Eq.7 的 rank(P)，rank 會被 rater 數量 R 天生封頂
    （rank(P) <= min(R, L_v)，R 常常個位數），導致 N_remove = λ(L_v-rank(P))
    系統性地被低估成『幾乎丟光』（實測：R=6 時第一個 stage 直接從 576 丟到
    24，明顯不合理，遠比論文描述的漸進式 576→300→200→110 激進）。改用全部
    問題文字 token（維度通常有幾十個）當 P 的列，rank(P) 才有機會落在一個
    合理範圍，行為更接近論文描述的『漸進、非一次丟光』。

    這個修正是我實測後回頭發現、自己修的，論文/官方碼都沒有明講 Eq.7 的 P
    到底該用 rater 子集還是全部文字 token——這裡選「全部問題文字 token」
    是因為它在數學上（rank 上界）更合理，不是找到了官方的明確定義。

    row_offset：語意跟 extract_text_visual_relation() 一樣，見該函式說明。
    """
    attn = attn_weights.mean(dim=0)
    vis_slice = slice(vis_start, vis_start + vis_len)
    row_start = text_start - row_offset
    return attn[row_start:, vis_slice]                          # [L_t, vis_len]


# ──────────────────────────────────────────────────────────────────────────
# B'. Rank-based Sparsification Level Adaptation —— 論文 Section 3.2 / Eq.(7)
# ──────────────────────────────────────────────────────────────────────────
def rank_based_stage_removal(P: torch.Tensor, lam: float, energy_threshold: float = 0.99) -> tuple[int, int]:
    """
    N_remove = λ * (L_v − rank(P))

    P : [R, L_v]  extract_text_visual_relation() 回傳的 rater→視覺 attention
        子矩陣（R = rater 文字 token 數，L_v = 這個 stage 開始時的視覺 token 數）。
    lam : 論文沒有公布數值，這裡當成自由參數。直覺：
        lam=0   → 完全不丟（N_remove 恆為 0）
        lam=1   → 丟掉「跟 rank(P) 比起來多出來的」那些視覺 token，也就是
                  假設 attention pattern 的『有效自由度』就是這一層真正需要
                  保留的視覺 token 數
        lam>1   → 比 rank(P) 暗示的還激進，多丟一點
        建議做法：固定其他超參數，對你關心的幾張代表性圖片跑一次，
        微調 lam 讓三個 pruning layer 累積丟完之後的最終 token 數落在
        你想比較的 budget 附近（Eq.7 本質上是自適應、不是精確配額，跟
        target_budget 對不準是預期行為，不是 bug——SparseVLMPruner.step()
        在最後一個 stage 仍然會硬裁切保底，見下方）。
    energy_threshold : 論文沒有給「數值秩」的明確定義/threshold，這裡用數值
        方法裡常見的作法：奇異值平方(能量)累積比例超過這個門檻所需的最少
        奇異值個數，當作 rank(P) 的估計值。threshold 越接近 1 → 算出來的
        rank 越接近 L_v(對雜訊越敏感，n_remove 越保守/越少丟)；threshold
        越低 → rank 估得越小（只認"最主要"的幾個奇異值），n_remove 越大。

    回傳 (n_remove, rank_p)，後者純粹方便你 log/debug 觀察每層实际估到的秩。
    """
    R, L_v = P.shape
    if R < 2 or L_v < 2:
        return 0, L_v
    svals = torch.linalg.svdvals(P.float())
    energy = (svals ** 2).cumsum(0) / (svals ** 2).sum().clamp(min=1e-12)
    rank_p = int((energy < energy_threshold).sum().item()) + 1
    rank_p = min(rank_p, L_v)
    n_remove = lam * (L_v - rank_p)
    n_remove = max(0, min(L_v - 1, int(round(n_remove))))
    return n_remove, rank_p


# ──────────────────────────────────────────────────────────────────────────
# C. Density-Peak/KNN 聚類 + 加權平均 merge
# ──────────────────────────────────────────────────────────────────────────
def dpc_knn_merge(tokens: torch.Tensor, weights: torch.Tensor, num_clusters: int, knn_k: int = 5) -> torch.Tensor:
    """
    tokens  : [N, D]  要被聚類合併的候選 token（已經是「值得留一點痕跡、
              但排不進硬保留名額」的那群，不是隨便什麼 token 都拿來聚類）
    weights : [N]     合併時的加權依據（用 B 步驟算出的重要性分數）
    num_clusters : 要聚成幾群（= merge 完之後這批候選會變成幾個代表 token）
    回傳: merged [num_clusters, D]

    演算法（density-peak clustering 的 KNN 密度版，token merging 文獻的
    常見做法，非這篇論文獨創，這裡的張量運算是我自己重新寫的）：
      1. 用每個點到其 knn_k 個最近鄰的平均距離估「局部密度」（越密集 =
         距離越小 = 密度越高）。
      2. 每個點找「密度比自己高的點裡，距離最近的那一個」，這個最近距離
         當作這個點的 δ；全局密度最高的點沒有比它更高密度的鄰居，δ 設成
         它到任何點的最大距離（保證它一定被選為某個群心）。
      3. peak_score = density * δ 排序，前 num_clusters 名當群心（這樣選出
         的群心彼此密度高、又互相離得夠遠，是 density-peak 的核心概念）。
      4. 其餘所有點（含群心自己）依「離哪個群心最近」分群，群心強制分到
         自己那一群。
      5. 群內用 weights 加權平均，合併成一個代表 token。
    """
    N, D = tokens.shape
    num_clusters = max(1, min(num_clusters, N))
    if num_clusters >= N:
        return tokens.clone()

    x = tokens.float()
    dist = torch.cdist(x, x) / (D ** 0.5)                       # [N, N]，除以 sqrt(D) 貼近官方
                                                                 # cluster_and_merge() 的 distance/(C**0.5)：
                                                                 # 全體距離做同一個常數縮放不會改變「誰是誰的
                                                                 # 最近鄰」，但 density=exp(-距離²) 是非線性的，
                                                                 # 縮放會改變 density 的對比度，連帶可能改變
                                                                 # peak_score=density*δ 選出來的群心——這一步
                                                                 # 之前漏掉，是跟官方行為的真落差，不是無害的
                                                                 # 數值細節。

    k = min(knn_k, N)
    knn_dist, _ = dist.topk(k, largest=False, dim=-1)
    density = torch.exp(-(knn_dist ** 2).mean(dim=-1))
    density = density + torch.rand_like(density) * 1e-6         # 打散同分，避免 tie-breaking 不穩定

    # higher_density[i, j] = True 代表「j 的密度比 i 高」
    higher_density = density.unsqueeze(1) < density.unsqueeze(0)
    masked_dist = dist.masked_fill(~higher_density, dist.max() + 1.0)
    delta, _ = masked_dist.min(dim=-1)
    global_peak = density.argmax()
    delta[global_peak] = dist[global_peak].max()

    peak_score = density * delta
    center_idx = peak_score.topk(num_clusters).indices          # [num_clusters]

    assign = dist[:, center_idx].argmin(dim=-1)                 # [N] 每個 token 分到哪個群心
    assign[center_idx] = torch.arange(num_clusters, device=tokens.device)

    w = weights.float().clamp(min=1e-6)
    merged = tokens.new_zeros(num_clusters, D, dtype=torch.float32)
    weight_sum = tokens.new_zeros(num_clusters, dtype=torch.float32)
    merged.index_add_(0, assign, x * w.unsqueeze(-1))
    weight_sum.index_add_(0, assign, w)
    merged = merged / weight_sum.clamp(min=1e-6).unsqueeze(-1)
    return merged.to(tokens.dtype)


# ──────────────────────────────────────────────────────────────────────────
# D. 單一 pruning stage：硬保留 + 篩選 merge 候選 + 聚類 + 接回序列
# ──────────────────────────────────────────────────────────────────────────
def sparsevlm_prune_step(
    hidden_states: torch.Tensor,   # [L, D]  單一序列（batch=1）目前完整的 hidden states
    attn_weights: torch.Tensor,    # [H, L, L]（或省記憶體版的 [H, L_t, L]，見 row_offset）
    vis_start: int, vis_len: int, text_start: int,
    rater_idx: torch.Tensor,
    lam: float = 1.0,
    rank_energy_threshold: float = 0.99,
    merge_filter_ratio: float = 0.3,
    cluster_divisor: int = 10,
    knn_k: int = 5,
    row_offset: int = 0,
):
    """
    回傳 (new_hidden_states, new_vis_len, debug_stats)。

    hidden_states 的排版固定是 [pre-prompt 文字][視覺 token][問題文字]，
    只有中間的視覺 token 區段會被動；pre-prompt 跟問題文字原封不動接回去
    （對應官方碼永遠把 policy 裡文字部分設成 1，不參與淘汰）。

    這個 stage 要丟幾個視覺 token，由 Eq.7 的 rank_based_stage_removal()
    現場算出來（不是外部傳一個固定 retain_k 進來），因為 rank(P) 本來就要
    看這一層實際的 attention pattern，是資料相依的。

    row_offset：見 extract_text_visual_relation() 的說明。呼叫端如果只算了
    「文字 token query」那幾列的 attn_weights（省掉視覺 token 自己用不到的
    O(L^2) 那塊），把 text_start 傳進來就好。
    """
    vis_tokens = hidden_states[vis_start: vis_start + vis_len]
    text_tokens = hidden_states[text_start:]

    relation = extract_text_visual_relation(attn_weights, vis_start, vis_len, text_start, rater_idx, row_offset)
    importance = relation.mean(dim=0)                          # [vis_len]，rater 子集做重要性排序

    full_relation = extract_full_question_relation(attn_weights, vis_start, vis_len, text_start, row_offset)
    n_remove, rank_p = rank_based_stage_removal(full_relation, lam, rank_energy_threshold)  # 全部文字 token 做 Eq.7 的秩
    debug_stats = {"rank_p": rank_p, "n_remove": n_remove, "vis_len_before": vis_len}

    k = max(0, vis_len - n_remove)
    if k >= vis_len:
        debug_stats["vis_len_after"] = vis_len   # 沒剔除，維持原值——跟下面正常路徑的欄位一致，
                                                   # 呼叫端（例如 auto_tune_lam()）才能穩定讀到這個 key
        return hidden_states, vis_len, debug_stats   # rank(P) 估出來這一層不用剔除

    keep_idx = importance.topk(k).indices if k > 0 else importance.new_zeros(0, dtype=torch.long)
    keep_mask = torch.zeros(vis_len, dtype=torch.bool, device=hidden_states.device)
    keep_mask[keep_idx] = True
    drop_idx = (~keep_mask).nonzero(as_tuple=True)[0]
    kept_tokens = vis_tokens[keep_mask]

    if drop_idx.numel() == 0:
        new_vis_tokens = kept_tokens
    else:
        drop_scores = importance[drop_idx]
        n_filter = min(drop_idx.numel(), int(drop_idx.numel() * merge_filter_ratio) + 1)
        filter_scores, filter_local_idx = drop_scores.topk(n_filter)
        merge_candidates = vis_tokens[drop_idx[filter_local_idx]]

        num_clusters = merge_candidates.shape[0] // cluster_divisor + 1
        merged = dpc_knn_merge(merge_candidates, filter_scores, num_clusters, knn_k=knn_k)
        new_vis_tokens = torch.cat([kept_tokens, merged], dim=0)
        # 篩掉的另外 (1 - merge_filter_ratio) 那批（分數最低、被判定為純雜訊）
        # 直接消失，不進 new_vis_tokens——這一步是硬丟棄，對應官方碼「不進
        # merge_token_stage1」的那些 token。

    new_hidden_states = torch.cat(
        [hidden_states[:vis_start], new_vis_tokens, text_tokens], dim=0
    )
    debug_stats["vis_len_after"] = new_vis_tokens.shape[0]
    return new_hidden_states, new_vis_tokens.shape[0], debug_stats


# ──────────────────────────────────────────────────────────────────────────
# 多層協調器
# ──────────────────────────────────────────────────────────────────────────
class SparseVLMPruner:
    """
    協調 A~D 跨多個 pruning layer 執行。使用方式（虛擬碼，實際要靠 forward
    hook 在對的層數呼叫 step()，見檔案最下面 Integration Notes）：

        pruner = SparseVLMPruner(num_layers=model.config.num_hidden_layers,
                                  target_budget=budget)
        vis_start, vis_len = ...   # image token 在 prompt 裡的起始位置/長度
        text_start = vis_start + vis_len
        for layer_idx in range(num_layers):
            hidden_states, attn = <這一層的 forward，output_attentions=True>
            if layer_idx in pruner.pruning_loc:
                hidden_states, vis_len = pruner.step(
                    layer_idx, hidden_states, attn, vis_start, vis_len, text_start)
                text_start = vis_start + vis_len
    """

    # 論文原始設定：32 層模型（Vicuna-7B/LLaMA）在第 2/6/15 層剔除，
    # 換算成深度比例約 6.25% / 18.75% / 46.9%，用來等比縮放到任意層數的模型。
    _PAPER_PRUNING_LOC_32LAYER = (2, 6, 15)

    def __init__(
        self, num_layers: int, target_budget: int,
        lam: float = 1.0, rank_energy_threshold: float = 0.99,
        merge_filter_ratio: float = 0.3, cluster_divisor: int = 10, knn_k: int = 5,
    ):
        self.pruning_loc = sorted(set(
            max(0, min(num_layers - 1, round(l / 32 * num_layers)))
            for l in self._PAPER_PRUNING_LOC_32LAYER
        ))
        self.num_stages = len(self.pruning_loc)
        self.target_budget = target_budget
        self.lam = lam                                  # Eq.7 的 λ，論文沒公布數值，自己調
        self.rank_energy_threshold = rank_energy_threshold
        self.merge_filter_ratio = merge_filter_ratio
        self.cluster_divisor = cluster_divisor
        self.knn_k = knn_k

        self._rater_idx = None

    def step(self, stage_idx: int, hidden_states, attn_weights, vis_start: int, vis_len: int, text_start: int,
              row_offset: int = 0):
        """回傳 (new_hidden_states, new_vis_len, debug_stats)。

        每個 stage 剔除幾個視覺 token，完全由 Eq.7 現場算（見
        sparsevlm_prune_step / rank_based_stage_removal），不是查預先算好
        的表。stage_idx 目前只用來判斷是不是最後一個 stage（要不要硬裁切
        保底），本身不影響剔除數量的計算。

        row_offset：見 extract_text_visual_relation() 的說明；呼叫端若只算
        了文字 token query 那幾列的 attn_weights（省記憶體），傳 text_start
        進來即可，預設 0 對應「attn_weights 是完整 [H, L, L]」的舊行為。
        """
        vis_tokens = hidden_states[vis_start: vis_start + vis_len]
        text_tokens = hidden_states[text_start:]
        if self._rater_idx is None:
            self._rater_idx = select_text_raters(vis_tokens, text_tokens)

        new_hidden, new_vis_len, debug_stats = sparsevlm_prune_step(
            hidden_states, attn_weights, vis_start, vis_len, text_start,
            rater_idx=self._rater_idx, lam=self.lam,
            rank_energy_threshold=self.rank_energy_threshold,
            merge_filter_ratio=self.merge_filter_ratio,
            cluster_divisor=self.cluster_divisor, knn_k=self.knn_k,
            row_offset=row_offset,
        )

        if stage_idx == self.num_stages - 1 and new_vis_len > self.target_budget:
            # Eq.7 是資料自適應公式，不保證精確命中 target_budget（這是
            # 自適應方法本來就有的性質，不是近似誤差）；最後一個 stage 結束後
            # 硬裁切到 target_budget，確保跟本專案其他 bank 一樣有
            # final_size <= budget 的硬保證，方便公平比較同一個 budget 下的準確率。
            vis_part = new_hidden[vis_start: vis_start + new_vis_len][: self.target_budget]
            new_hidden = torch.cat(
                [new_hidden[:vis_start], vis_part, new_hidden[vis_start + new_vis_len:]], dim=0
            )
            new_vis_len = vis_part.shape[0]
            debug_stats["clipped_to_budget"] = True

        return new_hidden, new_vis_len, debug_stats


# ──────────────────────────────────────────────────────────────────────────
# Integration Notes：要接進本專案 InternVL pipeline，實際要動哪裡
# ──────────────────────────────────────────────────────────────────────────
# 跟 sparsevlm_baseline.py（pre-LLM、ViT 輸出端一次性決定 token 數）不同，
# 這個「真」版本的剔除發生在 LLM decoder 內部、跨好幾層漸進式進行，勢必要
# 動到 internvl_core.py / stream_common.py 的 prefill 邏輯本身，不是換一個
# score_fn 就能插進 TileStreamingMemoryBank。具體要做的事：
#
#   1. output_attentions + eager attention：
#      pruner.pruning_loc 那幾層的 self-attention 必須拿到原始 attention
#      weights。目前 internvl_adapter.py 的 lm_prefill() 直接呼叫
#      model.language_model(...)，走的多半是 SDPA（可能還做了 FlashAttn2），
#      兩者預設都不吐 attention 張量。作法：在 build_model()（見
#      internvl_preprocess.py）載入模型時，對 pruner.pruning_loc 對應的
#      language_model.model.layers[i].self_attn 個別替換成 eager 實作
#      （或整個 language_model 用 attn_implementation="eager" 載入，犧牲一點
#      速度換取每一層都能拿到 attention，先求正確性）。
#
#   2. 在 chunked prefill 迴圈中插入剔除：
#      stream_common.py 的 Step C（chunked vision injection）目前是把
#      encode_and_bank() 已經決定好、長度固定的視覺 token 一路灌完。這個
#      「真」版本要求剔除發生在 LLM 內部特定層，代表 vision chunk 灌完、
#      LLM 已經跑到第 2 層時才第一次真正丟 token，後面第 6、15 層再各丟
#      一次——需要把 pruning_loc 對應的層數改成用 forward hook 攔截該層的
#      輸出（hidden_states, attn_weights），呼叫 pruner.step() 拿到縮短後
#      的 hidden_states，再手動把這個縮短後的序列餵給下一層，並同步縮短
#      position_ids、attention_mask。這是目前 adapter 介面（lm_prefill 只
#      回傳 past_key_values，不暴露逐層 hidden_states）沒有的粒度，需要
#      新增一個類似 InternVLAdapter.lm_prefill_with_sparsevlm() 的方法，
#      內部直接呼叫 model.language_model.model.layers[i](...) 逐層跑，取代
#      現在整個 model.language_model(...) 一次呼叫到底的作法。
#
#   3. 文字 rater 的『文字』範圍：
#      InternVL 的 prompt 排版是 [system+使用者前綴][<img>...<img>][問題
#      文字]，跟論文的 LLaVA 排版一致（image 在前、question 在後），vis_start
#      /text_start 可以直接從 internvl_core.py 的 split_prompt_at_vision()
#      算出的 text_before/text_after token 長度取得，不需要额外改 prompt
#      組裝邏輯。
#
# 這三點都是「改 LLM forward 本身」的工程量，會讓 InternVLAdapter 多一條
# 平行的 prefill 路徑，本檔案目前只做到「給定 hidden_states + attn_weights，
# 正確跑出 SparseVLM 的剔除/合併結果」這個核心邏輯，還沒有接線。


if __name__ == "__main__":
    # Smoke test：合成資料驗證整條流程 shape 正確、budget 硬保證有效，
    # 不需要真的載入 InternVL/LLaVA 模型。
    torch.manual_seed(0)
    D, H = 32, 4
    pre_len, vis_len0, text_len = 20, 576, 12
    L0 = pre_len + vis_len0 + text_len
    budget = 192

    hidden = torch.randn(L0, D)
    num_layers = 28   # 模擬 InternVL3.5-8B 的 Qwen3 backbone 層數量級
    # lam 沒有官方數值，這裡先用 1.0 當起點；合成的隨機 attn 讓 rank(P) 天生
    # 就很接近滿秩（隨機矩陣沒有真實語意上的冗餘），所以這個 smoke test 主要
    # 是驗證 shape / budget 硬保證，不代表真實圖片上 lam=1.0 剛好合適。
    pruner = SparseVLMPruner(num_layers=num_layers, target_budget=budget, lam=1.0)
    print("pruning_loc:", pruner.pruning_loc)

    vis_start, vis_len, text_start = pre_len, vis_len0, pre_len + vis_len0
    L = L0
    for stage_idx, layer_idx in enumerate(pruner.pruning_loc):
        attn = torch.rand(H, L, L)
        hidden, vis_len, stats = pruner.step(stage_idx, hidden, attn, vis_start, vis_len, text_start)
        text_start = vis_start + vis_len
        L = hidden.shape[0]
        print(f"  layer {layer_idx}: vis_len={vis_len}, total_len={L}, {stats}")

    assert vis_len <= budget, vis_len
    assert hidden.shape[0] == pre_len + vis_len + text_len
    print("OK")
