"""
FastV「忠於官方原始碼」核心演算法（純函式，不含任何 InternVL/KV cache 工程）。

參考對象：
  - 論文：Chen et al., "An Image is Worth 1/2 Tokens After Layer 2: Plug-and-Play
    Inference Acceleration for Large Vision-Language Models", ECCV 2024
    (arXiv:2403.06764)
  - 官方原始碼：https://github.com/pkunlp-icler/FastV
    實際抓了官方 repo 兩份 fork 的 modeling_llama.py 讀完才寫這個檔案，不是憑
    論文文字或印象重寫：

      (a) src/transformers/src/transformers/models/llama/modeling_llama.py
          第 717~811 行，`LlamaModel.forward()` 裡 `elif USE_FAST_V:` 那個分支
          （"FastV Token Rerank, Attention Mask Implementation"）。這是官方所有
          accuracy 評測腳本（inference_ocrvqa.py 等，搭配
          eval_ocrvqa_fastv_token_mask.sh）實際使用的版本：靠把被丟的 image
          token 的 attention mask 設成 False 來『等效模擬』剪枝效果，tensor
          shape 全程不變，所以能正常支援 KV cache + model.generate() 多步
          decode。

      (b) src/FastV/llava-hf/transformers/src/transformers/models/llama/
          modeling_llama.py 第 955~1105 行，`LlamaModel.fastv_forward()` 裡
          `layer_idx==FASTV_k` 那個分支。這才是真正「縮短」hidden_states/
          position_ids 的版本（會實際省 FLOPs/記憶體），但官方自己在
          inference_aokvqa.py 的 `--fast-v-inplace` 這個 CLI 參數的 help
          字串裡承認「whether to use fast-v inplace version to check real
          latency, no supported kv cache yet」——對應腳本
          eval_aokvqa_latency_fastv_inplace.sh 呼叫 model.generate() 時也是
          用 use_cache=False, max_new_tokens=1（AOKVQA 是選擇題，答案只有一個
          字母，不需要真的多步 decode 就能量測 accuracy/latency），沒有一個
          官方腳本拿它做過『多個 token 的自迴歸生成』。

    (a)(b) 兩份 fork 的『剪枝決策』本身（要丟哪些 image token）逐行比對後
    完全一致，這是這篇論文真正的算法貢獻，也是這個檔案要忠實重現的部分：

        last_layer_attention_avg = torch.mean(last_layer_attention, dim=1)[0]
        last_layer_attention_avg_last_tok = last_layer_attention_avg[-1]
        last_layer_attention_avg_last_tok_image = last_layer_attention_avg_last_tok[
            SYS_LENGTH : SYS_LENGTH + IMAGE_TOKEN_LENGTH]
        top_attention_rank_index = last_layer_attention_avg_last_tok_image.topk(
            ATTENTION_RANK).indices + SYS_LENGTH
        keep_indexs = torch.cat((
            torch.arange(SYS_LENGTH),
            top_attention_rank_index,
            torch.arange(SYS_LENGTH + IMAGE_TOKEN_LENGTH, seq_length_with_past)))
        keep_indexs = keep_indexs.sort().values

    只用『序列裡最後一個 query 位置』這一列 attention（`[-1]`），不是對多個
    文字 token 的 attention 取平均——這點兩份官方 fork 的寫法完全一樣、沒有
    例外，所以照實作，不自己加一個「對多個 query 取平均比較穩」的版本
    （那會是 SparseVLM 的做法，不是 FastV 的）。

  官方超參數命名 → 這個檔案的命名對照（沿用同樣的語意，方便回頭對照原始碼）：
    SYS_LENGTH          -> vis_start   （image token 之前的文字長度）
    IMAGE_TOKEN_LENGTH  -> vis_len     （image token 總數）
    AGG_LAYER / FASTV_k -> agg_layer   （剪枝發生在第幾層，官方預設 2）
    ATTENTION_RANK      -> keep_count  （保留幾個 image token；官方腳本
                            eval_ocrvqa_fastv_token_mask.sh 直接給定整數，
                            論文正文用比例 R 描述，keep_count = round(
                            IMAGE_TOKEN_LENGTH * (1 - R))，兩者等價）

跟 SparseVLM（sparsevlm_core.py）的核心差異，如實記錄，不是這裡簡化的結果：
  - SparseVLM：3 個 pruning layer，每層用 rank(P)（SVD 數值秩）自適應決定
    丟幾個 token，還有 DPC-KNN 把丟棄候選聚類合併回去。
  - FastV：只在『一』層剪枝（預設第 2 層），一次性 top-K 保留，被丟的
    image token 直接消失，沒有 merge 步驟，keep_count 是使用者直接指定的
    整數（不像 SparseVLM 的 λ 需要另外校準），這是論文本身的設計。
"""
from __future__ import annotations

import torch


def fastv_select_keep_indices(
    attn_weights_last_query: torch.Tensor,   # [H, L]：剪枝發生那一層的「上一層」
                                              # 對「最後一個 query 位置」的 attention
                                              # （尚未跨 head 平均），H = head 數，
                                              # L = 剪枝前的完整序列長度
    vis_start: int,
    vis_len: int,
    total_len: int,
    keep_count: int,
    device,
) -> torch.Tensor:
    """回傳排序後的 keep_indexs（絕對 index，相對「剪枝前」的完整序列）。

    逐行對應官方碼（見檔案開頭列的兩個連結/行號）：
        attn_avg = mean over heads                      -> last_layer_attention_avg
        attn_avg[-1] 已經在呼叫端算好才傳進來             -> ...last_tok
        attn_avg[vis_start:vis_start+vis_len]            -> ...last_tok_image
        .topk(keep_count).indices + vis_start            -> top_attention_rank_index
        torch.cat([prefix, kept_image, suffix]).sort()   -> keep_indexs

    attn_weights_last_query 只需要『一個 query 位置』對『全部 key』的
    attention（[H, L]），不是完整 [H, L, L] 矩陣——呼叫端（見
    fastv_internvl.py 的 fastv_prefill）只手算這一列，不需要材料化整個
    attention matrix，比 SparseVLM 對多個文字 query 列打分還更省。

    keep_count 會被 clamp 到 [0, vis_len]：官方碼假設 ATTENTION_RANK <=
    IMAGE_TOKEN_LENGTH（所有測試設定都滿足，見 eval 腳本的 rank_list），這裡
    多做一層防呆，避免呼叫端傳出界的 budget 時直接在 .topk() 拋例外。
    """
    attn_avg = attn_weights_last_query.mean(dim=0)                # [L]，跨 head 平均
    image_scores = attn_avg[vis_start: vis_start + vis_len]       # [vis_len]

    keep_count = max(0, min(vis_len, keep_count))
    top_idx = image_scores.topk(keep_count).indices + vis_start   # [keep_count]

    keep_indices = torch.cat([
        torch.arange(vis_start, device=device),
        top_idx,
        torch.arange(vis_start + vis_len, total_len, device=device),
    ])
    return keep_indices.sort().values


if __name__ == "__main__":
    # Smoke test：合成資料驗證 shape / budget 硬保證，不需要真的載入模型。
    torch.manual_seed(0)
    H, pre_len, vis_len0, post_len = 4, 36, 576, 12
    total_len = pre_len + vis_len0 + post_len
    keep_count = 288   # 官方預設 R=0.5 -> keep = 576*(1-0.5)

    attn_last_row = torch.rand(H, total_len)
    keep_idx = fastv_select_keep_indices(
        attn_last_row, vis_start=pre_len, vis_len=vis_len0, total_len=total_len,
        keep_count=keep_count, device=attn_last_row.device,
    )
    assert keep_idx.shape[0] == pre_len + keep_count + post_len
    assert torch.equal(keep_idx, keep_idx.sort().values)
    assert torch.equal(keep_idx[:pre_len], torch.arange(pre_len))
    assert torch.equal(keep_idx[-post_len:], torch.arange(pre_len + vis_len0, total_len))
    print("OK", keep_idx.shape)
