"""
獨立、不改動任何既有檔案的「VisionZip × InternVL3.5」執行腳本。

參考：
  - 論文  ：VisionZip: Longer is Better but Not Necessary in Vision Language Models
            （arXiv 2412.04467，CVPR 2025）Sec.2.3、Algorithm 1 / 2、Table 8 / 9
  - 官方碼：https://github.com/JIA-Lab-research/VisionZip
            visionzip/clip_encoder.py   → CLIPVisionTower_VisionZip.forward()（選 token 的全部邏輯）
            visionzip/utils.py          → CLIPAttention_forward()（回傳 raw key 的 head 平均當 metric）、
                                          apply_info()（只在 layers[-2] 存 metric）
            visionzip/llava_arch.py     → LLaVA-NeXT 對每個 crop 各自跑一次（見下方「per-tile」）
  實作前實際讀過上面這幾份檔案跟論文原文（PDF 用 pdftotext 抽字，不是看摘要），下面所有
  「照官方」的地方都對得到行號/公式；官方碼沒涵蓋、必須自己決定的地方會逐條列在
  「與官方碼的差異」，不混在一起。

跟 fastv_internvl.py / sparsevlm_internvl.py 走同一套 CLI 與輸出 JSON 格式
（{"references": [...], "candidates": {tag: [...]}, "meta": [...]}），可直接餵給既有的
evaluate_anls.py / evaluate_hrbench.py / evaluate_mmmu.py。

用法範例：
    python visionzip_internvl.py \
        --image img_datasets/img01.jpg --budget 512 --run_standard --run_visionzip --save

【VisionZip 的機制（官方碼逐步對應）】
  1. 在 vision encoder『產生被選用特徵的那一層』（LLaVA：select_layer=-2，取
     hidden_states[-2] / attentions[-2] / layers[-2].metric，三者是同一層）：
       - dominant：CLS query 對所有 patch 的 attention，softmax 之後『對 head 維度 sum』
         （官方 `cls_attention.sum(dim=1)`），取 top-K（Algorithm 1）。
       - contextual：剩下（非 dominant、非 CLS）的 token 均勻切成 target / to_merge
         （`step = max(1, R // M)`，`arange(0, R, step)[:M]` 當 target），用該層 key 的 head
         平均（`raw_key_states.mean(1)`）做 L2 normalize 後的內積，每個 to_merge 指派給最像的
         target（argmax），同一個 target 底下的 to_merge 取平均，再『加』回 target 自己
         （`target_hidden + aggregated_hidden`，注意不是 (t+Σm)/(n+1)）（Algorithm 2）。
  2. 選完 / 合併完的 token 才進 projector（mm_projector）→ LLM。LLM 端完全不動，
     token 在進 LLM 之前就已經變短，所以 position_ids 天然連續、KV cache 每一層一樣長、
     可以直接用 HF 的 generate()（不需要像 FastV / SparseVLM 那樣手動逐層跑 decoder）。
  3. 數量：官方 API 是使用者直接給 dominant / contextual 兩個整數（visionzip(model,
     dominant=54, contextual=10)），沒有『總預算』的概念。論文 Table 8（64=54+10、
     128=108+20、192=162+30）與 Table 9（160=135+25、320=270+50、640=540+100）的
     contextual 佔比全部等於 5/32。
       → [我的推廣，非官方] contextual = round(keep * 5/32)：能重現表裡每一組數字，但對表外
         的 keep（例如每個 tile 只留 20 個）是我用同一比例外插，論文/官方碼沒有給。
  4. per-tile：官方 LLaVA-NeXT 是把所有 crop 疊成一個 batch 丟進 vision tower，topk / 合併都在
     batch 的每一張圖內各自做（clip_encoder.py 的 `topk(dim=1)`、`bmm` 都是 per-image），論文
     Sec.3.1 也寫 LLaVA-NeXT 把圖切成四份加一張縮放原圖。InternVL 的動態切 tile（含 thumbnail）
     同構，所以這裡也是每個 tile 各自用自己的 CLS 做。
       → [我的決定，非官方] --budget 是『總量』，平均分給每個 tile（餘數依序多給前面的 tile）；
         官方是使用者自己給每個 crop 的數字。這是為了跟 FastV / SparseVLM 的 budget 同義。

【VisionZip 以外、屬於本專案共用慣例的部分（不是 VisionZip 的一部分）】
  - 用 HF generate(inputs_embeds=...) 生成、prompt 由 build_prompt/split_prompt_at_vision 組
    （沿用 fastv_internvl.py 的做法，該 helper 會省略 <img> 與 </img> 兩個 token）。
  - KV cache MB 用 estimate_kv_cache_mb（prompt-only），與 fastv / sparsevlm / baseline 同一把尺。

【與官方碼的差異——InternVL 架構逼出來的，如實記錄】
  InternVL 在 ViT 後面還有 pixel_shuffle(0.5)（2×2 個 patch 併成一個 4C 維 token，1024 →
  256 個 token/tile）才進 mlp1。LLM 看到的最小單位是『2×2 group』，不能單獨留下 group 裡的
  一個 patch，所以 VisionZip 的『token』是 pixel_shuffle 之後的 256 個 token。
  (a) 選/合併的位置：在 mlp1『之前』（4C 維特徵）做，選完才進 mlp1。對應論文 Fig.3
      （Vision Encoder → VisionZip → Projector）與官方 LLaVA 版（clip_encoder.py 合併的是
      vision tower 的 hidden_states，之後才 mm_projector）；mlp1 之於 InternVL 就是
      mm_projector 之於 LLaVA。
      （官方 repo 另有 Qwen2_5_VL/ 版本是在 PatchMerger 之後才合併，超參數也不同——dominant
      65% / contextual 5%；那是社群要求加的第二套實作，不是論文的做法，這裡不採用。）
  (a') 2×2 group 的聚合方式：官方 LLaVA 沒有這一步；官方結構最接近的先例是
      Qwen2_5_VL/qwen2_5vl_visionzip.py 第 609-613 行（PatchMerger = 2×2 併接 + MLP，跟
      pixel_shuffle + mlp1 同構），這裡照它：
        · score ：組內 4 個 patch 的 attention 取『平均』（官方 `.view(T//4, -1).mean(-1)`）。
        · metric：組內 4 個 patch 的 key 取『平均』（官方 `.view(H, T//4, 4, d).mean(dim=2)`）。
  (b) CLS token：官方保留 CLS 當 dominant 之一（dominant=54 代表 53 個 top-K + 1 個 CLS，
      見 visionzip/main.py 的 `dominant-1`）。InternVL 本身就把 CLS 丟掉（feats[:, 1:]），
      CLS 沒有對應的 pixel_shuffle group，也進不了 mlp1，所以這裡 CLS 只當『打分的 query』，
      不進 LLM；top-K 的 K = keep - contextual（不扣 CLS 名額），每個 tile 最終仍剛好
      留下 keep 個 token。
  (c) 數值精度：CLS attention 的 softmax、metric normalize / 內積 / argmax、merge 的平均，
      官方在模型 dtype（fp16）算，這裡轉 float32 算完再轉回 bf16——只為避免 bf16 尾數不足導致
      top-K / argmax 平手時亂跳，不改變演算法。
  (d) attention 取得方式：官方靠 HF CLIP 的 output_attentions=True。InternViT 在有 flash_attn
      時不會回傳 attention，所以這裡用 forward pre-hook 抓『目標層 self-attn 的輸入』，
      自己用該層 qkv 權重算 CLS 那一列（[B,H,1,N]，很小）＋ key，數學上就是
      InternAttention._naive_attn 的 softmax((q*scale) @ k^T)。qk_normalization=True 的
      vision tower（InternViT-6B）官方沒有對應行為可抄，這裡直接 raise，不猜。
  (e) 輸出順序：先 dominant（依原始位置遞增）再 contextual，同官方 LLaVA 版。
  (f) 官方 forward() 會在 vision tower 內部直接吐選完的 token；這裡把同一段邏輯寫成
      獨立函式 visionzip_select_tile()，不 monkey-patch InternVL 的 remote code。
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import json
import os
import time

import torch
import torch.nn.functional as F

from internvl_stream import (
    MAX_NEW_TOKENS, build_prompt, clean_answer, generate_answer_standard,
    split_prompt_at_vision, build_model, load_image_tiles,
    DEVICE, measure_peak_memory, dedup_hrbench_open,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
)
# eos token / KV cache MB 估算 / baseline 的 prompt 長度：跟 fastv_internvl.py 用同一份，
# 三個對照方法才是同一把尺。
from sparsevlm_internvl import (
    _get_eos_token_id, estimate_kv_cache_mb, estimate_prompt_token_lengths,
)

# 論文 Table 8 / 9 全部 contextual : total = 5 : 32（見檔頭第 3 點）
CONTEXTUAL_RATIO = 5 / 32


# ──────────────────────────────────────────────────────────────────────────
# ViT：抓「產生被選用特徵那一層」的 CLS attention 與 key
# ──────────────────────────────────────────────────────────────────────────
@contextlib.contextmanager
def _capture_attn_input(attn_module):
    """forward pre-hook：存下該層 self-attn 的輸入（= norm1(hidden_states)，見
    InternVisionEncoderLayer.forward 第 291 行），用完立刻移除 hook。"""
    box = {}

    def _hook(_module, args):
        box["x"] = args[0]

    handle = attn_module.register_forward_pre_hook(_hook)
    try:
        yield box
    finally:
        handle.remove()


def _cls_attention_and_keys(attn_module, x: torch.Tensor):
    """回傳 (cls_attn [B, N-1], key_mean [B, N-1, head_dim])，都已去掉 CLS 那一格。

    對應官方：
      cls_attention_sum = attn_weights[:, :, cls_idx, cls_idx+1:].sum(dim=1)   # clip_encoder.py
      metric            = raw_key_states.mean(1)                               # utils.py
    """
    if attn_module.qk_normalization:
        raise NotImplementedError(
            "vision tower 開了 qk_normalization（InternViT-6B）：官方 VisionZip 沒有這個情況的"
            "對應行為（CLIP 沒有 qk norm），不猜 metric 該用 norm 前還是 norm 後的 key。"
            "InternVL3.5-8B 的 InternViT-300M 是 False，可正常使用。"
        )
    B, N, C = x.shape
    H = attn_module.num_heads
    qkv = attn_module.qkv(x).reshape(B, N, 3, H, C // H).permute(2, 0, 3, 1, 4)
    q, k, _ = qkv.unbind(0)                                              # [B, H, N, d]

    logits = (q[:, :, :1, :].float() * attn_module.scale) @ k.float().transpose(-2, -1)   # [B, H, 1, N]
    attn = logits.softmax(dim=-1)[:, :, 0, 1:]                           # [B, H, N-1]（去掉對 CLS 自己）
    cls_attn = attn.sum(dim=1)                                           # [B, N-1]，head 維度 sum
    key_mean = k.float().mean(dim=1)[:, 1:, :]                           # [B, N-1, d]，head 平均
    return cls_attn, key_mean


# ──────────────────────────────────────────────────────────────────────────
# 單一 tile 的 VisionZip：dominant + contextual（逐行對應 clip_encoder.py）
# ──────────────────────────────────────────────────────────────────────────
def visionzip_select_tile(
    tokens: torch.Tensor, score: torch.Tensor, metric: torch.Tensor,
    keep: int, contextual_ratio: float = CONTEXTUAL_RATIO,
):
    """
    tokens [N, D]（pixel_shuffle 後、mlp1 前的 4C 維特徵）、score [N]（CLS attention）、
    metric [N, dm]（key，已對 2×2 group 取平均）。回傳 (kept [keep, D], dominant_num, contextual_num)。

    順序跟官方一致：先是 dominant（依原始位置由小到大，= 官方 masked_select 的順序），
    再是 contextual（依 target 順序）。
    """
    N = tokens.shape[0]
    if keep >= N:
        return tokens, N, 0

    contextual_num = int(round(keep * contextual_ratio))
    dominant_num = keep - contextual_num
    if dominant_num < 1:
        raise ValueError(f"keep={keep} 太小，沒有名額留給 dominant token")

    # ── Dominant：CLS attention top-K ──
    topk_idx = score.topk(dominant_num).indices
    remain_mask = torch.ones(N, dtype=torch.bool, device=tokens.device)
    remain_mask[topk_idx] = False
    dominant_tokens = tokens[~remain_mask]                                # 原始位置遞增

    if contextual_num == 0:
        return dominant_tokens, dominant_num, 0

    # ── Contextual：剩下的 token 用 key 相似度合併 ──
    metric_f = metric[remain_mask].float()
    hidden_f = tokens[remain_mask].float()
    metric_n = metric_f / metric_f.norm(dim=-1, keepdim=True)

    R = metric_n.shape[0]
    step = max(1, R // contextual_num)
    target_idx = torch.arange(0, R, step, device=tokens.device)[:contextual_num]
    is_target = torch.zeros(R, dtype=torch.bool, device=tokens.device)
    is_target[target_idx] = True

    target_metric = metric_n[target_idx]                                  # [M, dm]
    merge_metric = metric_n[~is_target]                                   # [R-M, dm]
    similarity = merge_metric @ target_metric.T                           # [R-M, M]
    assign = torch.zeros_like(similarity)
    assign.scatter_(1, similarity.argmax(dim=1, keepdim=True), 1.0)
    counts = assign.sum(dim=0).clamp(min=1).unsqueeze(-1)                 # [M, 1]
    aggregated = (assign.T @ hidden_f[~is_target]) / counts               # [M, D]
    contextual_tokens = (hidden_f[target_idx] + aggregated).to(tokens.dtype)

    return torch.cat([dominant_tokens, contextual_tokens], dim=0), dominant_num, contextual_num


def split_budget(budget: int, num_tiles: int, tokens_per_tile: int) -> list[int]:
    """總預算平均分給每個 tile（餘數依序多給前面的 tile），每個 tile 不超過原本的 token 數。"""
    if budget < num_tiles:
        raise ValueError(
            f"budget={budget} < tiles={num_tiles}：VisionZip 每個 tile 至少要留 1 個 token，"
            f"請調大 --budget 或調小 --max_num。")
    base, rem = divmod(budget, num_tiles)
    return [min(base + (1 if i < rem else 0), tokens_per_tile) for i in range(num_tiles)]


@torch.no_grad()
def encode_tiles_visionzip(model, tile_pixel_values: torch.Tensor, keeps: list[int],
                           contextual_ratio: float, dtype):
    """對應 internvl_stream.encode_tile()，差別是在 mlp1 之前插入 VisionZip（論文 Fig.3 /
    官方 LLaVA 版：vision encoder → VisionZip → projector）。
    回傳 (tokens [sum(keeps), D_llm], (dominant_total, contextual_total))。"""
    select_layer = model.select_layer
    vit = model.vision_model
    num_layers = len(vit.encoder.layers)
    layer_idx = num_layers + select_layer      # 產生 hidden_states[select_layer] 的那一層（官方三者同層）
    assert 0 <= layer_idx < num_layers, f"select_layer={select_layer} 不合法"

    with _capture_attn_input(vit.encoder.layers[layer_idx].attn) as box:
        if select_layer == -1:
            feats = vit(pixel_values=tile_pixel_values, output_hidden_states=False,
                        return_dict=True).last_hidden_state
        else:
            feats = vit(pixel_values=tile_pixel_values, output_hidden_states=True,
                        return_dict=True).hidden_states[select_layer]
    cls_attn, key_mean = _cls_attention_and_keys(vit.encoder.layers[layer_idx].attn, box["x"])
    del box

    B = feats.shape[0]
    patches = feats[:, 1:, :]                                             # 去 CLS，跟 encode_tile 一致
    h = w = int(patches.shape[1] ** 0.5)

    def shuffle(t):    # 全部用 model.pixel_shuffle，確保 score / metric / feature 的 2×2 分組完全一致
        t = model.pixel_shuffle(t.reshape(B, h, w, -1), scale_factor=model.downsample_ratio)
        return t.reshape(B, -1, t.shape[-1])

    # 2×2 group 的聚合方式照官方 Qwen2_5_VL/qwen2_5vl_visionzip.py 第 609-613 行：
    #   attn_mean.view(T//4, -1).mean(-1)                       → score：四個 patch 取平均
    #   attn_key.view(H, T//4, 4, d).mean(dim=2)                → metric：四個 patch 的 key 取平均
    # pixel_shuffle 後 channel 是 4 個大小為原 channel 數的區塊依序排列，view(..., 4, c) 即可還原。
    tokens = shuffle(patches)                                             # [B, 256, 4C]
    score = shuffle(cls_attn.unsqueeze(-1)).mean(dim=-1)                  # [B, 256]
    metric = shuffle(key_mean)
    metric = metric.view(B, metric.shape[1], 4, -1).mean(dim=2)           # [B, 256, head_dim]
    del feats, patches, cls_attn, key_mean

    kept, dom_total, ctx_total = [], 0, 0
    for b in range(B):
        k_b, d_b, c_b = visionzip_select_tile(tokens[b], score[b], metric[b], keeps[b], contextual_ratio)
        kept.append(k_b)
        dom_total += d_b
        ctx_total += c_b
    kept = torch.cat(kept, dim=0)                                         # [sum(keeps), 4C]

    return model.mlp1(kept).to(dtype), (dom_total, ctx_total)


# ──────────────────────────────────────────────────────────────────────────
# 頂層：build prompt → ViT + VisionZip → HF generate
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_visionzip(
    model, tokenizer, pixel_values: torch.Tensor, question: str, budget: int,
    contextual_ratio: float = CONTEXTUAL_RATIO,
    vit_batch: int = 4, max_new_tokens: int = MAX_NEW_TOKENS,
):
    """budget = 所有 tile 合計要留給 LLM 的視覺 token 數（跟 fastv / sparsevlm 的 budget 同義）。"""
    device = DEVICE
    dtype = model.dtype
    num_tiles = pixel_values.shape[0]
    tokens_per_tile = model.num_image_token
    raw_vis_tokens = num_tiles * tokens_per_tile

    prompt = build_prompt(tokenizer, model, question, num_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)

    embed_fn = model.language_model.get_input_embeddings()
    ids_before = tokenizer(text_before, return_tensors="pt").input_ids.to(device)
    ids_after = tokenizer(text_after, return_tensors="pt").input_ids.to(device)
    embeds_before = embed_fn(ids_before)
    embeds_after = embed_fn(ids_after)

    keeps = split_budget(budget, num_tiles, tokens_per_tile)
    vision_chunks, dom_total, ctx_total = [], 0, 0
    for i in range(0, num_tiles, vit_batch):
        chunk = pixel_values[i:i + vit_batch].to(device, dtype=dtype)
        toks, (d, c) = encode_tiles_visionzip(
            model, chunk, keeps[i:i + vit_batch], contextual_ratio, dtype)
        vision_chunks.append(toks)
        dom_total += d
        ctx_total += c
        del chunk
    vision_tokens = torch.cat(vision_chunks, dim=0).unsqueeze(0)          # [1, sum(keeps), D_llm]

    inputs_embeds = torch.cat([embeds_before, vision_tokens, embeds_after], dim=1)
    total_len = inputs_embeds.shape[1]

    # prompt-only KV cache：視覺 token 在進 LLM 之前就縮短了，每一層一樣長
    # （跟 fastv / sparsevlm / baseline 用同一個 estimate_kv_cache_mb）。
    num_layers = model.language_model.config.num_hidden_layers
    kv_cache_mb = estimate_kv_cache_mb(model, [total_len] * num_layers)

    eos_id = _get_eos_token_id(tokenizer, model)
    out_ids = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=device),
        do_sample=False, max_new_tokens=max_new_tokens,
        eos_token_id=eos_id, pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id,
        use_cache=True,
    )
    answer = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]

    final_vis_tokens = vision_tokens.shape[1]
    stats = {
        "raw_vis_tokens": raw_vis_tokens,
        "final_vis_tokens": final_vis_tokens,
        "budget": budget,
        "tiles": num_tiles,
        "keep_per_tile": [min(keeps), max(keeps)],
        "dominant_tokens": dom_total,
        "contextual_tokens": ctx_total,
        "compression_ratio": round(raw_vis_tokens / max(final_vis_tokens, 1), 3),
        "kv_cache_mb": round(kv_cache_mb, 2),
    }
    return answer, stats


# ──────────────────────────────────────────────────────────────────────────
# CLI：跟 fastv_internvl.py 同款輸出格式，方便沿用既有評測腳本
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num", type=int, default=48, help="dynamic tile 上限（含縮圖），跟 internvl_stream.py 同語意")
    parser.add_argument("--vit_batch", type=int, default=4)
    parser.add_argument("--budget", type=int, default=512,
                         help="所有 tile 合計保留的視覺 token 數，平均分給每個 tile（LLaVA-NeXT 官方也是"
                              "每個 crop 各自選）；final_vis_tokens 恆等於 min(budget, raw_vis_tokens)")
    parser.add_argument("--contextual_ratio", type=float, default=CONTEXTUAL_RATIO,
                         help="contextual / 每個 tile 保留數；預設 5/32，可重現論文 Table 8/9 的每一組數字")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    parser.add_argument("--run_standard", action="store_true", help="跑未壓縮的 baseline（model.batch_chat）")
    parser.add_argument("--run_visionzip", action="store_true", help="跑 VisionZip")

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--dataset", type=str, default="hrbench", choices=["docvqa", "infovqa", "hrbench"],
                        help="which HF dataset to load when --use_ds is set")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_visionzip.json")
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    # ── 1. 載入模型 ──
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")

    torch.cuda.synchronize(DEVICE)
    print(f"Load model time: {time.time() - t0:.2f} s")

    # ── 2. 載入圖像與問題（跟 fastv_internvl.py 同一套 --use_ds/--dataset 選項）──
    if args.use_ds:
        from process_common import load_docvqa, load_infovqa, load_hrbench
        if args.dataset == "docvqa":
            datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                    num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
        elif args.dataset == "infovqa":
            # InfoVQA validation（HF split 名為 "val"）：同 internvl_stream.py
            datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                    num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
        elif args.dataset == "hrbench":
            datasets = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT,
                                     num_image=args.num_images, prompt_mode=HRBENCH_PROMPT)
            if HRBENCH_PROMPT == "open":
                datasets = dedup_hrbench_open(datasets)
        else:
            return
    else:
        from process_common import load_local
        datasets = load_local(args.image, batch_size=1, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process")

    # ── 3. 讀取舊的 output_results.json（如果存在），支援中斷後繼續跑 ──
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        output_results.setdefault("references", [])
        output_results.setdefault("candidates", {})
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', []))} references done)")
    else:
        output_results = {"references": [], "candidates": {}}

    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
             "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
             "answers": d.get("answers", [])}
            for k, d in enumerate(datasets)
        ]

    tag = f"visionzip_budget={args.budget}"
    output_results["candidates"].setdefault(tag, [])

    # ── 4. 逐張處理：每個 tile 的選擇各自獨立，理論上可以 batch，但為了跟 fastv / sparsevlm 的
    #        peak memory 量測條件一致，固定一次一張 ──
    for i, d in enumerate(datasets):
        need_ref = args.run_standard and len(output_results["references"]) <= i
        need_vz = args.run_visionzip and len(output_results["candidates"][tag]) <= i
        if not (need_ref or need_vz):
            continue   # 斷點續跑：這張兩邊都已經有結果

        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(datasets)}] Processing image...")
        print(f"{'='*70}")

        pixel_values = load_image_tiles(d["image"], input_size=448, max_num=max(1, args.max_num - 1))
        print(f"  tiles = {pixel_values.shape[0]}  raw_vis_tokens = {pixel_values.shape[0] * model.num_image_token}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        baseline_kv_mb = None
        visionzip_kv_mb = None

        try:
            if need_ref:
                num_layers = model.language_model.config.num_hidden_layers
                tb_len, v_len, ta_len = estimate_prompt_token_lengths(
                    tokenizer, model, d["question"], pixel_values.shape[0])
                total_len = tb_len + v_len + ta_len
                baseline_kv_mb = estimate_kv_cache_mb(model, [total_len] * num_layers)

                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("visionzip_script_standard_generate"):
                    ref = generate_answer_standard(
                        model, tokenizer, [pixel_values], [d["question"]],
                        generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                    )
                ref = clean_answer(ref)
                output_results["references"].append(ref[0])
                print(f"  [KV cache, prompt-only] {baseline_kv_mb:.1f} MB "
                      f"({total_len} tokens x {num_layers} layers)")
                print(f"\n[Baseline Answer]\n{ref[0]}")

            if need_vz:
                print(f"\n[VisionZip] dominant + contextual per tile (total keep={args.budget}) ...")
                with measure_peak_memory("visionzip_script_visionzip_generate"):
                    answer, stats = generate_visionzip(
                        model, tokenizer, pixel_values, d["question"],
                        budget=args.budget, contextual_ratio=args.contextual_ratio,
                        vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                    )
                visionzip_kv_mb = stats["kv_cache_mb"]
                print(f"  {stats['raw_vis_tokens']} -> {stats['final_vis_tokens']} tokens "
                      f"(compression={stats['compression_ratio']}x)  keep/tile={stats['keep_per_tile']}  "
                      f"dominant={stats['dominant_tokens']}  contextual={stats['contextual_tokens']}")
                print(f"  [KV cache, prompt-only] {visionzip_kv_mb:.1f} MB")
                output_results["candidates"][tag].append(answer)
                print(f"\n[VisionZip Answer]\n{answer}")

            if baseline_kv_mb is not None and visionzip_kv_mb is not None:
                reduction = 100 * (1 - visionzip_kv_mb / baseline_kv_mb)
                print(f"\n[KV cache, prompt-only] baseline={baseline_kv_mb:.1f} MB  "
                      f"visionzip={visionzip_kv_mb:.1f} MB  ({reduction:.1f}% 減少)")

            del pixel_values

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on image {i}: {e}")
            raise

        gc.collect()
        torch.cuda.empty_cache()

        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(output_results, f, indent=2, ensure_ascii=False)
            print(f"\n[Saved] {args.output_json} updated "
                  f"({len(output_results['candidates'][tag])}/{len(datasets)} images done)")

    torch.cuda.synchronize(DEVICE)
    print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
