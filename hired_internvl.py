"""
獨立、不改動任何既有檔案的「HiRED × InternVL3.5」執行腳本（docs/06 §9 #1，2026-11-08）。

參考：
  - 論文  ：HiRED: Attention-Guided Token Dropping for Efficient Inference of High-Resolution
            Vision-Language Models（arXiv 2408.10945v3，AAAI 2025）Sec.4.2、Algorithm 1、Sec.5.3 / Fig.6 / Table 5
  - 官方碼：https://github.com/hasanar1f/HiRED（commit c5978a5，2025-04-17）
            transformers/src/transformers/models/llava_next/modeling_llava_next.py
              ViT_Attn_Hook.get_attn()        第 46-96 行：q_proj / k_proj 輸出 → softmax(q k^T / sqrt(d))，全部 head
              HiRED_sub_img_budgets()         第 490-523 行：sub-image budget 分配
              HiRED_token_selection()         第 525-566 行：full-image / sub-image 的 token 選擇
              forward()                       第 966-1015 行：vit_top_layer = 0、vit_bottom_layer = 22、16 heads
              pack_HiRED_image_features()     第 789-868 行：依原本的排列套 mask
            accuracy_benchmarks/*.sh：token_budget_rate = 0.2、alpha = 0.5（官方預設）
  實作前實際讀過上面的檔案與論文原文（pdftotext），下面「照官方」的地方都對得到行號；官方碼沒涵蓋、
  必須自己決定的地方列在「與官方碼的差異」。

【HiRED 機制（官方碼逐步對應）】
  Phase 1  budget 分配
    1. N_total = budget；full image（LLaVA-NeXT 的 base image = InternVL 的 thumbnail）拿
       full = min(int(N_total * alpha), 每張圖 token 數)；sub-images 共拿 min(N_total - full, k * 每張圖 token 數)。
    2. full image 在『第一層』（vit_top_layer = 0）的 CLS attention（head 加總）取 top-100 個 patch，做成 0/1 遮罩；
       依 sub-image 的網格把遮罩切開，每塊 sub-image 的分數 = 落在它範圍內的 top-100 patch 數；
       budget_i = int(分數比例 × sub 總預算)，每塊上限為每張圖 token 數（官方 `(ratio * budget).int()`，取 floor，
       餘數直接丟掉，所以實際 token 數可能略少於 budget）。
  Phase 2  token 選擇
    3. 每個 partition（含 full image）用『特徵輸出層』（vit_bottom_layer = 22，即 LLaVA 的 vision_feature_layer=-2）的
       CLS attention（head 加總）取 top-budget_i 個 token。
    4. 選完的 token 依模型原本的排列順序送進 LLM（官方 pack_HiRED_image_features：在 LLaVA-NeXT 原生排列上套 mask）。

【論文與官方碼不一致的地方（照官方碼）】
  論文 Eq.(1) 的 sub-image 分數是範圍內 CLS attention 的『連續值總和』；官方碼是 top-100 patch 的『個數』。
  論文報的數字來自官方碼，所以照官方碼（`--spatial_mode sum` 可切換成論文 Eq.(1)，僅供對照，非預設）。

【與官方碼的差異——InternVL 架構逼出來的，如實記錄】
  (a) 選擇單位：InternVL 在 ViT 後有 pixel_shuffle(0.5)，LLM 看到的最小單位是 2×2 patch group（每塊 256 個 token）。
      token 分數 = 組內 4 個 patch 的 CLS attention 取平均（同 visionzip_internvl.py 的做法；平均或加總不影響 top-k）。
      選完才進 mlp1（mlp1 逐 token 作用，跟官方先 projector 再套 mask 等價）。
  (b) 層號：官方 vit_bottom_layer = 22 是『產生被選用特徵的那一層』（CLIP-L 24 層、feature_layer = -2）。
      InternVL3.5 的 select_layer = -1，所以對應到第 23 層（最後一層）；vit_top_layer = 0 不變。
      InternViT-300M 剛好也是 24 層、16 heads、qk_normalization=False。
  (c) top-100：官方是 24×24 = 576 個 patch 取 100 個（17.4%）。InternViT 每塊 32×32 = 1024 個 patch，
      依同比例取 round(1024 × 100 / 576) = 178 個（`--spatial_topk` 可改）。
  (d) 網格切分：官方把 24×24 直接 view 成 (rows, 24/rows, cols, 24/cols)，只在整除時成立（LLaVA-NeXT 的網格都整除）。
      InternVL 的網格（例如 7×6）不一定整除 32，這裡把 patch (y, x) 指派給 sub-image (⌊y·rows/32⌋, ⌊x·cols/32⌋)；
      整除時與官方完全相同。
  (e) InternVL 的 thumbnail 排在『最後』（官方 base image 在最前），只在 tile 數 > 1 時存在；token 依 InternVL 原生
      順序（sub-tiles 依 row-major，thumbnail 最後）送進 LLM，tile 內依原始位置遞增。
      只有 1 塊 tile（沒有 thumbnail）時：整個 budget 給那一塊，用特徵輸出層 CLS attention 取 top-k
      （同官方 LLaVA-1.5 單 partition 版本的做法）。
  (f) budget：官方是比例（token_budget_rate × 全部 token 數）；這裡 --budget 直接給 token 數，跟 FastV / VisionZip /
      IF-Prune / FoveaTile 同義（KV cache 裡的視覺 token 上限）。
  (g) attention 取得：沿用 visionzip_internvl._cls_attention_and_keys（forward pre-hook 抓該層 attn 輸入，自己用 qkv 權重
      算 CLS 那一列，數學上 = softmax(q·scale @ k^T)），float32 計算。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from process_common import get_image

from foveatile_internvl import (
    MAX_NEW_TOKENS, build_prompt, clean_answer, generate_answer_standard,
    split_prompt_at_vision, build_model, load_image_tiles,
    DEVICE, measure_peak_memory,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
)
from internvl_preprocess import dynamic_grid
from sparsevlm_internvl import _get_eos_token_id, estimate_kv_cache_mb, estimate_prompt_token_lengths
from visionzip_internvl import _capture_attn_input, _cls_attention_and_keys

ALPHA = 0.5                     # 官方 accuracy_benchmarks/*.sh 與論文 Sec.5.3 / Fig.6
TOP_LAYER = 0                   # 官方 vit_top_layer
OFFICIAL_TOPK, OFFICIAL_PATCHES = 100, 576


# ──────────────────────────────────────────────────────────────────────────
# ViT：一次 forward 同時抓第 0 層與特徵輸出層的 CLS attention
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def encode_tiles_with_cls(model, pixel_values, vit_batch, dtype):
    """回傳 tokens [T, 256, 4C]（pixel_shuffle 後、mlp1 前）、top_attn [T, 1024]（第 0 層）、
    bottom_attn [T, 1024]（特徵輸出層），attention 都已去 CLS、head 加總。"""
    vit = model.vision_model
    L = len(vit.encoder.layers)
    feat_layer = L + model.select_layer               # select_layer=-1 → 23
    toks, tops, bots = [], [], []
    for i in range(0, pixel_values.shape[0], vit_batch):
        chunk = pixel_values[i:i + vit_batch].to(DEVICE, dtype=dtype)
        with _capture_attn_input(vit.encoder.layers[TOP_LAYER].attn) as bt, \
                _capture_attn_input(vit.encoder.layers[feat_layer].attn) as bb:
            if model.select_layer == -1:
                feats = vit(pixel_values=chunk, output_hidden_states=False, return_dict=True).last_hidden_state
            else:
                feats = vit(pixel_values=chunk, output_hidden_states=True, return_dict=True).hidden_states[model.select_layer]
        top, _ = _cls_attention_and_keys(vit.encoder.layers[TOP_LAYER].attn, bt["x"])
        bot, _ = _cls_attention_and_keys(vit.encoder.layers[feat_layer].attn, bb["x"])
        B = feats.shape[0]
        patches = feats[:, 1:, :]
        h = w = int(patches.shape[1] ** 0.5)
        t = model.pixel_shuffle(patches.reshape(B, h, w, -1), scale_factor=model.downsample_ratio)
        toks.append(t.reshape(B, -1, t.shape[-1]))
        tops.append(top)
        bots.append(bot)
        del feats, patches, bt, bb, chunk
    return torch.cat(toks), torch.cat(tops), torch.cat(bots), h


def group_scores(model, attn, h):
    """[T, h*h] patch 分數 → [T, 256] token 分數（2×2 組內平均，與 pixel_shuffle 的分組一致）。"""
    T = attn.shape[0]
    s = model.pixel_shuffle(attn.reshape(T, h, h, 1), scale_factor=model.downsample_ratio)
    return s.reshape(T, -1, s.shape[-1]).mean(-1)


# ──────────────────────────────────────────────────────────────────────────
# Phase 1：budget 分配（官方 HiRED_token_selection / HiRED_sub_img_budgets）
# ──────────────────────────────────────────────────────────────────────────
def hired_budgets(full_top_attn, h, cols, rows, budget, per_tile, alpha, spatial_topk, spatial_mode="count"):
    k = cols * rows
    full = min(int(budget * alpha), per_tile)
    sub_total = min(budget - full, k * per_tile)
    if sub_total <= 0:
        return full, [0] * k
    if spatial_mode == "count":       # 官方碼：top-k patch 的 0/1 遮罩
        dist = torch.zeros_like(full_top_attn)
        dist[full_top_attn.topk(spatial_topk).indices] = 1.0
    else:                             # 論文 Eq.(1)：連續值總和（只供對照）
        dist = full_top_attn.clone()
    ys = torch.arange(h, device=dist.device).repeat_interleave(h)
    xs = torch.arange(h, device=dist.device).repeat(h)
    part = (ys * rows // h) * cols + (xs * cols // h)        # 整除時 = 官方 view/permute 的切法
    score = torch.zeros(k, device=dist.device).index_add_(0, part, dist.float())
    ratio = score / score.sum()
    subs = torch.clamp((ratio * sub_total).int(), max=per_tile)   # 官方 `.int()` = floor（非負）
    return full, subs.tolist()


@torch.no_grad()
def generate_hired(model, tokenizer, pixel_values, grid, question, budget, alpha=ALPHA,
                   spatial_topk=None, spatial_mode="count", vit_batch=4, max_new_tokens=MAX_NEW_TOKENS):
    dtype = model.dtype
    T = pixel_values.shape[0]
    per_tile = model.num_image_token
    prompt = build_prompt(tokenizer, model, question, T)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
    embed_fn = model.language_model.get_input_embeddings()
    e_before = embed_fn(tokenizer(text_before, return_tensors="pt").input_ids.to(DEVICE))
    e_after = embed_fn(tokenizer(text_after, return_tensors="pt").input_ids.to(DEVICE))

    tokens, top_attn, bot_attn, h = encode_tiles_with_cls(model, pixel_values, vit_batch, dtype)
    tok_score = group_scores(model, bot_attn, h)                           # [T, 256]
    if spatial_topk is None:
        spatial_topk = round(h * h * OFFICIAL_TOPK / OFFICIAL_PATCHES)

    if T == 1:                                                              # 差異 (e)：沒有 thumbnail
        keeps, full = [min(budget, per_tile)], None
    else:
        cols, rows = grid
        assert cols * rows == T - 1, (grid, T)
        full, subs = hired_budgets(top_attn[T - 1], h, cols, rows, budget, per_tile, alpha, spatial_topk, spatial_mode)
        keeps = subs + [full]                                               # InternVL：thumbnail 在最後

    kept = []
    for t, n in enumerate(keeps):
        if n <= 0:
            continue
        idx = tok_score[t].topk(n).indices.sort().values                    # tile 內依原始位置
        kept.append(tokens[t, idx])
    vision = model.mlp1(torch.cat(kept)).to(dtype).unsqueeze(0)
    inputs_embeds = torch.cat([e_before, vision, e_after], dim=1)
    kv_cache_mb = estimate_kv_cache_mb(model, [inputs_embeds.shape[1]] * model.language_model.config.num_hidden_layers)

    eos_id = _get_eos_token_id(tokenizer, model)
    out_ids = model.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=torch.ones(inputs_embeds.shape[:2], dtype=torch.long, device=DEVICE),
        do_sample=False, max_new_tokens=max_new_tokens, eos_token_id=eos_id,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id, use_cache=True)
    answer = tokenizer.batch_decode(out_ids, skip_special_tokens=True)[0]
    stats = {"raw_vis_tokens": T * per_tile, "final_vis_tokens": vision.shape[1], "budget": budget, "tiles": T,
             "full_budget": full, "sub_budgets": keeps[:-1] if T > 1 else [], "kv_cache_mb": round(kv_cache_mb, 2)}
    return answer, stats


# ──────────────────────────────────────────────────────────────────────────
# CLI：跟 visionzip_internvl.py 同款輸入輸出（可直接接 run_test_suite.sh 與既有評測）
# ──────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    p.add_argument("--max_num", type=int, default=48, help="dynamic tile 上限（含縮圖），同 foveatile_internvl.py")
    p.add_argument("--vit_batch", type=int, default=4)
    p.add_argument("--budget", type=int, default=1536, help="KV cache 裡的視覺 token 上限（floor 之後可能略少）")
    p.add_argument("--alpha", type=float, default=ALPHA, help="full image（thumbnail）的 budget 比例，官方 0.5")
    p.add_argument("--spatial_topk", type=int, default=None, help="Phase 1 的 top-k patch 數；預設依官方 100/576 比例 = 178")
    p.add_argument("--spatial_mode", choices=["count", "sum"], default="count",
                   help="count = 官方碼（預設）；sum = 論文 Eq.(1)，只供對照")
    p.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    p.add_argument("--run_standard", action="store_true")
    p.add_argument("--run_hired", action="store_true")
    p.add_argument("--use_ds", action="store_true")
    p.add_argument("--dataset", type=str, default="hrbench",
                   choices=["docvqa", "infovqa", "hrbench", "mme_rw_lite", "vstar", "realworldqa"])
    p.add_argument("--num_images", type=int, default=None)
    p.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg")
    p.add_argument("--save", action="store_true")
    p.add_argument("--output_json", type=str, default="output_results_hired.json")
    args = p.parse_args()
    if args.num_images is not None and args.num_images < 0:
        args.num_images = None

    dtype = torch.bfloat16
    t0 = time.time()
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    print(f"Load model time: {time.time() - t0:.2f} s")

    if args.use_ds:
        from process_common import load_docvqa, load_infovqa, load_hrbench
        if args.dataset == "docvqa":
            datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                   num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
        elif args.dataset == "infovqa":
            datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val", num_image=args.num_images,
                                    prompt_mode=INFOVQA_PROMPT)
        elif args.dataset == "hrbench":
            datasets = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT, num_image=args.num_images,
                                    prompt_mode=HRBENCH_PROMPT)
        else:
            from foveatile_internvl import load_eval_dataset
            datasets = load_eval_dataset(args.dataset, args.num_images)
    else:
        from process_common import load_local
        datasets = load_local(args.image, batch_size=1, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process")

    if args.save and os.path.exists(args.output_json):
        out = json.load(open(args.output_json, encoding="utf-8"))
        out.setdefault("references", []); out.setdefault("candidates", {})
    else:
        out = {"references": [], "candidates": {}}
    if datasets and "answer" in datasets[0]:
        out["meta"] = [{"answer": d.get("answer", ""), "options": d.get("options", []),
                        "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
                        "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
                        "answers": d.get("answers", [])} for k, d in enumerate(datasets)]
    tag = f"hired_budget={args.budget}"
    if args.alpha != ALPHA: tag += f"_a{args.alpha}"
    if args.spatial_topk is not None: tag += f"_k{args.spatial_topk}"
    if args.spatial_mode != "count": tag += f"_{args.spatial_mode}"
    out["candidates"].setdefault(tag, [])
    out.setdefault("stats", {}).setdefault(tag, [])

    tile_max = max(1, args.max_num - 1)
    for i, d in enumerate(datasets):
        need_ref = args.run_standard and len(out["references"]) <= i
        need_h = args.run_hired and len(out["candidates"][tag]) <= i
        if not (need_ref or need_h):
            continue
        img = get_image(d)
        pixel_values = load_image_tiles(img, input_size=448, max_num=tile_max)
        grid = dynamic_grid(img, 1, tile_max, 448)
        torch.cuda.reset_peak_memory_stats(DEVICE)
        if need_ref:
            with measure_peak_memory("hired_script_standard_generate"):
                ref = generate_answer_standard(model, tokenizer, [pixel_values], [d["question"]],
                                               generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False))
            out["references"].append(clean_answer(ref)[0])
        if need_h:
            with measure_peak_memory("hired_script_hired_generate"):
                answer, stats = generate_hired(model, tokenizer, pixel_values, grid, d["question"], args.budget,
                                               args.alpha, args.spatial_topk, args.spatial_mode, args.vit_batch,
                                               args.max_new_tokens)
            out["candidates"][tag].append(answer)
            out["stats"][tag].append({k: stats[k] for k in ("tiles", "final_vis_tokens", "full_budget", "kv_cache_mb")})
            print(f"[{i+1}/{len(datasets)}] tiles={stats['tiles']} grid={grid} -> {stats['final_vis_tokens']} tokens "
                  f"(full={stats['full_budget']}, subs={stats['sub_budgets']})  answer={answer!r}")
        del pixel_values
        gc.collect(); torch.cuda.empty_cache()
        if args.save:
            json.dump(out, open(args.output_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"All done. Total time: {time.time() - t0:.2f} s")


if __name__ == "__main__":
    main()
