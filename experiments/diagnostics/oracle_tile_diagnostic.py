"""
第 0 步診斷：SVMA 在高壓縮比下輸 IF-Prune／FastV，是「thumbnail scoring 不準」
還是「tile 粒度太粗（b1024 扣掉 thumbnail 只剩 3 塊 grid tile）」？

做法：answer-aware oracle tile selection ——「偷看」GT 答案文字，用 teacher-forcing
量每個候選 tile 讓「大模型輸出這句 GT 答案」的 log-prob 提升多少，藉此排序 tile，
取 K 個提交進 prompt，再正常 greedy generate 出真正的答案文字（generate 階段完全
不偷看 GT，之後照樣可以拿去跑 evaluate_anls.py 算 ANLS）。

這是「若 tile 選擇是完美的，SVMA 這個 tile-level 框架的準確率上限在哪裡」，
不是可部署的方法（不能在推論時知道 GT）。跟 internvl_stream.py 裡
`InternVLAdapter.stream_image_to_kv_oracle()` 是同一個精神（那個版本只支援
HR-Bench 的單一字母選擇題答案，這裡把 gold 換成 DocVQA/InfoVQA 的自由文字
答案，用整句 teacher-forced log-prob 排序，不是單一 token 的 argmax）。

讀法：
  oracle ANLS 跟 IF-Prune（86.9 @ DocVQA b1024）差不多       -> 問題在 scoring，
      thumbnail-attention 沒抓到答案位置，優先改善 tile/scoring 機制。
  oracle ANLS 還是遠低於 IF-Prune                            -> 3 塊 grid tile
      本身就裝不下答案（粒度問題），scoring 再準也沒用，要把 tile 拆成更小的
      spatially-contiguous sub-block（仍然保留空間結構，不是逐 token 選）。

────────────────────────────────────────────────────────────────────────────
不動 internvl_stream.py：本檔案是獨立腳本，只 import 既有函式，不改動任何
既有檔案。跟 fastv_internvl.py / ifprune_internvl.py / visionzip_internvl.py
同一套慣例：CLI 介面、output JSON 格式（references / candidates / meta）完全
比照這幾個檔案，跑完可以直接餵給 evaluate_anls.py。

跟 stream_image_to_kv_oracle() 的差異（如實記錄）：
  1) gold 從單一字母換成整句答案文字，用 teacher-forced 整句 log-prob
     （每 token 取 log_softmax 後平均）取代單 token 的 argmax log-prob。
     gold 用 DocVQA/InfoVQA meta 裡的 d["answer"]（=第一個可接受答案，跟
     process_common.py 的 answers[0] 同一個字串）。
  2) per_tile 模式這裡把所有候選 tile 的 scoring forward 批次化（同一個
     text_before/text_after/gold 長度下，vis 長度固定是 2*num_image_token，
     shape 一致可以疊 batch），減少 forward 次數，跑一次資料集的診斷才划算。
  3) 選完 tile 之後，這裡直接建好完整的 inputs_embeds 呼叫
     model.language_model.generate()（跟 visionzip_internvl.py 的
     generate_visionzip() 同一招），不做逐層剪枝也不需要手動 decode
     loop——因為 tile 選擇發生在進 LLM「之前」，跟 FastV/IF-Prune 那種
     「進第 K 層才剪」完全不同，沒有 per-layer cache 長度不一致的問題，
     可以用最單純的 generate() 路徑。
  4) tokenize gold 答案時比照 stream_image_to_kv_oracle 對 gold_letter 的
     處理，用「前面加一個空白」的形式（" " + answer），因為 text_after 是
     chat template 的 assistant 起始（結尾通常沒有空白），BPE tokenizer
     會把這個空白併進答案第一個 token，跟模型自然生成時的 token 化方式一致。

用法範例：
    python oracle_tile_diagnostic.py --use_ds --dataset docvqa --num_images 200 \
        --budget 1024 --oracle_mode per_tile \
        --run_standard --run_oracle --save --output_json out_oracle_docvqa_b1024.json

    python evaluate_anls.py --input_json out_oracle_docvqa_b1024.json \
        --baseline_tag oracle_budget=1024_mode=per_tile \
        --summary_json out_oracle_docvqa_b1024.summary.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

from internvl_stream import (
    MAX_NEW_TOKENS, build_prompt, clean_answer, generate_answer_standard,
    split_prompt_at_vision, encode_tile, build_model, load_image_tiles,
    DEVICE, measure_peak_memory, DOCVQA_PROMPT, INFOVQA_PROMPT, _infer_tile_grid,
)
from sparsevlm_internvl import _get_eos_token_id, estimate_kv_cache_mb


# ──────────────────────────────────────────────────────────────────────────
# Oracle tile ranking：teacher-forced 整句 GT log-prob
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def _batched_gold_logprob(
    model, embeds_before, embeds_after, target_embeds, target_ids, vis_batch, device,
):
    """一次跑一批候選的 {thumbnail, tile} 組合，回傳每個候選的平均 per-token log-prob。

    vis_batch: [B, 2*num_image_token, D]（每個候選各自的 thumbnail+tile pair，
    tile 順序已依 index 排好，跟哪個在前哪個在後對這個 2-tile 短 forward 不影響
    語意，只是要固定，避免同一個 pair 每次算出不同分數）。
    """
    B = vis_batch.shape[0]
    prompt_len = embeds_before.shape[1] + vis_batch.shape[1] + embeds_after.shape[1]
    T = target_ids.shape[-1]

    eb = embeds_before.expand(B, -1, -1)
    ea = embeds_after.expand(B, -1, -1)
    te = target_embeds.expand(B, -1, -1)
    inp = torch.cat([eb, vis_batch, ea, te], dim=1)
    att = torch.ones(inp.shape[:2], dtype=torch.long, device=device)

    out = model.language_model(inputs_embeds=inp, attention_mask=att, use_cache=False, return_dict=True)
    # 預測第 j 個 gold token的 logits 在位置 prompt_len-1+j（teacher forcing）
    logp = torch.log_softmax(out.logits[:, prompt_len - 1: prompt_len - 1 + T, :].float(), dim=-1)
    tgt = target_ids.expand(B, -1)                          # [B, T]
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)  # [B, T]
    scores = tok_lp.mean(dim=-1)                             # [B]，跟句長無關，可跨候選比較
    del out, inp, eb, ea, te, logp, tok_lp
    return scores.tolist()


def oracle_rank_tiles(
    model, tokenizer, tile_blocks, embeds_before, embeds_after,
    protected: int, gold_text: str, budget: int, num_image_token: int,
    device, oracle_mode: str = "per_tile", batch_size: int = 8,
):
    """回傳 (chosen_tiles_sorted, per_tile_scores, K)。

    chosen 一定包含 protected（whole-image thumbnail，跟 SVMA 自己的
    protected_indices 慣例一致，見 internvl_stream.py 的
    TileStreamingMemoryBank(protected_indices=...)）。

    gold_text 為空字串時（理論上 docvqa/infovqa short 模式不會發生，防呆用）
    退化成「按 tile index 由小到大取前 K 個」，並印警告。
    """
    num_tiles = len(tile_blocks)
    K = max(1, min(num_tiles, budget // num_image_token))
    cand = [t for t in range(num_tiles) if t != protected]

    gold_text = (gold_text or "").strip()
    if not gold_text:
        print("  [oracle][WARN] empty gold answer, fallback to index-order top-K (no GT to peek at)")
        chosen = sorted({protected, *cand[: max(0, K - 1)]})
        return chosen, {}, K

    # 見檔頭第 4 點：前面加空白，跟 chat template assistant 段結尾接續生成時的
    # tokenization 一致。
    gold_ids = tokenizer(" " + gold_text, add_special_tokens=False).input_ids
    if not gold_ids:
        print(f"  [oracle][WARN] could not tokenize gold_text={gold_text!r}, fallback to index-order top-K")
        chosen = sorted({protected, *cand[: max(0, K - 1)]})
        return chosen, {}, K

    embed_fn = model.language_model.get_input_embeddings()
    target_ids = torch.tensor(gold_ids, device=device).unsqueeze(0)   # [1, T]
    target_embeds = embed_fn(target_ids)                              # [1, T, D]

    def pair_vis(a, b):
        lo, hi = (a, b) if a < b else (b, a)
        return torch.cat([tile_blocks[lo], tile_blocks[hi]], dim=0)   # [2*num_image_token, D]

    per_tile_score = {}

    if oracle_mode == "per_tile":
        for s in range(0, len(cand), batch_size):
            batch_t = cand[s: s + batch_size]
            vis_batch = torch.stack([pair_vis(protected, t) for t in batch_t], dim=0)
            scores = _batched_gold_logprob(
                model, embeds_before, embeds_after, target_embeds, target_ids, vis_batch, device)
            for t, sc in zip(batch_t, scores):
                per_tile_score[t] = sc
            del vis_batch
        ranked = sorted(cand, key=lambda t: per_tile_score[t], reverse=True)
        chosen = [protected] + ranked[: max(0, K - 1)]

    elif oracle_mode == "greedy":
        # forward-greedy：每一輪把「目前已選 tile + 候選 t」一起餵進去，取最能
        # 拉高 GT log-prob 的候選，處理 tile 之間空間上重疊/互補的情況，同一輪
        # 內的候選 shape 相同（已選 tile 數固定），一樣可以批次化。
        chosen_set = [protected]
        remaining = list(cand)
        while len(chosen_set) < K and remaining:
            round_scores = []
            for s in range(0, len(remaining), batch_size):
                batch_t = remaining[s: s + batch_size]
                vis_batch = torch.stack(
                    [torch.cat([tile_blocks[i] for i in sorted(chosen_set + [t])], dim=0)
                     for t in batch_t], dim=0)
                scores = _batched_gold_logprob(
                    model, embeds_before, embeds_after, target_embeds, target_ids, vis_batch, device)
                round_scores.extend(zip(batch_t, scores))
                del vis_batch
            for t, sc in round_scores:
                per_tile_score.setdefault(t, sc)   # 第一輪的分數當成這個 tile 的代表分數
            best_t, best_sc = max(round_scores, key=lambda x: x[1])
            chosen_set.append(best_t)
            remaining.remove(best_t)
        chosen = chosen_set
    else:
        raise ValueError(f"Unknown oracle_mode={oracle_mode!r}")

    chosen = sorted(set(chosen))[:K] if len(set(chosen)) > K else sorted(set(chosen))
    return chosen, per_tile_score, K


# ──────────────────────────────────────────────────────────────────────────
# 頂層：oracle 選 tile → 正常 generate()（不偷看 GT）
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_oracle(
    model, tokenizer, pixel_values: torch.Tensor, question: str, gold_text: str,
    budget: int, oracle_mode: str = "per_tile", oracle_batch_size: int = 8,
    vit_batch: int = 4, max_new_tokens: int = MAX_NEW_TOKENS,
    image_size: tuple[int, int] = (1, 1), selection_only: bool = False,
):
    device = DEVICE
    dtype = model.dtype
    num_tiles = pixel_values.shape[0]
    num_image_token = model.num_image_token
    raw_vis_tokens = num_tiles * num_image_token

    prompt = build_prompt(tokenizer, model, question, num_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)

    embed_fn = model.language_model.get_input_embeddings()
    ids_before = tokenizer(text_before, return_tensors="pt").input_ids.to(device)
    ids_after = tokenizer(text_after, return_tensors="pt").input_ids.to(device)
    embeds_before = embed_fn(ids_before)
    embeds_after = embed_fn(ids_after)

    # ── 1. encode 每個 tile 一次，之後 oracle scoring 跟最終 commit 都重用 ──
    tile_blocks = []
    for i in range(0, num_tiles, vit_batch):
        chunk = pixel_values[i:i + vit_batch].to(device, dtype=dtype)
        tt = encode_tile(model, chunk, dtype=dtype)
        for j in range(tt.shape[0]):
            tile_blocks.append(tt[j])
        del chunk, tt

    # dynamic_preprocess 把整圖縮圖放在最後一個 tile（同 internvl_stream.py 的慣例，
    # num_tiles==1 時退化成指向唯一的那個 tile，無害）。
    protected = num_tiles - 1

    t0 = time.time()
    chosen, per_tile_score, K = oracle_rank_tiles(
        model, tokenizer, tile_blocks, embeds_before, embeds_after,
        protected, gold_text, budget, num_image_token, device,
        oracle_mode=oracle_mode, batch_size=oracle_batch_size,
    )
    oracle_s = time.time() - t0

    # grid 行列數（給 analyze_tile_selection.py 判斷 tile 之間的空間相鄰關係）；
    # 跟 internvl_stream.py 的 thumbnail prior 用同一個 _infer_tile_grid。
    cols, rows = _infer_tile_grid(max(num_tiles - 1, 1), image_size[0], image_size[1])

    # ── 2. 選定的 tile 依空間（index）順序組成最終 vision 序列，正常 generate ──
    vision_tokens = torch.stack([tile_blocks[t] for t in chosen], dim=0).reshape(1, -1, embeds_before.shape[-1])
    del tile_blocks

    if selection_only:
        # 只要 tile 選擇，不生成答案（省下 generate 時間；candidates 會是空字串，不能拿去算 ANLS）
        stats = {
            "raw_vis_tokens": raw_vis_tokens, "final_vis_tokens": vision_tokens.shape[1],
            "budget": budget, "num_tiles": num_tiles, "K": K, "chosen_tiles": chosen,
            "cols": cols, "rows": rows, "oracle_mode": oracle_mode, "oracle_s": round(oracle_s, 4),
            "compression_ratio": round(raw_vis_tokens / max(vision_tokens.shape[1], 1), 3),
            "tile_scores": {int(t): sc for t, sc in per_tile_score.items()},
        }
        return "", stats

    inputs_embeds = torch.cat([embeds_before, vision_tokens, embeds_after], dim=1)
    total_len = inputs_embeds.shape[1]
    final_vis_tokens = vision_tokens.shape[1]

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

    scores = list(per_tile_score.values())
    stats = {
        "raw_vis_tokens": raw_vis_tokens,
        "final_vis_tokens": final_vis_tokens,
        "budget": budget,
        "num_tiles": num_tiles,
        "K": K,
        "chosen_tiles": chosen,
        "cols": cols,
        "rows": rows,
        "oracle_mode": oracle_mode,
        "oracle_s": round(oracle_s, 4),
        "compression_ratio": round(raw_vis_tokens / max(final_vis_tokens, 1), 3),
        "kv_cache_mb": round(kv_cache_mb, 2),
        "score_min": min(scores) if scores else None,
        "score_max": max(scores) if scores else None,
        "tile_scores": {int(t): sc for t, sc in per_tile_score.items()},
    }
    return answer, stats


# ──────────────────────────────────────────────────────────────────────────
# CLI：跟 fastv_internvl.py / ifprune_internvl.py / visionzip_internvl.py 同款
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num", type=int, default=48, help="max dynamic tiles")
    parser.add_argument("--vit_batch", type=int, default=4)
    parser.add_argument("--budget", type=int, default=1024,
                        help="剪枝後保留的視覺 token 數，K = budget // num_image_token 塊 tile "
                             "（含 thumbnail），跟 run_ifprune.sh/run_fastv.sh 的 budget 同義")
    parser.add_argument("--oracle_mode", choices=["per_tile", "greedy"], default="per_tile",
                        help="per_tile：{thumbnail,tile} 逐一評分，O(num_tiles) forward，批次化過；"
                             "greedy：forward-greedy 逐輪挑最能拉高 GT log-prob 的 tile，"
                             "O(num_tiles*K) forward，處理 tile 間互補/重疊但慢很多")
    parser.add_argument("--oracle_batch_size", type=int, default=8,
                        help="oracle scoring forward 一次疊幾個候選 tile 一起跑")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    parser.add_argument("--run_standard", action="store_true", help="跑未壓縮的 baseline（model.batch_chat）")
    parser.add_argument("--run_oracle", action="store_true", help="跑 answer-aware oracle tile selection")

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--dataset", type=str, default="docvqa", choices=["docvqa", "infovqa"],
                        help="oracle 需要 GT 答案文字，目前只接 docvqa/infovqa（不是選擇題，跟 HR-Bench 的"
                             "單一字母 gold_letter 不同格式）")
    parser.add_argument("--num_images", type=int, default=None,
                        help="Number of images/questions to inference (None means all; 診斷建議先抽 200~300)")
    parser.add_argument("--image", type=str, default=None,
                        help="不用 --use_ds 時的單張圖路徑（診斷用途不建議，沒有 GT 答案可偷看）")
    parser.add_argument("--gold_answer", type=str, default=None,
                        help="搭配 --image 用，手動指定這張圖的 GT 答案文字")
    parser.add_argument("--selection_only", action="store_true",
                        help="只做 tile 選擇、不 generate（candidates 為空字串，不能算 ANLS）；"
                             "選擇結果存在 output json 的 tile_selection，給 analyze_tile_selection.py 用")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_oracle.json")
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    baseline_attn_impl = model.language_model.config._attn_implementation
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
    torch.cuda.synchronize(DEVICE)
    print(f"Load model time: {time.time() - t0:.2f} s")

    if args.use_ds:
        from process_common import load_docvqa, load_infovqa
        if args.dataset == "docvqa":
            datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                    num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
        else:
            datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                    num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
        if DOCVQA_PROMPT != "short" or INFOVQA_PROMPT != "short":
            print("[WARN] oracle 診斷是拿 GT 答案文字算 teacher-forced log-prob，"
                  "prompt_mode 不是 'short' 時 GT 沒意義，結果不能用")
    else:
        if not args.image or not args.gold_answer:
            raise ValueError("不用 --use_ds 時必須同時給 --image 和 --gold_answer")
        from PIL import Image
        datasets = [{"question": "Answer the question using a single word or phrase.",
                     "image": Image.open(args.image).convert("RGB"),
                     "answer": args.gold_answer, "answers": [args.gold_answer],
                     "question_type": "docvqa", "id": "0"}]
    print(f"Found {len(datasets)} image(s) to process")

    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        output_results.setdefault("references", [])
        output_results.setdefault("candidates", {})
        output_results.setdefault("tile_selection", {})
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', {}))} images already done)")
    else:
        output_results = {"references": [], "candidates": {}, "tile_selection": {}}

    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "answers": d.get("answers", []),
             "question_type": d.get("question_type", "docvqa"), "id": d.get("id", str(k))}
            for k, d in enumerate(datasets)
        ]

    tag = f"oracle_budget={args.budget}_mode={args.oracle_mode}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        print(f"\nAll done already. Total time: {time.time() - t0:.2f} s")
        return

    for i, d in enumerate(datasets):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(datasets)}] Processing image...")
        print(f"{'='*70}")

        pixel_values = load_image_tiles(d["image"], input_size=448, max_num=args.max_num)
        print(f"  tiles = {pixel_values.shape[0]}  raw_vis_tokens = {pixel_values.shape[0] * model.num_image_token}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("oracle_script_standard_generate"):
                    ref = generate_answer_standard(
                        model, tokenizer, [pixel_values], [d["question"]],
                        generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                    )
                ref = clean_answer(ref)
                output_results["references"].append(ref[0])
                print(f"\n[Baseline Answer]\n{ref[0]}")

            if args.run_oracle:
                output_results["candidates"].setdefault(tag, [])
                output_results["tile_selection"].setdefault(tag, [])
                print(f"\n[Oracle] answer-aware tile selection (mode={args.oracle_mode}, "
                      f"budget={args.budget}, peeking at GT={d.get('answer', '')!r}) ...")
                with measure_peak_memory("oracle_script_oracle_generate"):
                    answer, stats = generate_oracle(
                        model, tokenizer, pixel_values, d["question"], d.get("answer", ""),
                        budget=args.budget, oracle_mode=args.oracle_mode,
                        oracle_batch_size=args.oracle_batch_size,
                        vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                        image_size=d["image"].size, selection_only=args.selection_only,
                    )
                output_results["tile_selection"][tag].append({
                    "num_tiles": stats["num_tiles"], "cols": stats["cols"], "rows": stats["rows"],
                    "chosen_tiles": stats["chosen_tiles"], "K": stats["K"],
                    "tile_scores": stats["tile_scores"],
                })
                print(f"  {stats['raw_vis_tokens']} -> {stats['final_vis_tokens']} tokens "
                      f"(K={stats['K']}, chosen_tiles={stats['chosen_tiles']}, "
                      f"compression={stats['compression_ratio']}x)  oracle scoring={stats['oracle_s']}s")
                output_results["candidates"][tag].append(answer)
                print(f"\n[Oracle Answer]\n{answer}")

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
                  f"({len(output_results['references'])}/{len(datasets)} images done)")

    torch.cuda.synchronize(DEVICE)
    print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
