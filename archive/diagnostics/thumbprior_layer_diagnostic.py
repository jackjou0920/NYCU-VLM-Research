"""
thumbnail_saliency_prior 該讀哪幾層、哪幾列的 attention？（診斷用，不動 internvl_stream.py）

現況（internvl_stream.py InternVLAdapter.thumbnail_saliency_prior）：thumbnail 跑一次
LLM forward，取「最後 THUMB_ATTN_LAYERS=8 層」、「答案位置（[-1] 列）」對 256 個 thumbnail
patch 的 attention → 16x16 saliency → 每個 grid tile 一個 prior，再跟 info_density 以
w=THUMB_ATTN_W(0.9) 混合。

layersweep_tile_diagnostic.py（每塊全解析度 tile 各打一次分）發現：最後幾層的 attention
接近隨機、中段層（~12~24）最有效，而且「問題 token 平均」(q) 比「只看答案位置」(last)
穩定。這裡把同樣的選擇套回 thumbnail prior——只是在『同一次』thumbnail forward 裡讀
不同的層與列，TTFT / peak memory 跟現在的 SVMA 一樣（中段層的變體甚至可以只跑到第
hi 層就停，更便宜；這支診斷為了公平一律跑完整 forward）。

做法：繼承 InternVLAdapter，只覆寫 thumbnail_saliency_prior 的第 3 步（讀哪些層/列），
其他全部沿用 internvl_stream.run_online_kv_with_memory_bank（streaming、info_density、
w=0.9 混合、delay_tiles=0、topk、空間順序 commit、decode）＝ 跟你跑 SVMA 的
run_internvl.sh「1) Ours」完全同一條路徑，所以數字能直接跟 SVMA 比。

variant 格式 "lo-hi:rows"（層 1-indexed、含頭尾）：rows = last（答案位置 [-1]）或
q（text_after 全部位置的平均＝問題 + 作答指示 + assistant 開頭）。
"29-36:last" 就是現在的 SVMA（Qwen3-8B 共 36 層），拿來驗證這支腳本有重現 SVMA：
報表會列它跟 SVMA 原始 log 的 tile 選擇一致率。

用法：
    python thumbprior_layer_diagnostic.py --use_ds --dataset infovqa --num_images 200 \
        --budgets 1024,2048 --reuse_from out_oracle_infovqa_b1024.json \
        --save --output_json out_thumbprior_infovqa.json

    python evaluate_anls.py --input_json out_thumbprior_infovqa.json \
        --baseline_tag thumbprior_L29-36_last_budget=1024

    python thumbprior_layer_diagnostic.py --report_only --output_json out_thumbprior_infovqa.json \
        --svma_log results/svma_v1/internvl_svma_infovqa/output_short_infovqa.log
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

DEFAULT_VARIANTS = "29-36:last,29-36:q,5-12:q,9-16:q,13-20:q,17-24:q,21-28:q,13-24:q,17-24:last"


def parse_variant(v: str):
    rng, rows = v.split(":")
    lo, hi = (int(x) for x in rng.split("-"))
    if rows not in ("last", "q") or lo < 1 or hi < lo:
        raise ValueError(f"bad variant {v!r}")
    return lo, hi, rows


def vname(v: str) -> str:
    lo, hi, rows = parse_variant(v)
    return f"L{lo}-{hi}_{rows}"


def tag_of(v: str, budget: int) -> str:
    return f"thumbprior_{vname(v)}_budget={budget}"


def make_adapter_class():
    from internvl_stream import InternVLAdapter, encode_tile, DEVICE

    class ThumbPriorVariantAdapter(InternVLAdapter):
        """跟 InternVLAdapter.thumbnail_saliency_prior 逐步相同，只有第 3 步改成讀
        self.variant 指定的層/列。每個 variant 都重跑一次 thumbnail forward（不 cache），
        這樣每個 variant 量到的 TTFT 都包含這次 forward，彼此可比。"""

        variant = "29-36:last"

        @torch.no_grad()
        def _thumb_attention_rows(self, model, pixel_values, text_before, text_after, dtype):
            num_tiles = pixel_values.shape[0]
            thumb_px = pixel_values[num_tiles - 1: num_tiles].to(DEVICE, dtype=dtype)
            thumb_tok = encode_tile(model, thumb_px, dtype=dtype)[0]
            n_patch = thumb_tok.shape[0]
            emb_layer = self.get_input_embeddings(model)
            tb = self.tokenizer([text_before], return_tensors="pt").to(DEVICE)
            ta = self.tokenizer([text_after], return_tensors="pt").to(DEVICE)
            inp = torch.cat([emb_layer(tb.input_ids), thumb_tok.unsqueeze(0), emb_layer(ta.input_ids)], dim=1)
            att = torch.ones((1, inp.shape[1]), dtype=torch.long, device=DEVICE)
            thumb_start = tb.input_ids.shape[1]
            q_start = thumb_start + n_patch

            # 同原實作：只在這次 forward 暫時切 eager 拿 attention weights，之後還原
            lm = model.language_model
            base = getattr(lm, "model", None) or lm
            prev_impl = getattr(lm.config, "_attn_implementation", None)
            switched = False
            if prev_impl not in ("eager", "eager_paged", "flex_attention") and hasattr(lm, "set_attn_implementation"):
                lm.set_attn_implementation("eager")
                switched = True
            try:
                out = base(inputs_embeds=inp, attention_mask=att, use_cache=False,
                           output_attentions=True, return_dict=True)
            finally:
                if switched:
                    lm.set_attn_implementation(prev_impl)
            attns = out.attentions
            rows = [a[0, :, q_start:, thumb_start: thumb_start + n_patch].float().mean(dim=0).cpu()
                    for a in attns]                                           # 每層 [n_rows, n_patch]
            del out, attns, inp
            return rows

        @torch.no_grad()
        def thumbnail_saliency_prior(self, model, pixel_values, text_before, text_after, cols, rows, dtype):
            num_tiles = pixel_values.shape[0]
            grid_tiles = cols * rows
            per_layer = self._thumb_attention_rows(model, pixel_values, text_before, text_after, dtype)
            lo, hi, mode = parse_variant(self.variant)
            hi = min(hi, len(per_layer))
            layers = per_layer[lo - 1: hi]
            if mode == "last":
                sal = torch.stack([r[-1] for r in layers], 0).mean(0)
            else:
                sal = torch.stack([r.mean(0) for r in layers], 0).mean(0)
            n_patch = sal.shape[0]
            side = int(round(n_patch ** 0.5))
            # ↓ 以下跟原實作第 3 步後半、第 4 步逐行相同
            sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
            sal = sal.reshape(side, side)
            acc = [0.0] * grid_tiles
            cnt = [0] * grid_tiles
            for pr in range(side):
                for pc in range(side):
                    gc_ = min(cols - 1, int((pc + 0.5) / side * cols))
                    gr = min(rows - 1, int((pr + 0.5) / side * rows))
                    k = gr * cols + gc_
                    acc[k] += float(sal[pr, pc])
                    cnt[k] += 1
            prior = [acc[k] / cnt[k] if cnt[k] else 0.0 for k in range(grid_tiles)]
            lo_p, hi_p = min(prior), max(prior)
            prior = [(p - lo_p) / (hi_p - lo_p + 1e-6) for p in prior]
            d = {k: prior[k] for k in range(grid_tiles)}
            d[num_tiles - 1] = 1.0
            return d

    return ThumbPriorVariantAdapter


def report(output_json: str, svma_log: str | None):
    with open(output_json, encoding="utf-8") as f:
        d = json.load(f)
    sel = d.get("tile_selection", {})
    ttft = d.get("ttft", {})
    print(f"\n{'tag':45s} {'n':>4s} {'TTFT mean':>10s}   （TTFT 略過第 1 張圖：第一次呼叫含 CUDA warmup）")
    for tag in sel:
        t = [x for x in ttft.get(tag, [])[1:] if x is not None]
        print(f"{tag:45s} {len(sel[tag]):4d} {sum(t)/len(t) if t else float('nan'):10.3f}")
    if svma_log:
        from analyze_tile_selection import parse_svma_log
        for tag in sel:
            if "L29-36_last" not in tag:
                continue
            b = int(tag.split("budget=")[1])
            log = parse_svma_log(svma_log, b)
            if not log:
                print(f"[repro] log 裡沒有 b{b} 的 --run_stream 區段，跳過")
                continue
            mine = sel[tag]
            n = min(len(mine), len(log))
            same = sum(sorted(mine[i]) == sorted(log[i][1]) for i in range(n))
            print(f"[repro] {tag}: 跟 SVMA 原始 log 的 tile 選擇完全一致 {same}/{n} = {same/max(n,1):.1%}"
                  "（應接近 100%；差異來自 bf16 / batch_size 不同的數值誤差）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--max_num", type=int, default=48,
                    help="同 internvl_stream.py 的語意：總 tile 數上限（含縮圖），grid 上限 = max_num-1")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--budgets", type=str, default="1024,2048")
    ap.add_argument("--variants", type=str, default=DEFAULT_VARIANTS)
    ap.add_argument("--thumb_attn_w", type=float, default=None, help="預設 = internvl_stream.THUMB_ATTN_W")
    ap.add_argument("--max_new_tokens", type=int, default=None, help="預設 = internvl_stream.MAX_NEW_TOKENS")
    ap.add_argument("--use_ds", action="store_true")
    ap.add_argument("--dataset", type=str, default="infovqa", choices=["docvqa", "infovqa"])
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--reuse_from", type=str, default=None,
                    help="沿用既有 json 的 references / meta（例如 out_oracle_infovqa_b1024.json）")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output_json", type=str, default="out_thumbprior.json")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--svma_log", type=str, default=None)
    args = ap.parse_args()

    if args.report_only:
        report(args.output_json, args.svma_log)
        return

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        parse_variant(v)

    import internvl_stream as S
    from process_common import load_docvqa, load_infovqa

    w = S.THUMB_ATTN_W if args.thumb_attn_w is None else args.thumb_attn_w
    max_new = S.MAX_NEW_TOKENS if args.max_new_tokens is None else args.max_new_tokens
    torch.manual_seed(S.RANDOM_SEED)
    torch.cuda.manual_seed_all(S.RANDOM_SEED)

    dtype = torch.bfloat16
    tokenizer, model = S.build_model(args.model_name, dtype=dtype, device=S.DEVICE)
    n_layers = len(model.language_model.model.layers)
    for v in variants:
        if parse_variant(v)[1] > n_layers:
            raise SystemExit(f"variant {v} 超過模型層數 {n_layers}")
    Adapter = make_adapter_class()
    adapter = Adapter(tokenizer, model, budget=max(budgets))

    if args.dataset == "docvqa":
        datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                num_image=args.num_images, prompt_mode=S.DOCVQA_PROMPT)
    else:
        datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                num_image=args.num_images, prompt_mode=S.INFOVQA_PROMPT)
    combos = [(b, v) for b in budgets for v in variants]
    tags = [tag_of(v, b) for b, v in combos]
    print(f"Found {len(datasets)} image(s); budgets={budgets}; variants={variants}; w={w}")

    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, encoding="utf-8") as f:
            out = json.load(f)
        for k in ("candidates", "tile_selection", "ttft"):
            out.setdefault(k, {})
        out.setdefault("references", [])
    else:
        out = {"references": [], "candidates": {}, "tile_selection": {}, "ttft": {}}
        if args.reuse_from:
            with open(args.reuse_from, encoding="utf-8") as f:
                src = json.load(f)
            out["references"] = src.get("references", [])[: len(datasets)]
            if "meta" in src:
                out["meta"] = src["meta"][: len(datasets)]
    if "meta" not in out:
        out["meta"] = [{"answer": d.get("answer", ""), "answers": d.get("answers", []),
                        "question_type": d.get("question_type", "docvqa"), "id": d.get("id", str(k))}
                       for k, d in enumerate(datasets)]
    out["config"] = {"budgets": budgets, "variants": variants, "thumb_attn_w": w,
                     "dataset": args.dataset, "max_num": args.max_num}
    for t in tags:
        for k in ("candidates", "tile_selection", "ttft"):
            out[k].setdefault(t, [])

    max_num_grid = max(1, args.max_num - 1)      # 同 internvl_stream.main
    t_all = time.time()
    for i, d in enumerate(datasets):
        todo = [(c, t) for c, t in zip(combos, tags) if len(out["candidates"][t]) <= i]
        if not todo:
            continue
        pv = S.load_image_tiles(d["image"], input_size=448, max_num=max_num_grid)
        print(f"\n######## [{i+1}/{len(datasets)}] tiles={pv.shape[0]}  ({len(todo)} configs)")
        for (b, v), t in todo:
            adapter.variant = v
            answers, stats, timing = S.run_online_kv_with_memory_bank(
                model, adapter, [pv], [d["question"]],
                dtype=dtype, vit_batch=args.vit_batch, chunk_size=args.chunk_size, budget=b,
                score_fn="info_density", tile_agg="max", select="topk", max_new_tokens=max_new,
                delay_tiles=0, thumb_attn=True, thumb_attn_w=w,
                orig_sizes=[tuple(d["image"].size)], abs_pos=False,
            )
            out["candidates"][t].append(answers[0])
            out["tile_selection"][t].append(stats[0].get("committed_tile_indices", []))
            out["ttft"][t].append(timing.get("TTFT"))
            print(f"  >> {t}: tiles={stats[0].get('committed_tile_indices')}  TTFT={timing.get('TTFT')}s")
        del pv
        gc.collect()
        torch.cuda.empty_cache()
        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nAll done. Total time: {time.time() - t_all:.1f}s")
    if args.save:
        print(f"Saved {args.output_json}")
        report(args.output_json, None)


if __name__ == "__main__":
    main()
