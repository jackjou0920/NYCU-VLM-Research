"""
主張指標 harness：量「peak memory / TTFT vs budget」，未壓縮 vs streaming 對照。

論文的賣點是「省 peak memory、TTFT 不掉太多」，但 accuracy 腳本量不到這個。這支
獨立跑一小批圖（預設 24 張就夠，指標不需要大 n），對每張圖分別量：

  uncompressed  ：batch_chat 路徑，max_new_tokens=1 → TTFT；prefill 當下的 peak
                  allocated / reserved。budget 無關（吃滿 --max_num 個 tile）。
  stream b<K>   ：run_online_kv_with_memory_bank，回傳的 timing["TTFT"] /
                  timing["Vision+OnlineKV"] / timing["_peak_mem_gb"]["OVERALL"]。

每個 (model, dataset, method) 匯總 mean±std 到一份 CSV，直接貼 PPT / 畫圖。

用法：
  python bench_metrics.py --model_name OpenGVLab/InternVL3_5-8B --dataset hrbench \
      --num_images 24 --budgets 1024,2048,4096 \
      --thumb_attn --out_csv metrics_internvl8b_hrbench4k.csv

  （HR-Bench split / prompt、DocVQA prompt 沒有旗標，改 internvl_stream.py 頂端的
   HRBENCH_SPLIT / HRBENCH_PROMPT / DOCVQA_PROMPT 常數）
"""
import gc
import csv
import time
import argparse
import statistics as st

import torch

from internvl_preprocess import build_model, load_image_tiles
import internvl_stream as S
from process_common import load_hrbench, load_mmmu, load_docvqa


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize(S.DEVICE)


def _peak_reset():
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(S.DEVICE)


def _peak_alloc_gb():
    _sync()
    return torch.cuda.max_memory_allocated(S.DEVICE) / 1e9 if torch.cuda.is_available() else 0.0


def _peak_resv_gb():
    return torch.cuda.max_memory_reserved(S.DEVICE) / 1e9 if torch.cuda.is_available() else 0.0


def _load_dataset(name, n):
    if name == "hrbench":
        return load_hrbench(dataset="DreamMr/HR-Bench", split=S.HRBENCH_SPLIT,
                            num_image=n, prompt_mode=S.HRBENCH_PROMPT)
    if name == "mmmu":
        return load_mmmu(split="validation", num_image=n)
    if name == "docvqa":
        return load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                            num_image=n, prompt_mode=S.DOCVQA_PROMPT)
    raise ValueError(name)


def _agg(rows, keys):
    """rows: list[dict] → {key: (mean, std)} for the numeric keys."""
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = (None, None)
        elif len(vals) == 1:
            out[k] = (vals[0], 0.0)
        else:
            out[k] = (st.mean(vals), st.pstdev(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--dataset", default="hrbench", choices=["hrbench", "mmmu", "docvqa"])
    ap.add_argument("--num_images", type=int, default=24)
    ap.add_argument("--max_num", type=int, default=48, help="dynamic tiles 上限（含縮圖）")
    ap.add_argument("--budgets", default="1024,2048,4096")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--max_new_tokens", type=int, default=8, help="指標只需要量到第一個 token + 幾步 decode")
    ap.add_argument("--score_fn", default="info_density")
    ap.add_argument("--thumb_attn", action="store_true", help="streaming 是否開 thumbnail-attention prior")
    ap.add_argument("--thumb_attn_w", type=float, default=S.THUMB_ATTN_W)
    ap.add_argument("--skip_uncompressed", action="store_true")
    ap.add_argument("--out_csv", default="bench_metrics.csv")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    dtype = torch.bfloat16
    print(f"device={S.DEVICE}  model={args.model_name}  dataset={args.dataset}"
          + (f":{S.HRBENCH_SPLIT}" if args.dataset == "hrbench" else ""))

    _peak_reset()
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=S.DEVICE)
    model_load_peak = _peak_alloc_gb()
    print(f"model loaded, peak_alloc={model_load_peak:.2f} GB  num_image_token={model.num_image_token}")

    ds = _load_dataset(args.dataset, args.num_images)
    max_num_grid = max(1, args.max_num - 1)

    unc_rows, stream_rows = [], {b: [] for b in budgets}
    for i, d in enumerate(ds):
        q = d["question"]
        img = d["image"]
        pv = load_image_tiles(img, input_size=448, max_num=max_num_grid)
        n_tiles = pv.shape[0]
        ow, oh = tuple(img.size)

        # ---- uncompressed（budget 無關）----
        if not args.skip_uncompressed:
            adapter = S.InternVLAdapter(tokenizer, model, budget=max(budgets))
            _peak_reset()
            _sync()
            t0 = time.time()
            S.generate_answer_standard(
                model, tokenizer, [pv], [q],
                generation_config=dict(max_new_tokens=1, do_sample=False),
            )
            _sync()
            unc_rows.append({
                "raw_tiles": n_tiles,
                "ttft_s": time.time() - t0,
                "peak_alloc_gb": _peak_alloc_gb(),
                "peak_resv_gb": _peak_resv_gb(),
                "vision_s": None,
            })
            del adapter
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- streaming @ each budget ----
        for b in budgets:
            adapter = S.InternVLAdapter(tokenizer, model, budget=b)
            _peak_reset()
            _, _, timing = S.run_online_kv_with_memory_bank(
                model, adapter, [pv], [q],
                dtype=dtype, vit_batch=args.vit_batch, chunk_size=args.chunk_size,
                budget=b, score_fn=args.score_fn, tile_agg="max", select="topk",
                max_new_tokens=args.max_new_tokens, delay_tiles=0,
                thumb_attn=args.thumb_attn, thumb_attn_w=args.thumb_attn_w,
                orig_sizes=[(ow, oh)],
            )
            pk = timing.get("_peak_mem_gb", {})
            stream_rows[b].append({
                "raw_tiles": n_tiles,
                "ttft_s": timing.get("TTFT"),
                "peak_alloc_gb": pk.get("OVERALL", _peak_alloc_gb()),
                "peak_resv_gb": _peak_resv_gb(),
                "vision_s": timing.get("Vision+OnlineKV"),
            })
            del adapter
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if (i + 1) % 5 == 0 or i + 1 == len(ds):
            print(f"  [{i + 1}/{len(ds)}] done  (raw_tiles={n_tiles})")

    # ---- 匯總 + 寫 CSV ----
    num_keys = ["raw_tiles", "ttft_s", "peak_alloc_gb", "peak_resv_gb", "vision_s"]
    out_rows = []
    tag = f"model={args.model_name}"
    ds_tag = args.dataset + (f":{S.HRBENCH_SPLIT}" if args.dataset == "hrbench" else "")
    method_note = f"thumb_attn=on(w={args.thumb_attn_w:g})" if args.thumb_attn else "thumb_attn=off"

    def _row(method, agg, extra=""):
        r = {"model": args.model_name, "dataset": ds_tag, "method": method,
             "n_images": len(ds), "model_load_peak_gb": round(model_load_peak, 3),
             "note": (extra or method_note)}
        for k in num_keys:
            m, s = agg[k]
            r[k + "_mean"] = None if m is None else round(m, 4)
            r[k + "_std"] = None if s is None else round(s, 4)
        return r

    if unc_rows:
        out_rows.append(_row("uncompressed", _agg(unc_rows, num_keys), extra="max_num tiles, budget-independent"))
    for b in budgets:
        out_rows.append(_row(f"stream_b{b}", _agg(stream_rows[b], num_keys)))

    fields = ["model", "dataset", "method", "n_images", "model_load_peak_gb", "note"]
    fields += [k + "_mean" for k in num_keys] + [k + "_std" for k in num_keys]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    # stdout 表
    print(f"\n=== metrics  {tag}  {ds_tag}  ({len(ds)} imgs, {method_note}) ===")
    print(f"{'method':<16}{'raw_tiles':>10}{'TTFT s':>12}{'peak_alloc GB':>16}"
          f"{'peak_resv GB':>14}{'vision s':>11}")
    print("-" * 79)
    for r in out_rows:
        def g(k):
            v = r.get(k + "_mean")
            return "   —" if v is None else f"{v:.3f}"
        print(f"{r['method']:<16}{g('raw_tiles'):>10}{g('ttft_s'):>12}"
              f"{g('peak_alloc_gb'):>16}{g('peak_resv_gb'):>14}{g('vision_s'):>11}")
    if unc_rows:
        u = out_rows[0]
        for r in out_rows[1:]:
            du = u.get("peak_alloc_gb_mean") or 0
            dr = r.get("peak_alloc_gb_mean") or 0
            tu = u.get("ttft_s_mean") or 0
            tr = r.get("ttft_s_mean") or 0
            print(f"  {r['method']:<14} Δpeak_alloc={dr - du:+.2f} GB  "
                  f"({(dr - du) / du * 100:+.1f}%)   ΔTTFT={tr - tu:+.3f} s "
                  f"({(tr - tu) / tu * 100:+.1f}%)" if du and tu else "")
    print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
