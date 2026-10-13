"""
SVMA V2 結果彙整成「投影片那張表」：每個 budget 一段，列 = 設定。

  IF-Prune    : 投影片上的數字（只有 b1024 / 2048 / 3072）
  Orig SVMA   : 舊單階段（info_density + thumbnail prior L29-36:last），
                取自 results/svma_v1/internvl_svma_<dataset>/out_block_short_<model>_<dataset>_b<B>.json 與該資料夾的 metrics csv
  S=<S>       : 邊界打分，tag budget=<B>_svma_S<S>[_qa<L>ctx]（out_v2_<model>_<dataset>_b<B>.json）

ANLS 直接從結果 json 現算；Δ 是對 Orig SVMA 的 paired bootstrap（題目順序相同時才算）。
TTFT(bench) = bench_metrics.py 的 metrics_v2_*_S<S>[_qa..].csv（小 n，跟其他方法同一套量法）；
TTFT(full)  = 結果 json 裡逐張 TTFT 的平均（全量，略過第 1 張 warmup）。

用法：
    python summarize_svma_v2.py --budgets 1024,2048,3072,4096,5120 --s_list 0,12,auto
"""
from __future__ import annotations

import argparse
import csv
import json
import os

IFP = {1024: (70.74, 1.02), 2048: (73.43, 1.18), 3072: (75.08, 1.32)}   # 投影片上的 IF-Prune


def _scores(preds, meta):
    from evaluate_anls import calculate_single_anls
    return [calculate_single_anls(p, m.get("answers") or [m.get("answer", "")]) for p, m in zip(preds, meta)]


def load_results(path: str) -> dict:
    if not os.path.exists(path):
        return {}
    return json.load(open(path, encoding="utf-8"))


def bench_ttft(metric_csv: str) -> dict:
    """{budget: (ttft_s_mean, peak_alloc_gb_mean)}，讀 bench_metrics.py 的 stream_b<B> 列。"""
    if not os.path.exists(metric_csv):
        return {}
    out = {}
    with open(metric_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            m = r.get("method", "")
            if m.startswith("stream_b") and r.get("ttft_s_mean"):
                out[int(m[len("stream_b"):])] = (float(r["ttft_s_mean"]), float(r.get("peak_alloc_gb_mean") or "nan"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_tag", default="InternVL3_5-8B")
    ap.add_argument("--dataset", default="infovqa")
    ap.add_argument("--budgets", default="1024,2048,3072,4096,5120")
    ap.add_argument("--s_list", default="0,12,auto")
    ap.add_argument("--qa", default="20ctx", help="q_attn 的 tag 後綴（例：20ctx）")
    ap.add_argument("--outbase", default=None,
                    help="結果 json 的前綴（<outbase>_b<B>.json）；預設 out_v2_<model>_<dataset>（全量）")
    ap.add_argument("--out_csv", default=None, help="預設 table_v2_<model>_<dataset>.csv")
    args = ap.parse_args()

    from svma_config import default_qattn_tiles
    from evaluate_anls import paired_bootstrap_delta

    budgets = [int(b) for b in args.budgets.split(",") if b]
    outbase = args.outbase or f"out_v2_{args.model_tag}_{args.dataset}"
    mbase = f"metrics_v2_{args.model_tag}_{args.dataset}"
    old_dir = f"results/svma_v1/internvl_svma_{args.dataset}"
    old_metrics = bench_ttft(f"{old_dir}/metrics_{args.model_tag}_{args.dataset}.csv")
    fmt = lambda x, nd: "—" if x is None else f"{x:.{nd}f}"

    header = ["budget", "設定", "ANLS", "vs Orig SVMA (95% CI)", "TTFT(bench)", "TTFT(full)", "peak GB"]
    table = []
    for B in budgets:
        d = load_results(f"{outbase}_b{B}.json")
        meta = d.get("meta", [])
        old = load_results(f"{old_dir}/out_block_short_{args.model_tag}_{args.dataset}_b{B}.json")
        orig_tag = f"budget={B}_info_density_delay0_thumbattn"
        # Orig SVMA 依題目 id 對齊到這次的題目集合（全量或隨機抽樣的子集都可以）
        orig = None
        if old and orig_tag in old.get("candidates", {}):
            o_all = _scores(old["candidates"][orig_tag], old["meta"])
            pos = {str(m.get("id")): i for i, m in enumerate(old["meta"])}
            ids = [str(m.get("id")) for m in meta] if meta else list(pos)
            if all(i in pos for i in ids):
                orig = [o_all[pos[i]] for i in ids]
        same_order = orig is not None and bool(meta)

        rows = []
        if B in IFP:
            rows.append(["IF-Prune", fmt(IFP[B][0], 2), "—", fmt(IFP[B][1], 2), "—", "—"])
        if orig:
            ot = old_metrics.get(B)
            rows.append(["Orig SVMA", fmt(100 * sum(orig) / len(orig), 2), "—",
                         fmt(ot and ot[0], 3), "—", fmt(ot and ot[1], 2)])
        s_vals = sorted({default_qattn_tiles(B) if s == "auto" else int(s)
                         for s in args.s_list.split(",") if s})
        for S in s_vals:
            tag = f"budget={B}_svma_S{S}" + (f"_qa{args.qa}" if S > 0 else "")
            preds = d.get("candidates", {}).get(tag, [])
            mname = f"S{S}" + (f"_qa{args.qa}" if S > 0 else "")
            bt = bench_ttft(f"{mbase}_{mname}.csv").get(B)
            if not preds or len(preds) != len(meta):
                rows.append([f"S={S}" + (" (預設)" if S == default_qattn_tiles(B) else ""),
                             f"({len(preds)}/{len(meta)})" if preds else "—", "—",
                             fmt(bt and bt[0], 3), "—", fmt(bt and bt[1], 2)])
                continue
            s = _scores(preds, meta)
            delta = "—"
            if orig and same_order:
                r = paired_bootstrap_delta(s, orig)
                delta = f"{100*r[0]:+.2f} [{100*r[1]:+.2f},{100*r[2]:+.2f}]{'*' if r[1] > 0 or r[2] < 0 else ''}"
            t = [x for x in d.get("ttft", {}).get(tag, [])[1:] if x is not None]
            rows.append([f"S={S}" + (" (預設)" if S == default_qattn_tiles(B) else ""),
                         fmt(100 * sum(s) / len(s), 2), delta, fmt(bt and bt[0], 3),
                         fmt(sum(t) / len(t) if t else None, 3), fmt(bt and bt[1], 2)])
        table += [[f"b{B}"] + r for r in rows]

    widths = [max(len(str(r[i])) for r in [header] + table) for i in range(len(header))]
    print(f"\n{args.model_tag} / {args.dataset}   * = 95% CI 不含 0；TTFT(bench) 是小 n 的 bench_metrics，"
          f"Orig SVMA 的 TTFT 來自舊量測")
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    prev = None
    for r in table:
        if prev and r[0] != prev:
            print()
        print("  ".join(str(c).ljust(w) for c, w in zip(r, widths)))
        prev = r[0]

    out_csv = args.out_csv or f"table_v2_{args.model_tag}_{args.dataset}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(table)
    print(f"\n[saved] {out_csv}")


if __name__ == "__main__":
    main()
