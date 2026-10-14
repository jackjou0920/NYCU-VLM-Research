"""
彙整 results/runs/<dataset>/*.json（internvl_stream.py 的輸出，一個 JSON 一個 tag）成一張表。

每個 dataset、每個 tag：題數、分數、平均 TTFT（SVMA 才有）、平均 KV cache，以及相對參考 tag
（預設 uncompressed）的配對差與 95% bootstrap CI —— 只在兩邊都有的題目上比，所以 n 不同的 run 也能比。
  分數：docvqa / infovqa 等 → ANLS（evaluate_anls.calculate_single_anls）
        hrbench* → CircularEval（4 種選項排列全對才算對，id // 4 一組）

用法：
  python summarize_results.py                                  # 所有 dataset
  python summarize_results.py --datasets docvqa hrbench_8k --ref svma_b1024
  python summarize_results.py --num_images 500                 # docvqa / infovqa 只看前 500 題
輸出：終端機表格 + <root>/summary.csv
"""
from __future__ import annotations

import os
import csv
import glob
import json
import random
import argparse
import statistics as st

from evaluate_anls import calculate_single_anls
from evaluate_hrbench import canon_pred, gold_letter


def METRIC_OF(dataset: str) -> str:
    """跟 run_svma.sh 的 evaluate() 同一個規則：hrbench* 用 CircularEval，mme_rw* 用一般選擇題 accuracy，
    其他用 ANLS。"""
    if dataset.startswith("hrbench"):
        return "hrbench_circular"
    if dataset.startswith("mme_rw"):
        return "mc_accuracy"
    return "anls"


def unit_scores(data: dict, tag: str, metric: str, num_images: int | None) -> dict:
    """回傳 {unit_id: score}。ANLS：unit = 題目 index；CircularEval：unit = id // 4（4 題全有才算）。"""
    preds, meta = data["candidates"][tag], data["meta"]
    n = min(len(preds), len(meta))
    if metric == "mc_accuracy":     # 每題一個 unit，輸出字母 == 正解字母
        if num_images and num_images > 0:
            n = min(n, num_images)
        return {i: float(canon_pred(str(preds[i]), meta[i], "letter") == gold_letter(meta[i])) for i in range(n)}
    if metric == "anls":
        if num_images and num_images > 0:
            n = min(n, num_images)
        return {i: calculate_single_anls(str(preds[i]), meta[i]["answers"]) for i in range(n)}
    groups = {}
    for i in range(n):
        ok = canon_pred(str(preds[i]), meta[i], "letter") == gold_letter(meta[i])
        groups.setdefault(int(meta[i]["id"]) // 4, []).append(ok)
    return {g: float(all(v)) for g, v in groups.items() if len(v) == 4}


def paired_delta(a: dict, b: dict, n_boot: int = 2000, seed: int = 0):
    """a - b 在共同 unit 上的平均差（百分點）與 95% bootstrap CI；沒有共同 unit 回傳 None。"""
    common = sorted(set(a) & set(b))
    if not common:
        return None
    d = [a[k] - b[k] for k in common]
    rng = random.Random(seed)
    boots = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return 100 * st.mean(d), 100 * boots[int(0.025 * n_boot)], 100 * boots[int(0.975 * n_boot) - 1], len(common)


def tag_order(tag: str):
    """uncompressed → lowres_fill_b* → svma_b* → 其他，同類依 budget 排序。"""
    kind, _, b = tag.rpartition("_b")
    rank = {"": 0, "lowres_fill": 1, "svma": 2}
    if tag == "uncompressed":
        return (0, 0, tag)
    return (rank.get(kind, 3), int(b) if b.isdigit() else 0, tag)


def summarize_dataset(root: str, dataset: str, ref: str, num_images: int | None) -> list[dict]:
    metric = METRIC_OF(dataset)
    runs = {}
    for path in sorted(glob.glob(os.path.join(root, dataset, "*.json"))):
        if path.endswith((".summary.json", ".circ.json")):
            continue
        data = json.load(open(path, encoding="utf-8"))
        for tag in data.get("candidates", {}):
            runs[tag] = (data, unit_scores(data, tag, metric, num_images))
    rows = []
    for tag in sorted(runs, key=tag_order):
        data, sc = runs[tag]
        ttft = [x for x in data.get("ttft", {}).get(tag, []) if x is not None]
        kv = [x for x in data.get("kv_cache_mb", {}).get(tag, []) if x is not None]
        row = {"dataset": dataset, "tag": tag, "metric": metric, "n_units": len(sc),
               "score": round(100 * st.mean(sc.values()), 2) if sc else None,
               "ttft_s": round(st.mean(ttft), 3) if ttft else None,
               "kv_cache_mb": round(st.mean(kv), 1) if kv else None,
               "ref": ref, "delta": None, "ci_lo": None, "ci_hi": None, "n_common": None}
        if ref in runs and tag != ref:
            pd = paired_delta(sc, runs[ref][1])
            if pd:
                row.update(delta=round(pd[0], 2), ci_lo=round(pd[1], 2), ci_hi=round(pd[2], 2), n_common=pd[3])
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default="results/runs")
    ap.add_argument("--datasets", nargs="*", default=None, help="預設 = root 底下所有資料夾")
    ap.add_argument("--ref", default="uncompressed", help="配對比較的參考 tag")
    ap.add_argument("--num_images", type=int, default=-1, help="ANLS 類 dataset 只看前 N 題（-1 = 全部）")
    args = ap.parse_args()

    datasets = args.datasets or sorted(d for d in os.listdir(args.root) if os.path.isdir(os.path.join(args.root, d)))
    all_rows = []
    for ds in datasets:
        rows = summarize_dataset(args.root, ds, args.ref, args.num_images)
        if not rows:
            print(f"\n[{ds}] 沒有結果")
            continue
        all_rows += rows
        print(f"\n[{ds}]  metric={rows[0]['metric']}  Δ = 相對 {args.ref} 的配對差（共同題目上，95% CI；* = 顯著）")
        print(f"  {'tag':22s} {'n':>5s} {'score':>7s} {'Δ':>7s} {'95% CI':>17s} {'TTFT':>7s} {'KV MB':>8s}")
        for r in rows:
            delta = ci = ttft = kv = ""
            if r["delta"] is not None:
                sig = "*" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else ""
                delta = "%+.2f" % r["delta"]
                ci = "[%+.1f, %+.1f]%s" % (r["ci_lo"], r["ci_hi"], sig)
            if r["ttft_s"]:
                ttft = "%.2fs" % r["ttft_s"]
            if r["kv_cache_mb"]:
                kv = "%.0f" % r["kv_cache_mb"]
            print(f"  {r['tag']:22s} {r['n_units']:5d} {r['score']:7.2f} {delta:>7s} {ci:>17s} {ttft:>7s} {kv:>8s}")
    if all_rows:
        out = os.path.join(args.root, "summary.csv")
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0]))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
