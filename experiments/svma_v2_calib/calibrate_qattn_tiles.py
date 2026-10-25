"""
依規則 A 為每個 budget 選 S（--qattn_tiles），寫出 svma_calibration.json（含每個 S 的 ANLS / TTFT 明細）；
加 --update_config 時同時把選定的 S 寫進 svma_config.py 的 QATTN_TILES_BY_BUDGET（auto 的預設表）。

規則 A：在 TTFT 比 IF-Prune 快（TTFT <= 目標 - margin）的 S 裡，選 ANLS 最高的；
        ANLS 差距在 --tol 以內時選比較小（比較快）的 S。都不夠快就選 S=0（最快）並標記。
  - ANLS：校準集 json（隨機抽樣，例：out_calib_..._b<B>.json）裡 budget=<B>_svma_S<S>[_qa..] 的結果
  - TTFT：bench_metrics.py 的 metrics_v2_<model>_<dataset>_S<S>[_qa..].csv（stream_b<B> 那列）
  - 目標：--ifprune_csv 有該 budget 的 ifprune_b<B> 就用同一個 session 實測值，否則用
          REF_TTFT（表上的數字，線性內插）
  - peak memory 不受 S 影響（且都低於 IF-Prune），所以不列入條件，只印出來確認

用法：
    python calibrate_qattn_tiles.py --outbase out_calib_InternVL3_5-8B_infovqa_s0n600 \
        --budgets 1024,1536,2048,2560,3072,4096,5120
    python calibrate_qattn_tiles.py ... --margin 0.03 --tol 0.3 --dry_run
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re

# 規則 A 的 TTFT 參考（秒，InfoVQA，bench_metrics 量法）；沒有同一個 session 的實測（--ifprune_csv）時才用，
# 不在表上的 budget 線性內插 / 外插。
REF_TTFT = {1024: 1.02, 2048: 1.18, 3072: 1.32, 4096: 1.45, 5120: 1.57}


def ref_ttft(budget: int) -> float:
    xs = sorted(REF_TTFT)
    i = 0 if budget <= xs[0] else (len(xs) - 2 if budget >= xs[-1] else max(j for j in range(len(xs) - 1) if xs[j] <= budget))
    x0, x1 = xs[i], xs[i + 1]
    return REF_TTFT[x0] + (REF_TTFT[x1] - REF_TTFT[x0]) * (budget - x0) / (x1 - x0)


def update_config(table: dict, path: str = "svma_config.py"):
    """把 svma_config.py 裡 QATTN_TILES_BY_BUDGET = { ... } 這一塊換成新的表（其他內容不動）。"""
    src = open(path, encoding="utf-8").read()
    body = "".join(f"    {b}: {s},\n" for b, s in sorted(table.items()))
    new, n = re.subn(r"QATTN_TILES_BY_BUDGET = \{\n.*?\n\}", "QATTN_TILES_BY_BUDGET = {\n" + body + "}",
                     src, count=1, flags=re.S)
    if n != 1:
        raise SystemExit(f"[update_config] {path} 裡找不到 QATTN_TILES_BY_BUDGET = {{ ... }}")
    open(path, "w", encoding="utf-8").write(new)
    print(f"[update_config] {path} 的 QATTN_TILES_BY_BUDGET 已更新為 {dict(sorted(table.items()))}")


def bench(path: str, method: str):
    if not os.path.exists(path):
        return None
    for r in csv.DictReader(open(path, encoding="utf-8")):
        if r.get("method") == method and r.get("ttft_s_mean"):
            return float(r["ttft_s_mean"]), float(r.get("peak_alloc_gb_mean") or "nan")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outbase", required=True, help="校準集 json 的前綴（<outbase>_b<B>.json）")
    ap.add_argument("--budgets", default="1024,1536,2048,2560,3072,4096,5120")
    ap.add_argument("--metrics_prefix", default="metrics_v2_InternVL3_5-8B_infovqa")
    ap.add_argument("--qa", default="20ctx")
    ap.add_argument("--ifprune_csv", default=None, help="同一個 session 量的 IF-Prune（bench_metrics --run_ifprune）")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="TTFT 至少要比 IF-Prune 快多少秒（吸收不同 session 的量測誤差，約 ±0.03s）")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="ANLS 跟最佳差在 tol 以內時選比較小的 S（0 = 純規則 A）")
    ap.add_argument("--out", default="svma_calibration.json")
    ap.add_argument("--dry_run", action="store_true", help="只印表、不寫校準檔")
    ap.add_argument("--update_config", action="store_true",
                    help="同時把選定的 S 寫進 svma_config.py 的 QATTN_TILES_BY_BUDGET（auto 的預設表）")
    args = ap.parse_args()

    from evaluate_anls import calculate_single_anls, paired_bootstrap_delta

    table, details = {}, {}
    for B in [int(x) for x in args.budgets.split(",") if x]:
        path = f"{args.outbase}_b{B}.json"
        if not os.path.exists(path):
            print(f"\n== b{B}: 沒有 {path}，略過")
            continue
        d = json.load(open(path, encoding="utf-8"))
        meta = d.get("meta", [])
        rows = {}
        for tag, preds in d.get("candidates", {}).items():
            m = re.fullmatch(rf"budget={B}_svma_S(\d+)(?:_qa{re.escape(args.qa)})?", tag)
            if not m or len(preds) != len(meta) or not meta:
                continue
            S = int(m.group(1))
            sc = [calculate_single_anls(p, x.get("answers") or [x.get("answer", "")]) for p, x in zip(preds, meta)]
            mname = f"S{S}" + (f"_qa{args.qa}" if S > 0 else "")
            rows[S] = {"scores": sc, "anls": 100 * sum(sc) / len(sc),
                       "bench": bench(f"{args.metrics_prefix}_{mname}.csv", f"stream_b{B}")}
        if not rows:
            print(f"\n== b{B}: {path} 裡沒有完整的 S 結果，略過")
            continue

        measured = bench(args.ifprune_csv, f"ifprune_b{B}") if args.ifprune_csv else None
        target = measured[0] if measured else ref_ttft(B)
        src = "實測" if measured else "表上 / 內插"
        limit = target - args.margin
        feasible = [S for S, r in rows.items() if r["bench"] and r["bench"][0] <= limit]
        if feasible:
            best = max(rows[S]["anls"] for S in feasible)
            chosen = min(S for S in feasible if rows[S]["anls"] >= best - args.tol)
            note = ""
        else:
            chosen = 0 if 0 in rows else min(rows)
            note = "（沒有任何 S 夠快，退回最快的）"

        print(f"\n== b{B}（n={len(meta)}）  IF-Prune TTFT 目標 {target:.3f}s（{src}），上限 {limit:.3f}s")
        print(f"  {'S':>3s} {'ANLS':>6s} {'vs S=0 (95% CI)':>24s} {'TTFT':>7s} {'peak GB':>7s}  可行")
        base = rows.get(0)
        for S in sorted(rows):
            r = rows[S]
            delta = "—"
            if base and S != 0:
                x = paired_bootstrap_delta(r["scores"], base["scores"])
                delta = f"{100*x[0]:+.2f} [{100*x[1]:+.2f},{100*x[2]:+.2f}]{'*' if x[1] > 0 or x[2] < 0 else ''}"
            t = r["bench"]
            ok = "✓" if S in feasible else ("—" if not t else "✗")
            mark = "  ← 選這個" if S == chosen else ""
            print(f"  {S:3d} {r['anls']:6.2f} {delta:>24s} {(f'{t[0]:.3f}' if t else '—'):>7s} "
                  f"{(f'{t[1]:.2f}' if t else '—'):>7s}  {ok}{mark}")
        if any(r["bench"] is None for r in rows.values()):
            print("  （有些 S 沒有 TTFT 量測，不能列入可行：先跑 bench_metrics）")
        print(f"  → b{B}: S = {chosen}{note}")
        table[B] = chosen
        details[B] = {"target_ttft": target, "target_source": src, "limit": limit, "n": len(meta),
                      "candidates": {S: {"anls": rows[S]["anls"], "ttft": rows[S]["bench"] and rows[S]["bench"][0]}
                                     for S in sorted(rows)}}

    print(f"\n選定：{ {b: s for b, s in sorted(table.items())} }")
    if args.dry_run or not table:
        return
    json.dump({"rule": "A: TTFT <= IF-Prune - margin 的 S 中 ANLS 最高（差距在 tol 內取小 S）",
               "margin": args.margin, "tol": args.tol, "outbase": args.outbase,
               "table": {str(b): s for b, s in sorted(table.items())},
               "details": {str(b): v for b, v in sorted(details.items())}},
              open(args.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[saved] {args.out}")
    if args.update_config:
        update_config(table)


if __name__ == "__main__":
    main()
