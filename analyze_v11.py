"""
v1.1 候選的判斷（docs/04_calibration_protocol.md §7b 事先定的標準）：
  1. 每個 budget 的平均保留率（InfoVQA train calib + MME-RW calib，相對 uncompressed）不比 v1.0 低超過 0.5 個百分點
  2. MME-RW 局部細節題（Remote Sensing + Monitoring + Autonomous Driving）3 個 budget 合併，相對 v1.0 顯著進步
     （paired bootstrap 4,000 次、seed 0，CI 下界 > 0）
  只為排列不變性採用 A：stem / stemopt 搭 mean，通過 1，且局部細節合併 Δ 的 CI 下界 > -1.0
  （標準 3 的 TTFT 選定後另外量，這支只看 accuracy）

用法：python analyze_v11.py [--root results/calib]
"""
from __future__ import annotations

import os
import json
import random
import argparse
import statistics as st

from summarize_results import unit_scores, METRIC_OF

BUDGETS = {1024: "m2_S5", 1536: "m3_S4", 2560: "m5_S3"}
VARIANTS = [(q, a) for q in ("all", "stem", "stemopt") for a in ("mean", "top0.1", "top0.25")] + [("sortopt", "mean")]
# sortopt 在沒有選項的題目上跟 v1.0 完全相同（scoring_query 直接回傳原本的 text_after），InfoVQA 用 v1.0 的結果
SAME_AS_V10 = {("sortopt", "mean"): ("infovqa_train_calib",)}
LOCAL = ("Remote Sensing", "Monitoring", "Autonomous_Driving")
MTAG = "InternVL3_5-8B"


def suffix(q, a):
    if q == "raw":                       # --extra 給的任意 tag 後綴（例如 _pha1_pbnlratio）
        return a
    return ("" if q == "all" else f"_q{q}") + ("" if a == "mean" else f"_agg{a}")


def load_scores(root, ds, tag):
    path = os.path.join(root, ds, f"{tag}_{MTAG}.json")
    if not os.path.exists(path):
        return None, None
    data = json.load(open(path, encoding="utf-8"))
    key = list(data["candidates"])[0]
    return unit_scores(data, key, METRIC_OF(ds), None), data["meta"]


def boot(d, n_boot=4000, seed=0):
    rng = random.Random(seed)
    bs = sorted(sum(rng.choice(d) for _ in d) / len(d) for _ in range(n_boot))
    return 100 * st.mean(d), 100 * bs[int(0.025 * n_boot)], 100 * bs[int(0.975 * n_boot) - 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="results/calib")
    ap.add_argument("--extra", nargs="*", default=[], help="另外比較的 tag 後綴，例如 _pha1_pbnlratio")
    args = ap.parse_args()
    VARIANTS.extend(("raw", x) for x in args.extra)
    unc = {ds: st.mean(load_scores(args.root, ds, "uncompressed")[0].values())
           for ds in ("infovqa_train_calib", "mme_rw_calib")}
    print("uncompressed:", {k: round(100 * v, 2) for k, v in unc.items()})

    res = {}
    for q, a in VARIANTS:
        ret, local_d, ok_all = {}, [], True
        for b, ma in BUDGETS.items():
            tag = f"svma_b{b}_{ma}{suffix(q, a)}"
            rr = []
            for ds in ("infovqa_train_calib", "mme_rw_calib"):
                sc, meta = load_scores(args.root, ds, f"svma_b{b}_{ma}" if ds in SAME_AS_V10.get((q, a), ()) else tag)
                if sc is None or len(sc) < len(meta):
                    ok_all = False
                    break
                rr.append(st.mean(sc.values()) / unc[ds])
                if ds == "mme_rw_calib":
                    base, _ = load_scores(args.root, ds, f"svma_b{b}_{ma}")
                    local_d += [sc[i] - base[i] for i in sc if meta[i]["category"].split("/")[0] in LOCAL]
            if not ok_all:
                break
            ret[b] = 100 * st.mean(rr)
        if ok_all:
            res[(q, a)] = (ret, boot(local_d) if (q, a) != ("all", "mean") else (0.0, 0.0, 0.0))

    if ("all", "mean") not in res:
        raise SystemExit("缺 v1.0 對照")
    base_ret = res[("all", "mean")][0]
    print(f"\n{'variant':18s} " + " ".join(f"ret b{b:<5d}(Δ)  " for b in BUDGETS)
          + "  local Δ [95% CI]          crit1 crit2 | invariance-only")
    for (q, a), (ret, (d, lo, hi)) in res.items():
        dr = {b: ret[b] - base_ret[b] for b in BUDGETS}
        c1 = all(v >= -0.5 for v in dr.values())
        c2 = lo > 0
        inv = q in ("stem", "stemopt") and a == "mean" and c1 and lo > -1.0
        name = f"{q}/{a}"
        print(f"{name:18s} " + " ".join(f"{ret[b]:6.2f}({dr[b]:+5.2f})  " for b in BUDGETS)
              + f"  {d:+5.2f} [{lo:+5.2f}, {hi:+5.2f}]   {'✓' if c1 else '✗':5s} {'✓' if c2 else '✗':5s} | "
              + ("✓" if inv else "-"))
    missing = [f"{q}/{a}" for q, a in VARIANTS if (q, a) not in res]
    if missing:
        print("\n尚未完成：", ", ".join(missing))


if __name__ == "__main__":
    main()
