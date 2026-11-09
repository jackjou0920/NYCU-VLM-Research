"""docs/06 §9 #1：HiRED（results/runs_hired）vs FoveaTile 與現有 baseline（results/runs），同一 budget。
配對 bootstrap = summarize_results.paired_delta（2,000 次、seed 0、95%；HR-Bench 以 CircularEval 組為單位）。
用法（repo 根目錄）：python results/paper/analyze_hired.py [budget]"""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import build_paper_results as P
import summarize_results as R

B = int(sys.argv[1]) if len(sys.argv) > 1 else 1536
DS = [("docvqa", "DocVQA"), ("infovqa", "InfoVQA"), ("mme_rw_lite", "MME-RW"), ("hrbench_4k", "HR-4K"), ("vstar", "V*")]
OTHERS = ["fastv", "sparsevlm", "visionzip", "ifprune"]


def us(ds, m):
    return P.unit_scores(ds, m, 0 if m == "uncompressed" else B, root="results/runs_hired" if m == "hired" else "results/runs")


print(f"=== B={B} ===")
print(f"{'dataset':8s} {'uncomp':>7s} {'HiRED':>7s} {'Fovea':>7s} " + " ".join(f"{m[:8]:>8s}" for m in OTHERS) + "   Fovea-HiRED")
ret = {m: [] for m in ["hired", "foveatile"] + OTHERS}
for ds, name in DS:
    S = {m: us(ds, m) for m in ["uncompressed", "hired", "foveatile"] + OTHERS}
    sc = {m: None if v is None else 100 * st.mean(v.values()) for m, v in S.items()}
    f = lambda x: "    -  " if x is None else f"{x:7.1f}"
    if S["hired"] is not None and S["foveatile"] is not None:
        d, lo, hi, _ = R.paired_delta(S["foveatile"], S["hired"])
        cell = f"{d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]" + ("↑" if lo > 0 else "↓" if hi < 0 else "")
    else:
        cell = "pending"
    print(f"{name:8s} {f(sc['uncompressed'])} {f(sc['hired'])} {f(sc['foveatile'])} "
          + " ".join(f"{f(sc[m]):>8s}" for m in OTHERS) + f"   {cell}")
    for m in ret:
        if sc[m] is not None and sc["uncompressed"]:
            ret[m].append(100 * sc[m] / sc["uncompressed"])
print("avg retention:", {m: (round(st.mean(v), 1) if len(v) == len(DS) else f"{len(v)}/{len(DS)}") for m, v in ret.items()})
