"""docs/06 §9 #2：同預算「直接降解析度」對照的結果彙整（不改任何論文表格）。
lowres = 官方動態切 tile 上限 K 塊；coverage = 覆蓋 tile 拿全部預算（m = K）。
FoveaTile / uncompressed 讀 results/runs，對照讀 results/runs_lowres。配對 bootstrap = summarize_results.paired_delta
（2,000 次、seed 0、95%；HR-Bench 以 CircularEval 組為單位）。
用法（repo 根目錄）：python results/paper/analyze_lowres.py"""
import os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import build_paper_results as P
import summarize_results as R

DS = [("docvqa", "DocVQA"), ("infovqa", "InfoVQA"), ("mme_rw_lite", "MME-RW"), ("hrbench_4k", "HR-4K"),
      ("vstar", "V*"), ("hrbench_8k", "HR-8K")]
MAIN5 = ["docvqa", "infovqa", "mme_rw_lite", "hrbench_4k", "vstar"]
LR = "results/runs_lowres"

def us(ds, m, b):
    return P.unit_scores(ds, m, b, root=LR if m in ("lowres", "coverage") else "results/runs")

for b in (1536, 2560, 3072):
    print(f"\n=== B={b} ===")
    print(f"{'dataset':8s} {'uncomp':>7s} {'lowres':>7s} {'coverage':>8s} {'FoveaTile':>9s}   Fovea-coverage            Fovea-lowres")
    ret = {m: [] for m in ("lowres", "coverage", "foveatile")}
    for ds, name in DS:
        U = us(ds, "uncompressed", b)
        u = None if U is None else 100 * st.mean(U.values())
        S = {m: us(ds, m, b) for m in ("lowres", "coverage", "foveatile")}
        sc = {m: (None if v is None else 100 * st.mean(v.values())) for m, v in S.items()}
        cells = []
        for base in ("coverage", "lowres"):
            if S["foveatile"] is None or S[base] is None:
                cells.append("pending"); continue
            d, lo, hi, n = R.paired_delta(S["foveatile"], S[base])
            sig = "↑" if lo > 0 else ("↓" if hi < 0 else " ")
            cells.append(f"{d:+5.1f} [{lo:+5.1f},{hi:+5.1f}]{sig}")
        f = lambda x: "   -  " if x is None else f"{x:7.1f}"
        print(f"{name:8s} {f(u)} {f(sc['lowres'])} {f(sc['coverage']):>8s} {f(sc['foveatile']):>9s}   {cells[0]:24s}  {cells[1]}")
        if ds in MAIN5 and u:
            for m in ret:
                if sc[m] is not None:
                    ret[m].append(100 * sc[m] / u)
    print("avg retention (5 main):", {m: (round(st.mean(v), 1) if len(v) == 5 else f"{len(v)}/5 done") for m, v in ret.items()})
