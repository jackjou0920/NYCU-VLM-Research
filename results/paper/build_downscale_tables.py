"""同預算降解析度對照（docs/06 §9 #2）→ 論文表格。
產生 paper/tables/tab_downscale.tex（主文）與 tab_downscale_full.tex（補充材料，分數 + 配對差 95% CI）。
Low-Res = 官方切 tile 上限 K 塊（--mode uncompressed --max_num K）；Low-Res-Fill = 覆蓋 tile 拿全部預算（m = K）。
實際 visual token 數由 KV 大小換算：每 token KV = 36 層 × 2 × 8 頭 × 128 維 × 2 bytes = 0.140625 MB，
與同一題的 Low-Res-Fill（剛好 B 個 visual token）相減即得。
用法（repo 根目錄）：python results/paper/build_downscale_tables.py"""
import os, sys, json, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import build_paper_results as P
import summarize_results as R

DS = [("docvqa", "DocVQA"), ("infovqa", "InfoVQA"), ("mme_rw_lite", "MME-RW"), ("hrbench_4k", "HR-4K"),
      ("vstar", "V$^*$"), ("hrbench_8k", "HR-8K")]
MAIN5 = ["docvqa", "infovqa", "mme_rw_lite", "hrbench_4k", "vstar"]
SMALL = ["hrbench_4k", "hrbench_8k", "vstar"]
BUD = (1536, 2560, 3072)
LR = "results/runs_lowres"
MB_PER_TOK = 36 * 2 * 8 * 128 * 2 / 2**20
MODEL = "InternVL3_5-8B"
OUT = "paper/tables"


def us(ds, m, b):
    return P.unit_scores(ds, m, b, root=LR if m in ("lowres", "coverage") else "results/runs")


def kv(ds, v, b):
    d = json.load(open(f"{LR}/{ds}/{v}_b{b}_{MODEL}.json"))
    return list(d["kv_cache_mb"].values())[0]


score, delta, tokens = {}, {}, {}
for b in BUD:
    for ds, _ in DS:
        S = {m: us(ds, m, b) for m in ("uncompressed", "lowres", "coverage", "foveatile")}
        assert all(v is not None for v in S.values()), (ds, b)
        for m, v in S.items():
            score[ds, m, b] = 100 * st.mean(v.values())
        for base in ("coverage", "lowres"):
            d, lo, hi, _ = R.paired_delta(S["foveatile"], S[base])
            delta[ds, base, b] = (d, lo, hi)
        a, c = kv(ds, "lowres", b), kv(ds, "coverage", b)
        assert len(a) == len(c)
        tokens[ds, b] = b + st.mean(x - y for x, y in zip(a, c)) / MB_PER_TOK


def ret(m, b, dss):
    return st.mean(100 * score[ds, m, b] / score[ds, "uncompressed", b] for ds in dss)


for b in BUD:
    print(b, "Low-Res tokens:", {ds: round(tokens[ds, b]) for ds, _ in DS},
          "mean(main5)=", round(st.mean(tokens[ds, b] for ds in MAIN5)))

ROWS = [("lowres", "Low-Res"), ("coverage", "Low-Res-Fill"), ("foveatile", "\\method (ours)")]
tok_main = {b: round(st.mean(tokens[ds, b] for ds in MAIN5)) for b in BUD}

# ── 主文小表 ───────────────────────────────────────────────────────────────
L = [r"\begin{table}[t]", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{3.4pt}",
     r"  \begin{tabular}{l cc cc cc}", r"    \toprule",
     "    & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$B={b}$}}" for b in BUD) + r" \\",
     r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
     r"    Method & Avg. & Small & Avg. & Small & Avg. & Small \\", r"    \midrule"]
for m, name in ROWS:
    cells = []
    for b in BUD:
        for dss in (MAIN5, SMALL):
            v = ret(m, b, dss)
            best = max(ret(x, b, dss) for x, _ in ROWS)
            cells.append(f"\\textbf{{{v:.1f}}}" if abs(v - best) < 1e-9 else f"{v:.1f}")
    if m == "foveatile":
        L.append(r"    \rowcolor{oursbg}")
    L.append(f"    {name} & " + " & ".join(cells) + r" \\")
L += [r"    \bottomrule", r"  \end{tabular}",
      r"  \caption{\textbf{Same-budget downscaling}, as a percentage of the uncompressed score. Avg.: the five "
      r"benchmarks of \cref{tab:main}; Small: HR-Bench 4K, HR-Bench 8K, and V$^*$. Low-Res is the official tiling "
      r"capped at $K=B/256$ tiles, which cannot always fill the budget (on average "
      + "/".join(f"{tok_main[b]:,}" for b in BUD).replace(",", "{,}") +
      r" visual tokens); Low-Res-Fill gives the whole budget to coverage tiles ($m{=}K$). Per-benchmark scores and paired "
      r"confidence intervals are in the supplementary material.}",
      r"  \label{tab:downscale}", r"\end{table}"]
open(f"{OUT}/tab_downscale.tex", "w").write("% 由 results/paper/build_downscale_tables.py 產生，勿手改。\n" + "\n".join(L) + "\n")

# ── 補充材料：分數 ─────────────────────────────────────────────────────────
F = [r"\begin{table*}[t]", r"  \centering", r"  \footnotesize", r"  \setlength{\tabcolsep}{4pt}",
     r"  \begin{tabular}{l c ccc ccc ccc}", r"    \toprule",
     "    & & " + " & ".join(f"\\multicolumn{{3}}{{c}}{{$B={b}$}}" for b in BUD) + r" \\",
     r"    \cmidrule(lr){3-5}\cmidrule(lr){6-8}\cmidrule(lr){9-11}",
     r"    Benchmark & Unc. & " + " & ".join(["Low-Res & Fill & Ours"] * 3) + r" \\", r"    \midrule"]
for ds, name in DS:
    cells = [f"{score[ds, 'uncompressed', BUD[0]]:.1f}"]
    for b in BUD:
        vals = [score[ds, m, b] for m, _ in ROWS]
        cells += [f"\\textbf{{{v:.1f}}}" if v == max(vals) else f"{v:.1f}" for v in vals]
    F.append(f"    {name} & " + " & ".join(cells) + r" \\")
F.append(r"    \midrule")
F.append(r"    Low-Res tokens & -- & " + " & ".join(
    f"\\multicolumn{{3}}{{c}}{{{tok_main[b]:,} / {b:,}}}".replace(",", "{,}") for b in BUD) + r" \\")
F += [r"    \bottomrule", r"  \end{tabular}",
      r"  \caption{\textbf{Same-budget downscaling: scores.} Fill: Low-Res-Fill ($m{=}K$). Low-Res tokens: average "
      r"number of visual tokens used by Low-Res over the five benchmarks of \cref{tab:main}, against the budget; "
      r"Low-Res-Fill and \method use exactly $B$.}",
      r"  \label{tab:downscale_full}", r"\end{table*}", ""]

# ── 補充材料：配對差 ───────────────────────────────────────────────────────
def cell(t):
    d, lo, hi = t
    s = f"{d:+.1f}\\,[{lo:+.1f},\\,{hi:+.1f}]".replace("-", "$-$")
    return f"\\textbf{{{s}}}" if lo > 0 else s

F += [r"\begin{table*}[t]", r"  \centering", r"  \scriptsize", r"  \setlength{\tabcolsep}{2.4pt}",
      r"  \begin{tabular}{l cc cc cc}", r"    \toprule",
      "    & " + " & ".join(f"\\multicolumn{{2}}{{c}}{{$B={b}$}}" for b in BUD) + r" \\",
      r"    \cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
      r"    Benchmark & " + " & ".join(["vs.\\ Fill & vs.\\ Low-Res"] * 3) + r" \\", r"    \midrule"]
for ds, name in DS:
    F.append(f"    {name} & " + " & ".join(cell(delta[ds, base, b]) for b in BUD for base in ("coverage", "lowres")) + r" \\")
F += [r"    \bottomrule", r"  \end{tabular}",
      r"  \caption{\textbf{Same-budget downscaling: paired differences} (\method minus baseline, points) with "
      r"95\% paired bootstrap intervals (2{,}000 resamples; HR-Bench resamples CircularEval groups). Bold: the "
      r"interval excludes zero in favor of \method.}",
      r"  \label{tab:downscale_delta}", r"\end{table*}"]
open(f"{OUT}/tab_downscale_full.tex", "w").write("% 由 results/paper/build_downscale_tables.py 產生，勿手改。\n" + "\n".join(F) + "\n")

for b in BUD:
    print(b, {m: (round(ret(m, b, MAIN5), 1), round(ret(m, b, SMALL), 1)) for m, _ in ROWS})
