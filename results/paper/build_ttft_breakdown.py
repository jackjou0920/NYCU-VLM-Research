"""TTFT 分段（docs/06 §9 #6）→ paper/tables/tab_ttft.tex（補充材料）。
資料全部來自既有的效率量測（bench_metrics.py 的 t_* 欄位 = foveatile_internvl.py 的 stats["ttft_breakdown"]），不另外跑 GPU。
各段加總 = 該題 TTFT 扣掉約 2 ms 的 CPU 時間（組 prompt 等），所以這裡用平均（加總才會等於平均 TTFT）；正文其他地方的 TTFT 是中位數。
  (a) 8B，五個主表資料集各 50 張的平均，B ∈ {1024,…,3072}（檔案同 build_paper_results.eff_file）
  (b) HR-Bench 8K，B=1536，tile 上限 7…201（results/scaling；FoveaTile 在每個 baseline 的縮放檔裡都量了一次，同一個 tile 上限合併平均）
用法（repo 根目錄）：python results/paper/build_ttft_breakdown.py"""
import csv, glob, os, sys, statistics as st
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.getcwd())
import build_paper_results as P

MAIN5 = ["docvqa", "infovqa", "mme_rw_lite", "hrbench_4k", "vstar"]
BUD = (1024, 1536, 2048, 2560, 3072)
MAXNUM = (7, 25, 97, 145, 201)          # 13 / 49 的實際 tile 數與 7 / 25 相同（HR-8K 的長寬比），不重複列
STAGES = [("text_prefill", r"Prefill of $\mathbf{x}_b$"),
          ("cover_vit", "Vision encoder, coverage tiles"),
          ("prior", "Prior pass"),
          ("detail_vit", "Vision encoder, detail tiles"),
          ("qattn", r"Boundary scoring ($q_t$)"),
          ("kv_commit", "Prefill of selected tiles"),
          ("first_token", r"Prefill of $\mathbf{x}_a$ + first token")]
SELECT = ("prior", "qattn")             # 只為了「選哪些 tile」而多花的時間


def rows_of(files, key):
    return [r for f in files for r in csv.DictReader(open(f)) if r["method"] == key]


def pooled(rows, col):
    """每個檔的平均依張數加權合併。"""
    n = [int(r["n_images"]) for r in rows]
    return sum(float(r[col]) * k for r, k in zip(rows, n)) / sum(n)


def col_budget(b):
    per_ds = []
    for ds in MAIN5:
        rs = rows_of([P.eff_file(ds, b, "8B")], f"foveatile_b{b}")
        assert len(rs) == 1, (ds, b)
        per_ds.append({k: float(rs[0][f"t_{k}_mean"]) for k, _ in STAGES} | {"ttft": float(rs[0]["ttft_s_mean"])})
    unc = st.mean(float(rows_of([P.eff_file(ds, 1024, "8B")], "uncompressed")[0]["ttft_s_mean"]) for ds in MAIN5)
    out = {k: st.mean(d[k] for d in per_ds) for k in per_ds[0]}
    return out | {"unc": unc, "tiles": None}


def col_tiles(n):
    fs = glob.glob(f"results/scaling/metrics_n{n}_*.csv")
    rs = rows_of(fs, "foveatile_b1536")
    assert rs, n
    out = {k: pooled(rs, f"t_{k}_mean") for k, _ in STAGES} | {"ttft": pooled(rs, "ttft_s_mean")}
    unc = rows_of([f"results/scaling/metrics_n{n}_uncompressed.csv"], "uncompressed")[0]
    return out | {"unc": float(unc["ttft_s_mean"]), "tiles": round(float(rs[0]["raw_tiles_mean"]))}


A = [col_budget(b) for b in BUD]
B = [col_tiles(n) for n in MAXNUM]
for c in A + B:                          # 各段加總與平均 TTFT 只差 CPU 時間
    assert 0 <= c["ttft"] - sum(c[k] for k, _ in STAGES) < 0.01, c

ms = lambda x: f"{1000 * x:.0f}"
L = ["% 由 results/paper/build_ttft_breakdown.py 產生，不要手改",
     r"\begin{table*}[t]", r"  \centering", r"  \small", r"  \setlength{\tabcolsep}{4.5pt}",
     r"  \begin{tabular}{l" + "c" * len(A) + "c" + "c" * len(B) + "}", r"    \toprule",
     r"    & \multicolumn{" + str(len(A)) + r"}{c}{(a) Budget $B$ (five benchmarks)}" +
     r" & & \multicolumn{" + str(len(B)) + r"}{c}{(b) Input tiles (HR-8K)} \\",
     r"    \cmidrule(lr){2-" + str(1 + len(A)) + r"}\cmidrule(lr){" + str(3 + len(A)) + "-" + str(2 + len(A) + len(B)) + "}",
     "    Stage (ms) & " + " & ".join(str(b) for b in BUD) + " & & " + " & ".join(str(c["tiles"]) for c in B) + r" \\",
     r"    \midrule"]
for k, name in STAGES:
    L.append(f"    {name} & " + " & ".join(ms(c[k]) for c in A) + " & & " + " & ".join(ms(c[k]) for c in B) + r" \\")
L.append(r"    \midrule")
L.append(r"    \rowcolor{oursbg}" + "\n    TTFT, \\method & " + " & ".join(ms(c["ttft"]) for c in A) + " & & "
         + " & ".join(ms(c["ttft"]) for c in B) + r" \\")
L.append(r"    \quad of which selection (prior + $q_t$) & "
         + " & ".join(f"{100 * sum(c[k] for k in SELECT) / c['ttft']:.0f}\\%" for c in A) + " & & "
         + " & ".join(f"{100 * sum(c[k] for k in SELECT) / c['ttft']:.0f}\\%" for c in B) + r" \\")
L.append("    TTFT, uncompressed & \\multicolumn{" + str(len(A)) + "}{c}{" + ms(A[0]["unc"]) + " (independent of $B$)} & & "
         + " & ".join(ms(c["unc"]) for c in B) + r" \\")
L += [r"    \bottomrule", r"  \end{tabular}",
      r"  \caption{\textbf{TTFT breakdown} of \method on InternVL3.5-8B (mean in ms; the stages add up to the mean TTFT"
      r" up to about 2\,ms of CPU time). (a) Mean over DocVQA, InfoVQA, MME-RealWorld, HR-Bench 4K, and V$^*$ Bench,"
      r" 50 images each. (b) HR-Bench 8K with the tile cap raised from 7 to 201, at $B=1536$; columns give the"
      r" mean number of input tiles including the thumbnail.}",
      r"  \label{tab:ttft}", r"\end{table*}"]
os.makedirs("paper/tables", exist_ok=True)
open("paper/tables/tab_ttft.tex", "w").write("\n".join(L) + "\n")
print("\n".join(L))
