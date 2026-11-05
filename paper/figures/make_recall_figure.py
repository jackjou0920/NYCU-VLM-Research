"""觀察章節 Experiment 2 的圖：V* Bench、FoveaTile v1.0、b1536 的 Recall@K。
目標 = 每個物件 >= 50% bbox 面積被覆蓋（與 tab_analysis 相同）。
資料：results/diagnostics/vstar_selection/b1536_observation.json（experiments/diagnostics/vstar_observation_run.py 產生，
凍結 v1.0 後的事後分析；換題 = 同類別位移 37 題的問題）。
輸出：paper/figures/fig_recall.{pdf,png}、paper/tables/tab_recall.tex、paper/tables/tab_window.tex、
      results/diagnostics/vstar_selection/recall_curves_b1536.csv
用法（repo 根目錄）：python paper/figures/make_recall_figure.py"""
import csv, itertools, json, os, random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
DIAG = os.path.join(HERE, "..", "..", "results", "diagnostics", "vstar_selection")
ROOT = os.path.expanduser("~/datasets/vstar_bench")
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "dejavusans"})
qs = [json.loads(l) for l in open(f"{ROOT}/test_questions.jsonl")]
R = json.load(open(f"{DIAG}/b1536_observation.json"))
G = []
for q, r in zip(qs, R):
    p = f"{ROOT}/{q['image']}"
    G.append((*Image.open(p).size, r["detail_grid"][0], r["detail_grid"][1],
              json.load(open(p.rsplit(".", 1)[0] + ".json"))["bbox"]))

def hit(g, T):
    W, H, c, rw, bb = g
    tw, th = W / c, H / rw
    for x, y, w, h in bb:
        a = sum(max(0, min(x + w, (t % c + 1) * tw) - max(x, t % c * tw)) *
                max(0, min(y + h, (t // c + 1) * th) - max(y, t // c * th)) for t in T)
        if a / max(1, w * h) < .5:
            return False
    return True

order = lambda v: sorted(range(len(v)), key=lambda i: -v[i])
need = lambda g, o: next(i + 1 for i in range(len(o)) if hit(g, o[:i + 1]))

def refined(r):
    sc = {int(k): v for k, v in r["candidate_scores"].items()}
    d, w = set(r["direct_kept"]), set(r["window"])
    return ([t for t in r["prior_ranked"] if t in d] + sorted(w, key=lambda t: (-sc[t], t)) +
            [t for t in r["prior_ranked"] if t not in d and t not in w])

def uniform(g):
    c, rw = g[2], g[3]
    pts = [((t % c + .5) / c, (t // c + .5) / rw) for t in range(c * rw)]
    o = [min(range(len(pts)), key=lambda t: (pts[t][0] - .5) ** 2 + (pts[t][1] - .5) ** 2)]
    while len(o) < len(pts):
        o.append(max((t for t in range(len(pts)) if t not in o),
                     key=lambda t: min((pts[t][0] - pts[u][0]) ** 2 + (pts[t][1] - pts[u][1]) ** 2 for u in o)))
    return o

Ks = list(range(1, 13))
N = {"prior": [need(g, r["prior_ranked"]) for g, r in zip(G, R)],
     "refined": [need(g, refined(r)) for g, r in zip(G, R)],
     "swapped": [need(g, order(r["swapped"])) for g, r in zip(G, R)],
     "uniform": [need(g, uniform(g)) for g in G]}
curves = {k: [100 * float(np.mean(np.array(v) <= K)) for K in Ks] for k, v in N.items()}
rng = random.Random(0)
curves["random"] = [100 * float(np.mean([np.mean([hit(g, rng.sample(range(g[2] * g[3]), K)) for _ in range(200)]) for g in G]))
                    for K in Ks]
with open(f"{DIAG}/recall_curves_b1536.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["K"] + list(curves))
    for i, K in enumerate(Ks):
        w.writerow([K] + [round(curves[c][i], 1) for c in curves])

S = [("refined", "After boundary scoring", "#2a78d6", "o", "-", 1.5),
     ("prior", "Coarse prior", "#eb6834", "s", "-", 1.1),
     ("swapped", "Prior, swapped question", "#1baf7a", "^", "--", 1.0),
     ("random", "Random", "#eda100", "D", ":", 1.0),
     ("uniform", "Spatially uniform", "#e87ba4", "v", ":", 1.0)]
fig, ax = plt.subplots(figsize=(3.3, 2.05))
for key, lab, col, mk, ls, lw in S:
    ax.plot(Ks[:9], curves[key][:9], color=col, marker=mk, ms=2.8, lw=lw, ls=ls, label=lab,
            zorder=4 if key == "refined" else 3, clip_on=False)
ax.axvline(4, color="#bbbbbb", lw=0.6, zorder=1)
ax.annotate("detail slots ($k\\approx4$)", (4, 3), xytext=(3, 0), textcoords="offset points", fontsize=5.4, color="#555555")
ax.set(xlim=(1, 9), ylim=(0, 100), xticks=Ks[:9], xlabel="Top-$K$ detail tiles",
       ylabel="Recall@$K$ (%)")
ax.grid(color="#eeeeee", lw=0.5)
ax.set_axisbelow(True)
ax.legend(frameon=False, fontsize=5.6, loc="lower center", ncol=2, handlelength=2.4, columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))
fig.savefig(f"{HERE}/fig_recall.pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(f"{HERE}/fig_recall.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
print(open(f"{DIAG}/recall_curves_b1536.csv").read())


# ───────────── 表：各 budget 的 Recall 與窗口去向（b1024 / b2048 / b3072 來自 3a 診斷，b1536 同上）─────────────
TAB = os.path.join(HERE, "..", "tables")

def load(b):
    if b == 1536:
        return [dict(r, committed_tile_indices=r["committed"]) for r in R]
    return json.load(open(f"{DIAG}/b{b}.json"))

T1, T2 = [], []
for b, K in ((1024, 2), (1536, 4), (2048, 4), (3072, 6)):
    D = load(b)
    assert all(d["detail_grid"] == list(g[2:4]) for d, g in zip(D, G))
    pr = np.mean([need(g, d["prior_ranked"]) <= K for g, d in zip(G, D)]) * 100
    rf = np.mean([need(g, refined(d)) <= K for g, d in zip(G, D)]) * 100
    T1.append((b, K, curves["random"][K - 1], curves["uniform"][K - 1], pr, rf))
    ph = rec = lost = final = entered = 0; elig = []; rnd = []
    for g, d in zip(G, D):
        sel, k = d["committed_tile_indices"], len(d["committed_tile_indices"])
        dk = d["direct_kept"]; win = d["window"]; a = k - len(dk)
        p, s = hit(g, d["prior_ranked"][:k]), hit(g, sel)
        ph += p; final += s; lost += p and not s
        if hit(g, dk + win) and not hit(g, dk):
            entered += 1
            if not p:
                rec += s; elig.append(s)
                # 隨機從窗口選 a 塊的命中機率（窮舉，精確值）
                rnd.append(np.mean([hit(g, dk + list(c)) for c in itertools.combinations(win, a)]))
    assert ph + rec - lost == final, b
    T2.append((b, len(D[0]["window"]), len(D[0]["committed_tile_indices"]) - len(D[0]["direct_kept"]), ph, rec, lost, final, 100 * final / len(D),
               100 * rec / len(elig), 100 * float(np.mean(rnd)), len(elig), entered))

L = ["\\begin{table}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{lc cc cc}", "    \\toprule",
     "    $B$ & $K$ & Random & Uniform & Coarse prior & After scoring \\\\", "    \\midrule"]
for b, K, rn, un, pr, rf in T1:
    L.append(f"    {b} & {K} & {rn:.1f} & {un:.1f} & {pr:.1f} & \\textbf{{{rf:.1f}}} \\\\")
L += ["    \\bottomrule", "  \\end{tabular}",
      "  \\caption{\\textbf{Recall@$K$ (\\%) on V$^*$~Bench} at about the number of detail tiles each budget keeps, with $K$ fixed so that all columns are comparable. Random and uniform selection do not depend on the budget.}",
      "  \\label{tab:recall}", "\\end{table}"]
open(f"{TAB}/tab_recall.tex", "w").write("\n".join(L) + "\n")

L = ["\\begin{table}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{2.5pt}", "  \\begin{tabular}{lc ccc cc}", "    \\toprule",
     "    $B$ & Window & Prior & Recov. & Lost & Final & Rate (rand.) \\\\", "    \\midrule"]
for b, S_, a, ph, rec, lost, fin, fp, rr, rd, el, en in T2:
    L.append(f"    {b} & {a} of {S_} & {ph} & +{rec} & {f'$-${lost}' if lost else 0} & \\textbf{{{fp:.1f}\\%}} & {rr:.1f}\\% ({rd:.1f}\\%) \\\\")
L += ["    \\bottomrule", "  \\end{tabular}",
      "  \\caption{\\textbf{From the prior to the final selection} on the 191 questions of V$^*$~Bench. Prior: questions on which the prior alone covers the target with the tiles of the budget. Recov.: the target enters the window, the prior alone would miss it, and boundary scoring keeps it. Lost: the reverse. Final = (prior + recovered $-$ lost) / 191. Rate: the recovery rate, computed over the questions whose target enters the window and that the prior alone would miss (" + ", ".join(str(t[10]) for t in T2) + " questions); in parentheses, the expected rate when the window tiles are kept at random.}",
      "  \\label{tab:window}", "\\end{table}"]
open(f"{TAB}/tab_window.tex", "w").write("\n".join(L) + "\n")
print("T1", [tuple(round(x, 1) for x in t) for t in T1]); print("T2", [tuple(round(float(x), 1) for x in t) for t in T2])
print("ratio", [round(t[8] / t[9], 2) for t in T2])

# 配對 bootstrap：prior vs 換題 @4；重排 vs prior @4（實際名額）
rs = np.random.default_rng(0)
def ci(a, c):
    a, c = np.array(a, float), np.array(c, float); n = len(a)
    d = [(a[i] - c[i]).mean() for i in (rs.integers(0, n, n) for _ in range(10000))]
    return f"{(a-c).mean()*100:+.1f} [{np.percentile(d,2.5)*100:+.1f}, {np.percentile(d,97.5)*100:+.1f}]"
print("prior-swapped@4", ci(np.array(N["prior"]) <= 4, np.array(N["swapped"]) <= 4))
print("prior-random@4 mean diff", round(curves["prior"][3] - curves["random"][3], 1))


# ───────────── Supplementary：完整 Recall 表，以及文字用的統計 ─────────────
Kf = [1, 2, 3, 4, 5, 6, 8, 12]; L = []; pos = {}; cols = []
for b in (1024, 1536, 2048, 3072):
    D = load(b)
    p_, r_ = np.array([need(g, d["prior_ranked"]) for g, d in zip(G, D)]), np.array([need(g, refined(d)) for g, d in zip(G, D)])
    cols.append([(100 * (p_ <= K).mean(), 100 * (r_ <= K).mean()) for K in Kf])
    pos[b] = 100 * np.mean([((d["prior_ranked"][0] // d["detail_grid"][0]) + .5) / d["detail_grid"][1] < .25 for d in D])
L = ["\\begin{table*}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{r cc cc cc cc}", "    \\toprule",
     "    & \\multicolumn{2}{c}{$B=1024$} & \\multicolumn{2}{c}{$B=1536$} & \\multicolumn{2}{c}{$B=2048$} & \\multicolumn{2}{c}{$B=3072$} \\\\",
     "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}\\cmidrule(lr){8-9}",
     "    $K$ & Prior & After scoring & Prior & After scoring & Prior & After scoring & Prior & After scoring \\\\", "    \\midrule"]
for i, K in enumerate(Kf):
    L.append(f"    {K} & " + " & ".join(f"{c[i][0]:.1f} & {c[i][1]:.1f}" for c in cols) + " \\\\")
L += ["    \\bottomrule", "  \\end{tabular}",
      "  \\caption{\\textbf{Recall@$K$ (\\%) of the coarse prior and after boundary scoring} on V$^*$~Bench, at every budget with saved selections.}",
      "  \\label{tab:recall_full}", "\\end{table*}"]
open(f"{TAB}/tab_recall_full.tex", "w").write("\n".join(L) + "\n")
print("top-quarter", {k: round(float(v), 1) for k, v in pos.items()})
print("swapped rank-1 same", round(100 * np.mean([order(r["swapped"])[0] == order(r["prior"])[0] for r in R]), 1))
