"""第 4 章的所有表格（LaTeX）與圖，全部直接從結果檔產生；還沒跑完的格子輸出 \\pend（紅色問號）。
實驗補完後重跑一次即可：python results/paper/build_paper_results.py

輸出：
  paper/tables/tab_main.tex      Table：8B 主表（budget 區塊 × 方法 × 6 個測試集 + 平均保留率 + TTFT / 記憶體 / FLOPs）
  paper/tables/tab_scale.tex     Table：4B / 8B / 14B @ b1536
  paper/tables/tab_hr8k.tex      Table：HR-Bench 8K，tile 上限 47（約 25 塊）vs 上限 200（188 塊）@ b1536
  paper/tables/tab_analysis.tex  Table：題型拆分 + V* Bench 命中 / 沒命中 @ b1536
  paper/tables/tab_ablation.tex  Table：held-out 校準集上的 ablation @ b1536
  paper/figures/fig_pareto.pdf   Fig：平均保留率 vs TTFT / 記憶體（8B，b1024~b3072）
  paper/figures/fig_scaling.pdf  Fig：TTFT / 記憶體 vs tile 數（HR-Bench 8K，b1536）
  results/paper/main_6ds.csv     所有數字（含 4B / 14B）
"""
import csv, json, os, statistics as st, sys
from collections import defaultdict

sys.path.insert(0, ".")
import summarize_results as R

OUT_TAB, OUT_FIG = "paper/tables", "paper/figures"
os.makedirs(OUT_TAB, exist_ok=True)

MODELS = {"4B": "InternVL3_5-4B", "8B": "InternVL3_5-8B", "14B": "InternVL3_5-14B"}
DS = [("docvqa", "DocVQA"), ("infovqa", "InfoVQA"), ("mme_rw_lite", "MME-RW"),
      ("hrbench_4k", "HR-4K"), ("hrbench_8k", "HR-8K"), ("vstar", "V$^*$ Bench")]
METHODS = [("fastv", "FastV~\\cite{fastv}"), ("sparsevlm", "SparseVLM~\\cite{sparsevlm}"),
           ("visionzip", "VisionZip~\\cite{visionzip}"), ("ifprune", "IF-Prune~\\cite{ifprune}"),
           ("foveatile", "\\method (ours)")]
SHORT = {"fastv": "FastV", "sparsevlm": "SparseVLM", "visionzip": "VisionZip", "ifprune": "IF-Prune",
         "foveatile": "\\method (ours)", "uncompressed": "Uncompressed"}
BUDGETS_ALL = [1024, 1536, 2048, 2560, 3072]
MAIN_BLOCKS = [1536, 2560, 3072]
DEFAULT_B = 1536
PEND = "\\pend"
PREAMBLE = "\\providecommand{\\pend}{\\textcolor{red}{?}}\n"


# ─────────────────────────────── 資料 ───────────────────────────────
def load_run(ds, method, budget, model="8B", root="results/runs"):
    tag = "uncompressed" if method == "uncompressed" else f"{method}_b{budget}"
    f = f"{root}/{ds}/{tag}_{MODELS[model]}.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    t = list(d["candidates"])[0]
    if len(d["candidates"][t]) < len(d["meta"]):          # 還在跑
        return None
    return d, t


def unit_scores(ds, method, budget, model="8B", root="results/runs"):
    r = load_run(ds, method, budget, model, root)
    return None if r is None else R.unit_scores(r[0], r[1], R.METRIC_OF(ds), None)


def score(ds, method, budget, model="8B", root="results/runs"):
    u = unit_scores(ds, method, budget, model, root)
    return None if u is None else 100 * st.mean(u.values())


def eff_file(ds, budget, model):
    if model != "8B":
        return f"results/efficiency/{MODELS[model]}/metrics_{ds}.csv"
    if ds == "hrbench_8k":
        return "results/efficiency/hrbench_8k/metrics_hrbench_8k.csv"
    return {1536: f"results/efficiency/b1536/metrics_{ds}.csv",
            2560: f"results/efficiency/b2560/metrics_{ds}.csv"}.get(budget, f"results/efficiency/metrics_{ds}.csv")


def eff(ds, method, budget, model="8B"):
    """(TTFT 中位數 s, 記憶體增量 GB, TFLOPs, 原始 tile 數)；沒有量測時 None。"""
    f = eff_file(ds, budget, model)
    if not os.path.exists(f):
        return None
    key = "uncompressed" if method == "uncompressed" else f"{method}_b{budget}"
    for r in csv.DictReader(open(f)):
        if r["method"] == key:
            return tuple(float(r[c]) for c in ("ttft_s_median", "mem_delta_gb_mean", "total_tflops_mean", "raw_tiles_mean"))
    return None


def mean_or_none(xs):
    return None if any(x is None for x in xs) else st.mean(xs)


def summary(method, budget, model="8B"):
    """{'raw': [6], 'ret': [6], 'avg': ret 平均, 'ttft', 'mem', 'flops'}（缺的是 None）"""
    unc = [score(d, "uncompressed", 0, model) for d, _ in DS]
    raw = [score(d, method, budget, model) for d, _ in DS]
    ret = [None if (r is None or u is None) else 100 * r / u for r, u in zip(raw, unc)]
    effs = [eff(d, method, budget, model) for d, _ in DS]
    cost = [mean_or_none([e[i] if e else None for e in effs]) for i in range(3)]
    return {"raw": raw, "ret": ret, "avg": mean_or_none(ret), "ttft": cost[0], "mem": cost[1], "flops": cost[2]}


def keep_ratio(budget, model="8B"):
    """budget 佔未壓縮視覺 token 的平均比例（各測試集的 B / (256 · 原始 tile 數) 再平均）。"""
    xs = [eff(d, "uncompressed", 0, model) for d, _ in DS]
    return None if any(x is None for x in xs) else 100 * st.mean(min(1.0, budget / (256 * x[3])) for x in xs)


def fmt(x, nd=1):
    return PEND if x is None else f"{x:.{nd}f}"


def bold_if(x, best, nd=1, lower=False):
    if x is None:
        return PEND
    s = f"{x:.{nd}f}"
    return f"\\textbf{{{s}}}" if best is not None and round(x, nd) == round(best, nd) else s


def best_of(vals, lower=False):
    v = [x for x in vals if x is not None]
    return None if not v else (min(v) if lower else max(v))


# ─────────────────────────────── Table：8B 主表 ───────────────────────────────
def tab_main():
    unc = summary("uncompressed", 0)
    L = [PREAMBLE, "\\begin{table*}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4pt}",
         "  \\begin{tabular}{l cc c ccc c ccc}", "    \\toprule",
         "    & \\multicolumn{2}{c}{Document} & Real-world & \\multicolumn{3}{c}{High resolution} & & "
         "\\multicolumn{3}{c}{Cost per image} \\\\",
         "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-7}\\cmidrule(lr){9-11}",
         "    Method & " + " & ".join(n for _, n in DS) + " & Avg.\\ (\\%) & TTFT$\\downarrow$ & "
         "Mem.$\\downarrow$ & TFLOPs$\\downarrow$ \\\\", "    \\midrule",
         "    \\multicolumn{11}{c}{\\textit{Upper bound: all visual tokens}} \\\\",
         f"    Uncompressed & " + " & ".join(fmt(x) for x in unc["raw"]) + " & 100.0 & "
         f"{fmt(unc['ttft'], 2)} & {fmt(unc['mem'], 2)} & {fmt(unc['flops'], 0)} \\\\"]
    for b in MAIN_BLOCKS:
        S = {m: summary(m, b) for m, _ in METHODS}
        kr = keep_ratio(b)
        L += ["    \\midrule",
              f"    \\multicolumn{{11}}{{c}}{{\\textit{{Budget $B={b}$ visual tokens"
              + (" (default)" if b == DEFAULT_B else "")
              + (f", $\\approx${kr:.0f}\\% of all visual tokens" if kr is not None else "") + "}} \\\\"]
        best_raw = [best_of([S[m]["raw"][i] for m, _ in METHODS]) for i in range(len(DS))]
        best = {k: best_of([S[m][k] for m, _ in METHODS], lower=k != "avg") for k in ("avg", "ttft", "mem", "flops")}
        for m, name in METHODS:
            s = S[m]
            cells = [bold_if(x, bb) for x, bb in zip(s["raw"], best_raw)]
            if m == "foveatile":                     # 跟該欄最強 baseline 的配對 bootstrap：↑ 顯著較好、↓ 顯著較差
                for i, (d, _) in enumerate(DS):
                    base = [(S[mm]["raw"][i], mm) for mm, _ in METHODS if mm != "foveatile" and S[mm]["raw"][i] is not None]
                    ours = unit_scores(d, "foveatile", b)
                    if base and ours is not None:
                        delta, lo, hi, _n = R.paired_delta(ours, unit_scores(d, max(base)[1], b))
                        cells[i] += "$^{\\uparrow}$" if lo > 0 else ("$^{\\downarrow}$" if hi < 0 else "")
            row1 = " & ".join(cells)
            row2 = " & ".join("{\\scriptsize " + fmt(x) + "}" for x in s["ret"])
            drop = "" if s["avg"] is None else f"{{\\scriptsize ($\\downarrow${100 - s['avg']:.1f})}}"
            color = "    \\rowcolor{oursbg}\n" if m == "foveatile" else ""
            L += [f"{color}    {name} & {row1} & {bold_if(s['avg'], best['avg'])} & & & \\\\",
                  f"{color}    & {row2} & {drop} & "
                  f"\\multirow{{-2}}{{*}}{{{bold_if(s['ttft'], best['ttft'], 2)}}} & "
                  f"\\multirow{{-2}}{{*}}{{{bold_if(s['mem'], best['mem'], 2)}}} & "
                  f"\\multirow{{-2}}{{*}}{{{bold_if(s['flops'], best['flops'], 0)}}} \\\\"]
    L += ["    \\bottomrule", "  \\end{tabular}",
          "  \\caption{\\textbf{Accuracy and cost on InternVL3.5-8B.} For each method, the first row is the score on "
          "each benchmark (ANLS for DocVQA and InfoVQA, CircularEval accuracy for HR-Bench, accuracy otherwise) and "
          "the second row is its percentage of the uncompressed model. Avg.\\ is the mean percentage over the six "
          "benchmarks. $^{\\uparrow}$/$^{\\downarrow}$: \\method is significantly better/worse than the strongest "
          "baseline on that benchmark (paired bootstrap, 95\\%); unmarked differences are not significant. Cost is the median time to first token (TTFT, s), the peak memory on top of the model weights (GB), "
          "and the prefill FLOPs, averaged over the six benchmarks (50 images each, one A6000). Best compressed "
          "result per budget in bold. Results for $B\\in\\{1024,2048,4096,5120\\}$ are in the supplementary "
          "material and in \\cref{fig:pareto}.}",
          "  \\label{tab:main}", "\\end{table*}"]
    open(f"{OUT_TAB}/tab_main.tex", "w").write("\n".join(L) + "\n")


# ─────────────────────────────── Table：模型尺寸 ───────────────────────────────
def tab_scale():
    L = [PREAMBLE, "\\begin{table*}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{4.5pt}",
         "  \\begin{tabular}{l cccccc c cc}", "    \\toprule",
         "    Method & " + " & ".join(n for _, n in DS) + " & Avg.\\ (\\%) & TTFT (s)$\\downarrow$ & "
         "Mem.\\ (GB)$\\downarrow$ \\\\"]
    for model in ("4B", "8B", "14B"):
        unc = summary("uncompressed", 0, model)
        S = {m: summary(m, DEFAULT_B, model) for m, _ in METHODS}
        best_ret = [best_of([S[m]["ret"][i] for m, _ in METHODS]) for i in range(len(DS))]
        best = {k: best_of([S[m][k] for m, _ in METHODS], lower=k != "avg") for k in ("avg", "ttft", "mem")}
        L += ["    \\midrule", f"    \\multicolumn{{10}}{{c}}{{\\textit{{InternVL3.5-{model}, $B={DEFAULT_B}$}}}} \\\\",
              "    Uncompressed (score) & " + " & ".join(fmt(x) for x in unc["raw"]) +
              f" & 100.0 & {fmt(unc['ttft'], 2)} & {fmt(unc['mem'], 2)} \\\\"]
        for m, _ in METHODS:
            s, name = S[m], SHORT[m]
            color = "    \\rowcolor{oursbg}\n" if m == "foveatile" else ""
            L.append(f"{color}    {name} & " + " & ".join(bold_if(x, bb) for x, bb in zip(s["ret"], best_ret)) +
                     f" & {bold_if(s['avg'], best['avg'])} & {bold_if(s['ttft'], best['ttft'], 2)} & "
                     f"{bold_if(s['mem'], best['mem'], 2)} \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}",
          "  \\caption{\\textbf{Model scale.} Percentage of the uncompressed score at $B=1536$ on InternVL3.5-4B, -8B, "
          "and -14B (the uncompressed rows give raw scores). \\method uses the hyperparameters calibrated on the 8B "
          "model without re-tuning; its attention layers are mapped by relative depth. Best compressed result per "
          "model in bold.}", "  \\label{tab:scale}", "\\end{table*}"]
    open(f"{OUT_TAB}/tab_scale.tex", "w").write("\n".join(L) + "\n")


# ─────────────────────────────── Table：HR-Bench 8K 兩種 tile 上限 ───────────────────────────────
def tab_hr8k():
    summ = json.load(open("results/scaling/summary.json"))
    rows = [("uncompressed", "Uncompressed")] + METHODS
    L = [PREAMBLE, "\\begin{table}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{3.2pt}",
         "  \\begin{tabular}{l ccc ccc}", "    \\toprule",
         "    & \\multicolumn{3}{c}{25 tiles} & \\multicolumn{3}{c}{188 tiles} \\\\",
         "    \\cmidrule(lr){2-4}\\cmidrule(lr){5-7}",
         "    Method & Acc. & TTFT & Mem. & Acc. & TTFT & Mem. \\\\", "    \\midrule"]
    vals = {}
    for m, _ in rows:
        a48 = score("hrbench_8k", m, DEFAULT_B)
        e48 = eff("hrbench_8k", m, DEFAULT_B)
        a200 = score("hrbench_8k", m, DEFAULT_B, root="results/runs_maxnum201")
        e200 = summ.get(f"201|{m}")
        vals[m] = (a48, e48[0] if e48 else None, e48[1] if e48 else None,
                   a200, float(e200["ttft_s_median"]) if e200 else None, float(e200["mem_delta_gb_mean"]) if e200 else None)
    best = [best_of([vals[m][i] for m, _ in METHODS], lower=i % 3 != 0) for i in range(6)]
    for m, _ in rows:
        v, name = vals[m], SHORT[m]
        cells = [bold_if(x, best[i] if m != "uncompressed" else None, 1 if i % 3 == 0 else 2) for i, x in enumerate(v)]
        color = "    \\rowcolor{oursbg}\n" if m == "foveatile" else ""
        L.append(f"{color}    {name} & " + " & ".join(cells) + " \\\\")
        if m == "uncompressed":
            L.append("    \\midrule")
    L += ["    \\bottomrule", "  \\end{tabular}",
          "  \\caption{\\textbf{Higher input resolution at a fixed budget} ($B=1536$) on HR-Bench 8K. Raising the tile "
          "cap from $N_{\\max}=47$ to 200 splits each image into 188 tiles instead of about 25. Acc.\\ is CircularEval accuracy, "
          "TTFT in seconds, Mem.\\ in GB.}", "  \\label{tab:hr8k}", "\\end{table}"]
    open(f"{OUT_TAB}/tab_hr8k.tex", "w").write("\n".join(L) + "\n")


# ─────────────────────────────── Table：題型拆分 + V* Bench 命中分析 ───────────────────────────────
def per_question(ds, method, budget, root="results/runs"):
    r = load_run(ds, method, budget, root=root)
    if r is None:
        return None
    d, t = r
    p = d["candidates"][t]
    return d, [float(R.canon_pred(str(p[i]), m, "letter") == R.gold_letter(m)) for i, m in enumerate(d["meta"])]


def by_category_circular(ds, method, budget):
    r = per_question(ds, method, budget)
    if r is None:
        return {}
    d, q = r
    g = defaultdict(list)
    for i, m in enumerate(d["meta"]):
        g[(m["category"], int(m["id"]) // 4)].append(q[i])
    out = {}
    for c in ("single", "cross"):
        v = [float(all(x)) for (cc, _), x in g.items() if cc == c and len(x) == 4]
        out[c] = 100 * st.mean(v) if v else None
    return out


def vstar_hits(budget):
    """FoveaTile 選中的細節 tile 是否涵蓋 V* Bench 的目標（每個物件 >= 50% 面積）。"""
    from PIL import Image
    from internvl_preprocess import dynamic_grid
    root = os.path.expanduser("~/datasets/vstar_bench")
    r = load_run("vstar", "foveatile", budget)
    if r is None:
        return None
    sel = r[0]["tile_selection"][r[1]]
    qs = [json.loads(l) for l in open(f"{root}/test_questions.jsonl")]
    hits = []
    for q, s in zip(qs, sel):
        path = f"{root}/{q['image']}"
        W, H = Image.open(path).size
        cols, rows = dynamic_grid(Image.new("RGB", (W, H)), 1, 47)
        tw, th = W / cols, H / rows
        cover = []
        for x, y, w, h in json.load(open(path.rsplit(".", 1)[0] + ".json"))["bbox"]:
            a = sum(max(0, min(x + w, (t % cols + 1) * tw) - max(x, t % cols * tw)) *
                    max(0, min(y + h, (t // cols + 1) * th) - max(y, t // cols * th)) for t in s)
            cover.append(a / max(1, w * h))
        hits.append(min(cover) >= 0.5)
    return hits


def tab_analysis():
    rows = [("uncompressed", "Uncompressed")] + METHODS
    hits = vstar_hits(DEFAULT_B)
    cols = []
    for m, _ in rows:
        b = 0 if m == "uncompressed" else DEFAULT_B
        h4, h8 = by_category_circular("hrbench_4k", m, b), by_category_circular("hrbench_8k", m, b)
        v = per_question("vstar", m, b)
        att = pos = hit = miss = None
        if v:
            d, q = v
            cat = [x["category"] for x in d["meta"]]
            att = 100 * st.mean(x for x, c in zip(q, cat) if c == "direct_attributes")
            pos = 100 * st.mean(x for x, c in zip(q, cat) if c == "relative_position")
            if hits:
                hit = 100 * st.mean(x for x, h in zip(q, hits) if h)
                miss = 100 * st.mean(x for x, h in zip(q, hits) if not h)
        cols.append((m, [h4.get("single"), h4.get("cross"), h8.get("single"), h8.get("cross"), att, pos, hit, miss]))
    best = [best_of([v[i] for m, v in cols if m != "uncompressed"]) for i in range(8)]
    n_hit = sum(hits) if hits else 0
    L = [PREAMBLE, "\\begin{table}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{2pt}",
         "  \\begin{tabular}{l cc cc cc cc}", "    \\toprule",
         "    & \\multicolumn{2}{c}{HR-4K} & \\multicolumn{2}{c}{HR-8K} & \\multicolumn{4}{c}{V$^*$ Bench} \\\\",
         "    \\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-9}",
         f"    Method & S & C & S & C & Attr. & Pos. & Hit & Miss \\\\",
         "    \\midrule"]
    for m, v in cols:
        name = SHORT[m]
        cells = [bold_if(x, best[i] if m != "uncompressed" else None) for i, x in enumerate(v)]
        color = "    \\rowcolor{oursbg}\n" if m == "foveatile" else ""
        L.append(f"{color}    {name} & " + " & ".join(cells) + " \\\\")
        if m == "uncompressed":
            L.append("    \\midrule")
    L += ["    \\bottomrule", "  \\end{tabular}",
          "  \\caption{\\textbf{Where \\method helps and where it fails} ($B=1536$). HR-Bench is split into single- (S) and "
          "cross-instance (C) questions (CircularEval), V$^*$ Bench into attribute and relative-position questions. "
          f"Hit / Miss split V$^*$ Bench by whether the detail tiles selected by \\method cover the target "
          f"({n_hit} hit, {len(hits) - n_hit if hits else 0} miss); every method is evaluated on the same split.}}",
          "  \\label{tab:analysis}", "\\end{table}"]
    open(f"{OUT_TAB}/tab_analysis.tex", "w").write("\n".join(L) + "\n")


# ─────────────────────────────── Table：held-out ablation ───────────────────────────────
CALIB = [("infovqa_train_calib", "InfoVQA"), ("mme_rw_calib", "MME-RW")]


def calib_units(ds, tag):
    f = f"results/calib/{ds}/{tag}_InternVL3_5-8B.json"
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    t = list(d["candidates"])[0]
    if len(d["candidates"][t]) < len(d["meta"]):
        return None
    return d, R.unit_scores(d, t, R.METRIC_OF(ds), None)


def calib_ttft(tag):
    xs = []
    for ds, _ in CALIB:
        v = None
        for f in (f"results/calib/metrics/metrics_{ds}.csv", f"results/calib/metrics/metrics_S_{ds}.csv"):
            if os.path.exists(f):
                for r in csv.DictReader(open(f)):
                    if r["method"] == tag:
                        v = float(r["ttft_s_median"])
        xs.append(v)
    return mean_or_none(xs)


S0_TABLE = {1024: 10, 1536: 9, 2048: 8, 2560: 7, 3072: 7}       # 校準前的舊規則 S（m = K/2 下）；預設 = S0/2
S_DEFAULT = {1024: 5, 1536: 4, 2048: 4, 2560: 3, 3072: 3}
LOCAL = {"Remote Sensing", "Monitoring", "Autonomous_Driving"}  # protocol 事先定義的局部細節題


def calib_stats(tag):
    """[InfoVQA %, MME-RW %, 局部細節 %, 平均 %, 最差任務 %, TTFT]；缺結果時 None。"""
    rets, groups, local = [], [], None
    for ds, _ in CALIB:
        r, ru = calib_units(ds, tag), calib_units(ds, "uncompressed")
        if r is None or ru is None:
            return None
        (d, u), ud = r, ru[1]
        rets.append(100 * st.mean(u.values()) / st.mean(ud.values()))
        if ds.startswith("infovqa"):
            groups.append(rets[-1])
            continue
        sub = defaultdict(list)
        for i, m in enumerate(d["meta"]):
            sub[m["category"].split("/")[0]].append(i)
        groups += [100 * st.mean(u[i] for i in ix) / st.mean(ud[i] for i in ix) for ix in sub.values()]
        li = [i for k, ix in sub.items() if k in LOCAL for i in ix]
        local = 100 * st.mean(u[i] for i in li) / st.mean(ud[i] for i in li)
    return rets + [local, st.mean(rets), min(groups), calib_ttft(tag)]


def tab_ablation():
    """每個變體在 b1024~b3072 五個 budget 上各算一次，再取平均（最差任務 = 每個 budget 的最差任務再平均）。"""
    B5 = [1024, 1536, 2048, 2560, 3072]
    variants = [(lambda b: f"foveatile_b{b}_m{b // 512}_S{S_DEFAULT[b]}", "\\method", True),
                (lambda b: f"foveatile_b{b}_m{b // 256}", "Coverage only, $m{=}K$", False),
                (lambda b: f"foveatile_b{b}_m1", "Thumbnail, $m{=}1$", False),
                (lambda b: f"foveatile_b{b}_m{b // 512}_S0", "Prior only, $S{=}0$", False),
                (lambda b: f"foveatile_b{b}_m{b // 512}", "Window $2\\times$", False),
                (lambda b: f"foveatile_b{b}_m{b // 512}_S{2 * S0_TABLE[b]}", "Window $4\\times$", False),
                (lambda b: f"foveatile_b{b}_m{b // 512}_S{S_DEFAULT[b]}_randprior", "Random prior", False)]
    L = [PREAMBLE, "\\begin{table}[t]", "  \\centering", "  \\footnotesize", "  \\setlength{\\tabcolsep}{2.6pt}",
         "  \\begin{tabular}{l ccc cc c}", "    \\toprule",
         "    Variant & Info & MME & Local & Avg. & Worst & TTFT \\\\", "    \\midrule"]
    for tag_of, name, ours in variants:
        rows = [calib_stats(tag_of(b)) for b in B5]
        agg = [None] * 6
        if all(rows):
            agg = [None if any(r[i] is None for r in rows) else st.mean(r[i] for r in rows) for i in range(6)]
        cells = [fmt(x) for x in agg[:5]] + [("--" if all(rows) else PEND) if agg[5] is None else f"{agg[5]:.2f}"]
        color = "    \\rowcolor{oursbg}\n" if ours else ""
        L.append(f"{color}    {name} & " + " & ".join(cells) + " \\\\")
    L += ["    \\bottomrule", "  \\end{tabular}",
          "  \\caption{\\textbf{Ablations} on the held-out calibration sets (Info: InfoVQA train; MME: MME-RealWorld "
          "without its Lite subset; 600 questions each), as a percentage of the uncompressed score, averaged over "
          "$B\\in\\{1024,1536,2048,2560,3072\\}$. Local: MME-RealWorld remote sensing, monitoring, and autonomous "
          "driving, where answers depend on small regions. Worst: lowest percentage among InfoVQA and the five "
          "MME-RealWorld subtasks, averaged over budgets. Window sizes are relative to the default $S$; TTFT in seconds.}",
          "  \\label{tab:ablation}", "\\end{table}"]
    open(f"{OUT_TAB}/tab_ablation.tex", "w").write("\n".join(L) + "\n")


# ─────────────────────────────── CSV（所有數字） ───────────────────────────────
def dump_csv():
    with open("results/paper/main_6ds.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "method", "budget"] + [d for d, _ in DS] + [f"ret_{d}" for d, _ in DS] +
                   ["avg_ret", "ttft_s", "mem_gb", "tflops"])
        for model in MODELS:
            for m, b in [("uncompressed", 0)] + [(m, b) for m, _ in METHODS for b in BUDGETS_ALL]:
                s = summary(m, b, model)
                if all(x is None for x in s["raw"]):
                    continue
                w.writerow([model, m, b] + [None if x is None else round(x, 2) for x in s["raw"] + s["ret"]] +
                           [None if s[k] is None else round(s[k], 3) for k in ("avg", "ttft", "mem", "flops")])


# ─────────────────────────────── 圖 ───────────────────────────────
def figures():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "mathtext.fontset": "dejavusans"})
    STY = [("foveatile", "FoveaTile (ours)", "#2a6fdb", "o"), ("ifprune", "IF-Prune", "#eb6834", "s"),
           ("fastv", "FastV", "#1f9e5a", "^"), ("sparsevlm", "SparseVLM", "#e0a100", "D"),
           ("visionzip", "VisionZip", "#d16a9b", "v")]

    # Pareto：平均保留率 vs TTFT / 記憶體（8B）
    S = {(m, b): summary(m, b) for m, *_ in STY for b in BUDGETS_ALL}
    unc = summary("uncompressed", 0)
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.75), sharey=True)
    for ax, key, xl in [(axes[0], "ttft", "Mean TTFT (s)"), (axes[1], "mem", "Mean memory increase (GB)")]:
        for m, lab, col, mk in reversed(STY):
            pts = [(S[(m, b)][key], S[(m, b)]["avg"]) for b in BUDGETS_ALL if S[(m, b)][key] is not None and S[(m, b)]["avg"] is not None]
            if not pts:
                continue
            xs, ys = zip(*pts)
            ours = m == "foveatile"
            ax.plot(xs, ys, color=col, marker=mk, ms=2.6 if not ours else 3.2, lw=1.4 if ours else 0.9,
                    label=lab, zorder=3 if ours else 2)
        d = S[("foveatile", DEFAULT_B)]
        if d[key] is not None:
            ax.scatter([d[key]], [d["avg"]], marker="*", s=55, color="#2a6fdb", edgecolor="white", lw=0.6, zorder=4)
        if unc[key] is not None:
            ax.scatter([unc[key]], [100], marker="X", s=18, color="#444444", zorder=4, label="Uncompressed")
        ax.axhline(100, color="#999999", ls="--", lw=0.6, zorder=1)
        ax.set_xlabel(xl, fontsize=6)
        ax.grid(color="#eeeeee", lw=0.5)
        ax.tick_params(labelsize=5.5, length=2)
    axes[0].set_ylabel("Mean % of uncompressed", fontsize=6)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ["FoveaTile (ours)", "IF-Prune", "FastV", "SparseVLM", "VisionZip", "Uncompressed"] if x in l]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False,
               fontsize=5.4, bbox_to_anchor=(0.53, 1.13), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.2, w_pad=0.6)
    fig.savefig(f"{OUT_FIG}/fig_pareto.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{OUT_FIG}/fig_pareto.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)

    # 解析度縮放：HR-Bench 8K，b1536
    summ = json.load(open("results/scaling/summary.json"))
    max_nums = [7, 25, 97, 145, 201]
    tiles = [int(float(summ[f"{n}|foveatile"]["raw_tiles_mean"])) for n in max_nums]
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.6))
    for ax, col, yl in [(axes[0], "ttft_s_median", "TTFT (s)"), (axes[1], "mem_delta_gb_mean", "Memory increase (GB)")]:
        for m, lab, c, mk in reversed(STY + [("uncompressed", "Uncompressed", "#444444", "X")]):
            ys = [float(summ[f"{n}|{m}"][col]) for n in max_nums]
            ours = m == "foveatile"
            ax.plot(tiles, ys, color=c, marker=mk, ms=2.6 if not ours else 3.2, lw=1.4 if ours else 0.9,
                    ls="--" if m == "uncompressed" else "-", label=lab, zorder=3 if ours else 2)
        top, ours_y = float(summ[f"201|uncompressed"][col]), float(summ[f"201|foveatile"][col])
        frac = 0.5 if col.startswith("ttft") else 0.28        # 記憶體面板：避開 188 塊附近的 baseline 線
        ax.annotate(f"{top / ours_y:.0f}$\\times$", (tiles[-1], ours_y + frac * (top - ours_y)), xytext=(-3, 0),
                    textcoords="offset points", ha="right", va="center", fontsize=6, color="#2a6fdb", fontweight="bold")
        ax.annotate("", xy=(tiles[-1], top), xytext=(tiles[-1], ours_y),
                    arrowprops=dict(arrowstyle="<->", color="#2a6fdb", lw=0.7, shrinkA=3, shrinkB=3))
        ax.set_xticks(tiles); ax.set_xticklabels([str(t) for t in tiles])
        ax.set_xlabel("Tiles per image $N$ (HR-Bench 8K)", fontsize=6)
        ax.set_ylabel(yl, fontsize=6)
        ax.grid(color="#eeeeee", lw=0.5)
        ax.tick_params(labelsize=5.5, length=2)
    h, l = axes[0].get_legend_handles_labels()
    order = [l.index(x) for x in ["FoveaTile (ours)", "IF-Prune", "FastV", "SparseVLM", "VisionZip", "Uncompressed"]]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, frameon=False,
               fontsize=5.4, bbox_to_anchor=(0.53, 1.15), handlelength=1.6, columnspacing=1.0)
    fig.tight_layout(pad=0.2, w_pad=0.8)
    fig.savefig(f"{OUT_FIG}/fig_scaling.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(f"{OUT_FIG}/fig_scaling.png", dpi=400, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


if __name__ == "__main__":
    tab_main(); tab_scale(); tab_hr8k(); tab_analysis(); tab_ablation(); dump_csv(); figures()
    n = sum(open(f"{OUT_TAB}/{f}").read().count(PEND) - 1 for f in os.listdir(OUT_TAB) if f.endswith(".tex"))
    print(f"tables -> {OUT_TAB}/, figures -> {OUT_FIG}/；待補的格子：{n}")
