"""
tile-score aggregation ablation：主圖（ANLS vs budget）+ 主表（ANLS / ΔANLS / 顯著性）。

吃的是 evaluate_anls.py --dump_csv 產生的 per-sample CSV，每個 budget 一份：
    anls_docvqa_internvl_2048.csv
    anls_docvqa_internvl_4096.csv
    anls_docvqa_internvl_5120.csv
每份欄位 = tag,idx,anls,gt,pred；tag 例如
    <references / uncompressed baseline>
    budget=2048_evict_info_density            (= baseline, mean 聚合)
    budget=2048_evict_info_density_max
    budget=2048_evict_info_density_topk_mean
    budget=2048_evict_info_density_mean_std
    budget=2048_evict_info_density_quantile

產出：
    <prefix>_curve.png / .pdf   主圖
    <prefix>_table.tex          booktabs 表（貼 paper）
    <prefix>_summary.csv        同表的數值版
    stdout                      人看的表

用法：
    python plot_agg_ablation.py
    python plot_agg_ablation.py --csv_glob "anls_docvqa_internvl_*.csv" --out_prefix docvqa_agg_ablation
"""
import re
import csv
import glob
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 聚合方式：內部 key -> (顯示名, tag 後綴)。順序 = 圖例/表格由上到下的順序。
AGGS = [
    ("mean",      "Mean (baseline)", ""),
    ("mean_std",  "Mean + Std",      "_mean_std"),
    ("topk_mean", "Top-k Mean",      "_topk_mean"),
    ("quantile",  "Quantile (0.9)",  "_quantile"),
    ("max",       "Max",             "_max"),
]
REF_KEYS = ("<references / uncompressed baseline>", "references")
STYLE = {   # 線條樣式：max 特別強調
    "mean":      dict(color="#7f7f7f", marker="o", lw=1.8, ls="-"),
    "mean_std":  dict(color="#8c9440", marker="s", lw=1.8, ls="-"),
    "topk_mean": dict(color="#1f77b4", marker="^", lw=1.8, ls="-"),
    "quantile":  dict(color="#9467bd", marker="D", lw=1.8, ls="-"),
    "max":       dict(color="#d62728", marker="*", lw=2.6, ls="-", ms=12),
}


def tag_to_agg(tag: str):
    if tag in REF_KEYS or "uncompressed" in tag or tag.startswith("<references"):
        return "REF"
    for key, _, suf in AGGS:
        if suf and tag.endswith(suf):
            return key
    return "mean"   # 沒有已知後綴 = baseline


def load_csv(path):
    """回傳 {agg_key: {idx: anls}}。"""
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            agg = tag_to_agg(row["tag"])
            out.setdefault(agg, {})[int(row["idx"])] = float(row["anls"])
    return out


def aligned(scores_by_idx_a, scores_by_idx_b):
    """取兩個 dict 共同的 idx，回傳兩條對齊的 np.array。"""
    common = sorted(set(scores_by_idx_a) & set(scores_by_idx_b))
    a = np.array([scores_by_idx_a[i] for i in common])
    b = np.array([scores_by_idx_b[i] for i in common])
    return a, b


def boot_mean_ci(x, n_boot, rng):
    """單樣本 bootstrap：回傳 (mean, lo, hi)，單位跟 x 相同。"""
    x = np.asarray(x)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    bm = x[idx].mean(axis=1)
    return x.mean(), np.percentile(bm, 2.5), np.percentile(bm, 97.5)


def boot_delta_ci(var, base, n_boot, rng):
    """paired bootstrap：Δ = mean(var) - mean(base)，回傳 (Δ, lo, hi)。"""
    d = np.asarray(var) - np.asarray(base)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    bd = d[idx].mean(axis=1)
    return d.mean(), np.percentile(bd, 2.5), np.percentile(bd, 97.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv_glob", default="anls_docvqa_internvl_*.csv")
    ap.add_argument("--budget_regex", default=r"(\d+)",
                    help="從檔名抽 budget 的 regex（第一個 group）")
    ap.add_argument("--out_prefix", default="docvqa_agg_ablation")
    ap.add_argument("--dataset_label", default="DocVQA validation (n=5349)")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    # ---- 讀所有 budget 的 CSV ----
    files = sorted(glob.glob(args.csv_glob))
    if not files:
        raise SystemExit(f"找不到符合 {args.csv_glob} 的檔案")
    data = {}   # budget -> {agg_key: {idx: anls}}
    for p in files:
        m = re.search(args.budget_regex, p.split("/")[-1])
        if not m:
            print(f"[skip] 檔名抽不出 budget: {p}")
            continue
        data[int(m.group(1))] = load_csv(p)
    budgets = sorted(data)
    print(f"budgets: {budgets}   (from {len(files)} csv)")

    # ---- 未壓縮 ceiling（各 budget 的 CSV 裡都一樣，取平均以防有缺）----
    ref_vals = []
    for b in budgets:
        if "REF" in data[b]:
            ref_vals.append(np.mean(list(data[b]["REF"].values())) * 100)
    ref_anls = float(np.mean(ref_vals)) if ref_vals else None

    # ---- 逐 (budget, agg) 算 ANLS / CI / Δ vs baseline ----
    #   rows[agg][budget] = dict(anls, lo, hi, exact, zero, n, d, dlo, dhi, sig)
    rows = {k: {} for k, _, _ in AGGS}
    for b in budgets:
        base_scores = data[b].get("mean")
        if base_scores is None:
            print(f"[warn] budget={b} 沒有 baseline(mean) 欄，跳過該 budget 的 Δ")
        for key, _, _ in AGGS:
            sb = data[b].get(key)
            if sb is None:
                continue
            vals = np.array(list(sb.values()))
            mean, lo, hi = boot_mean_ci(vals * 100, args.n_boot, rng)
            rec = dict(
                anls=mean, lo=lo, hi=hi,
                exact=float((vals >= 0.999).mean() * 100),
                zero=float((vals == 0.0).mean() * 100),
                n=len(vals), d=None, dlo=None, dhi=None, sig=False,
            )
            if base_scores is not None and key != "mean":
                va, ba = aligned(sb, base_scores)
                d, dlo, dhi = boot_delta_ci(va * 100, ba * 100, args.n_boot, rng)
                rec.update(d=d, dlo=dlo, dhi=dhi, sig=bool(dlo > 0 or dhi < 0))
            rows[key][b] = rec

    # ================= 主圖 =================
    fig, ax = plt.subplots(figsize=(6.6, 4.3), dpi=300)
    for key, label, _ in AGGS:
        xs = [b for b in budgets if b in rows[key]]
        if not xs:
            continue
        ys = [rows[key][b]["anls"] for b in xs]
        los = [rows[key][b]["lo"] for b in xs]
        his = [rows[key][b]["hi"] for b in xs]
        st = STYLE[key]
        ax.fill_between(xs, los, his, color=st["color"], alpha=0.13, linewidth=0)
        ax.plot(xs, ys, label=label, **st)

    if ref_anls is not None:
        ax.axhline(ref_anls, color="black", ls="--", lw=1.4, alpha=0.8)
        ax.text(budgets[-1], ref_anls + 0.6, f"Uncompressed  ({ref_anls:.1f})",
                ha="right", va="bottom", fontsize=9, style="italic")

    ax.set_xlabel("Visual-token budget", fontsize=11)
    ax.set_ylabel("DocVQA val ANLS (%)", fontsize=11)
    ax.set_xticks(budgets)
    ax.set_xticklabels([str(b) for b in budgets])
    ax.grid(True, alpha=0.3)
    ax.set_title(f"Tile-score aggregation vs. budget\n{args.dataset_label} · shaded = 95% CI (bootstrap)",
                 fontsize=10)
    ax.legend(fontsize=8.5, loc="lower right", framealpha=0.95)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out_prefix}_curve.{ext}", bbox_inches="tight")
    print(f"[saved] {args.out_prefix}_curve.png / .pdf")

    # ================= 主表（stdout + csv + tex）=================
    def cell(rec, is_base):
        if rec is None:
            return "--"
        if is_base:
            return f"{rec['anls']:.2f}"
        star = "*" if rec["sig"] else ""
        return f"{rec['anls']:.2f} ({rec['d']:+.2f}){star}"

    # 每個 budget 的最佳（非 baseline、非 ref）用於粗體
    best = {b: max((rows[k][b]["anls"] for k, _, _ in AGGS if b in rows[k]), default=None)
            for b in budgets}

    hdr = ["Aggregation"] + [f"b={b}" for b in budgets]
    print("\n" + "  ".join(f"{h:<22}" if i == 0 else f"{h:>18}" for i, h in enumerate(hdr)))
    print("-" * (22 + 20 * len(budgets)))
    if ref_anls is not None:
        print(f"{'Uncompressed (ceiling)':<22}" + "".join(f"{ref_anls:>18.2f}" for _ in budgets))
    for key, label, _ in AGGS:
        is_base = key == "mean"
        cells = [cell(rows[key].get(b), is_base) for b in budgets]
        print(f"{label:<22}" + "".join(f"{c:>18}" for c in cells))
    print("\n*  = paired bootstrap 95% CI of ΔANLS vs. baseline excludes 0")

    # summary csv
    with open(f"{args.out_prefix}_summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["budget", "aggregation", "anls", "ci_lo", "ci_hi",
                    "exact_pct", "zero_pct", "n", "delta_vs_base", "delta_lo", "delta_hi", "significant"])
        for b in budgets:
            if ref_anls is not None and "REF" in data[b]:
                rv = np.array(list(data[b]["REF"].values()))
                w.writerow([b, "uncompressed", f"{rv.mean()*100:.4f}", "", "",
                            f"{(rv>=0.999).mean()*100:.2f}", f"{(rv==0).mean()*100:.2f}", len(rv),
                            "", "", "", ""])
            for key, label, _ in AGGS:
                r = rows[key].get(b)
                if r is None:
                    continue
                w.writerow([b, key, f"{r['anls']:.4f}", f"{r['lo']:.4f}", f"{r['hi']:.4f}",
                            f"{r['exact']:.2f}", f"{r['zero']:.2f}", r["n"],
                            "" if r["d"] is None else f"{r['d']:.4f}",
                            "" if r["dlo"] is None else f"{r['dlo']:.4f}",
                            "" if r["dhi"] is None else f"{r['dhi']:.4f}",
                            int(r["sig"])])
    print(f"[saved] {args.out_prefix}_summary.csv")

    # latex (booktabs)
    def tex_cell(rec, is_base, b):
        if rec is None:
            return "--"
        if is_base:
            return f"{rec['anls']:.2f}"
        s = f"{rec['anls']:.2f}\\,({rec['d']:+.2f})"
        if rec["sig"]:
            s += "$^{*}$"
        if best[b] is not None and abs(rec["anls"] - best[b]) < 1e-6:
            s = f"\\textbf{{{s}}}"
        return s

    lines = [
        "\\begin{tabular}{l" + "c" * len(budgets) + "}",
        "\\toprule",
        "Aggregation & " + " & ".join(f"$b{{=}}{b}$" for b in budgets) + " \\\\",
        "\\midrule",
    ]
    if ref_anls is not None:
        lines.append("Uncompressed (ceiling) & " + " & ".join(f"{ref_anls:.2f}" for _ in budgets) + " \\\\")
        lines.append("\\midrule")
    for key, label, _ in AGGS:
        is_base = key == "mean"
        lines.append(f"{label} & " + " & ".join(tex_cell(rows[key].get(b), is_base, b) for b in budgets) + " \\\\")
    lines += [
        "\\bottomrule",
        "\\end{tabular}",
        "% $^{*}$: paired bootstrap 95\\% CI of $\\Delta$ANLS vs. baseline excludes 0.",
    ]
    with open(f"{args.out_prefix}_table.tex", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[saved] {args.out_prefix}_table.tex")


if __name__ == "__main__":
    main()
