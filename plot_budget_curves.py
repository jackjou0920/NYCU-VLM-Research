"""Score vs. token budget, one small panel per dataset (InternVL3.5-8B main table)."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/paper/main_internvl.csv"
OUT = "figures/budget_curves"

DATASETS = [("docvqa", "DocVQA"), ("infovqa", "InfoVQA"), ("hrbench_4k", "HR-Bench 4K"),
            ("mme_rw_lite", "MME-RW Lite"), ("vstar", "V*")]
# (key, label, color, marker) -- FoveaTile first so it gets slot 1 and draws on top
METHODS = [("svma", "FoveaTile (ours)", "#2a78d6", "o"),
           ("ifprune", "IF-Prune", "#eb6834", "s"),
           ("fastv", "FastV", "#1baf7a", "^"),
           ("sparsevlm", "SparseVLM", "#eda100", "D"),
           ("visionzip", "VisionZip", "#e87ba4", "v")]
BUDGETS = [1024, 1536, 2048, 2560, 3072]

scores = defaultdict(dict)
full = {}
with open(CSV) as f:
    for r in csv.DictReader(f):
        if r["method"] == "uncompressed":
            full[r["dataset"]] = float(r["score"])
        else:
            scores[(r["dataset"], r["method"])][int(r["budget"])] = float(r["score"])

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(10, 6))
axes = axes.ravel()
for ax, (ds, title) in zip(axes, DATASETS):
    for key, label, color, marker in reversed(METHODS):
        ys = [scores[(ds, key)][b] for b in BUDGETS]
        ax.plot(BUDGETS, ys, color=color, marker=marker, ms=5, lw=2 if key == "svma" else 1.5,
                label=label, zorder=3 if key == "svma" else 2)
    ax.axhline(full[ds], color="#52514e", ls="--", lw=1.2, label="Uncompressed", zorder=1)
    ax.set_title(title, fontsize=10)
    ax.set_xticks(BUDGETS)
    ax.set_xticklabels([str(b) for b in BUDGETS])
    ax.set_xlabel("Token budget")
    ax.grid(axis="y", color="#e5e4e0", lw=0.6)
    ax.tick_params(colors="#52514e")
for ax in axes[[0, 3]]:
    ax.set_ylabel("Score")

handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(m[1]) for m in METHODS] + [labels.index("Uncompressed")]
# sixth slot holds the legend
axes[5].axis("off")
axes[5].legend([handles[i] for i in order], [labels[i] for i in order], loc="center", frameon=False,
               fontsize=10)
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}.pdf / .png")
