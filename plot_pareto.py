"""Pareto plots over the 5 datasets: mean retention (score / uncompressed) vs. mean TTFT and vs. mean memory."""
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CSV = "results/paper/main_internvl.csv"
OUT = "figures/pareto_retention"

# (key, label, color, marker) -- same encoding as plot_budget_curves.py
METHODS = [("foveatile", "FoveaTile (ours)", "#2a78d6", "o"),
           ("ifprune", "IF-Prune", "#eb6834", "s"),
           ("fastv", "FastV", "#1baf7a", "^"),
           ("sparsevlm", "SparseVLM", "#eda100", "D"),
           ("visionzip", "VisionZip", "#e87ba4", "v")]
BUDGETS = [1024, 1536, 2048, 2560, 3072]
DEFAULT_BUDGET = 1536  # frozen FoveaTile operating point

rows = list(csv.DictReader(open(CSV)))
full = {r["dataset"]: r for r in rows if r["method"] == "uncompressed"}
acc = defaultdict(list)
for r in rows:
    if r["method"] != "uncompressed":
        ret = 100 * float(r["score"]) / float(full[r["dataset"]]["score"])
        acc[(r["method"], int(r["budget"]))].append((float(r["ttft_s"]), float(r["mem_delta_gb"]), ret))
mean = {k: tuple(sum(x) / len(v) for x in zip(*v)) for k, v in acc.items()}
full_mean = tuple(sum(float(r[c]) for r in full.values()) / len(full) for c in ("ttft_s", "mem_delta_gb"))


def label_offset(panel, b):
    # TTFT: above-left at low budgets, below-right at high ones, to stay off FastV / IF-Prune
    if panel == 0:
        return (-6, 5, "right") if b <= 1536 else (2, -12, "left")
    return (6, -3, "left")  # memory: FoveaTile sits alone left of the 1.3 GB FastV/SparseVLM column


plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
XLABELS = ["Mean TTFT (s)  ← faster", "Mean memory increase (GB)  ← less"]
for i, ax in enumerate(axes):
    for key, label, color, marker in reversed(METHODS):
        xs, ys = zip(*[(mean[(key, b)][i], mean[(key, b)][2]) for b in BUDGETS])
        ours = key == "foveatile"
        ax.plot(xs, ys, color=color, marker=marker, ms=5, lw=2 if ours else 1.5, label=label,
                zorder=3 if ours else 2)
        if ours:
            for b, x, y in zip(BUDGETS, xs, ys):
                dx, dy, ha = label_offset(i, b)
                ax.annotate(str(b), (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha,
                            fontsize=7, color="#52514e")
    d = mean[("foveatile", DEFAULT_BUDGET)]
    ax.scatter([d[i]], [d[2]], marker="*", s=260, color="#2a78d6", edgecolor="white", lw=1.2, zorder=4,
               label=f"FoveaTile default (b{DEFAULT_BUDGET})")
    ax.scatter([full_mean[i]], [100], marker="X", s=70, color="#52514e", zorder=4, label="Uncompressed")
    ax.axhline(100, color="#52514e", ls="--", lw=1, zorder=1)
    ax.set_xlabel(XLABELS[i])
    ax.grid(color="#e5e4e0", lw=0.6)
    ax.tick_params(colors="#52514e")
axes[0].set_ylabel("Mean retention (% of uncompressed)")
axes[1].tick_params(labelleft=True)

handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(m[1]) for m in METHODS] + [labels.index(f"FoveaTile default (b{DEFAULT_BUDGET})"),
                                                 labels.index("Uncompressed")]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=7,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {OUT}.pdf / .png")
