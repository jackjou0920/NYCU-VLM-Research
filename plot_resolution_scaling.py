"""Resolution scaling on HR-Bench 8K at b1536: TTFT and memory vs. raw tile count."""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUMMARY = "results/scaling/summary.json"
OUT = "figures/resolution_scaling"
LOG = "--log" in sys.argv  # log-log axes variant

# (key in summary.json, label, color, marker) -- same encoding as plot_budget_curves.py
METHODS = [("stream", "FoveaTile (ours)", "#2a78d6", "o"),
           ("ifprune", "IF-Prune", "#eb6834", "s"),
           ("fastv", "FastV", "#1baf7a", "^"),
           ("sparsevlm", "SparseVLM", "#eda100", "D"),
           ("visionzip", "VisionZip", "#e87ba4", "v"),
           ("uncompressed", "Uncompressed", "#52514e", "X")]
# max_num settings whose raw tile count is distinct (13 and 49 hit the image's native cap: same as 7 / 25)
MAX_NUMS = [7, 25, 97, 145, 201]
PANELS = [("ttft_s_median", "TTFT (s)"), ("mem_delta_gb_mean", "Memory increase (GB)")]

summ = json.load(open(SUMMARY))
tiles = [int(float(summ[f"{n}|stream"]["raw_tiles_mean"])) for n in MAX_NUMS]

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
for ax, (col, ylabel) in zip(axes, PANELS):
    for key, label, color, marker in reversed(METHODS):
        ys = [float(summ[f"{n}|{key}"][col]) for n in MAX_NUMS]
        ours, full = key == "stream", key == "uncompressed"
        ax.plot(tiles, ys, color=color, marker=marker, ms=5, lw=2 if ours else 1.5,
                ls="--" if full else "-", label=label, zorder=3 if ours else 2)
    # ratio callout at the largest input: uncompressed / FoveaTile
    n = MAX_NUMS[-1]
    ours_y, full_y = float(summ[f"{n}|stream"][col]), float(summ[f"{n}|uncompressed"][col])
    ax.annotate("", xy=(tiles[-1], full_y), xytext=(tiles[-1], ours_y),
                arrowprops=dict(arrowstyle="<->", color="#2a78d6", lw=1, shrinkA=6, shrinkB=6))
    mid = (full_y * ours_y) ** 0.5 if LOG else (full_y + ours_y) / 2
    ax.annotate(f"{full_y / ours_y:.0f}×", (tiles[-1], mid), xytext=(-5, 0), textcoords="offset points",
                ha="right", va="center", fontsize=9, color="#2a78d6", fontweight="bold")
    if LOG:
        ax.set_xscale("log")
        ax.set_yscale("log")
    ax.set_xticks(tiles)
    ax.set_xticklabels([str(t) for t in tiles])
    ax.minorticks_off()
    ax.set_xlabel("Raw tile count (HR-Bench 8K)")
    ax.set_ylabel(ylabel)
    ax.grid(color="#e5e4e0", lw=0.6)
    ax.tick_params(colors="#52514e")

handles, labels = axes[0].get_legend_handles_labels()
order = [labels.index(m[1]) for m in METHODS]
fig.legend([handles[i] for i in order], [labels[i] for i in order], loc="upper center", ncol=6,
           frameon=False, fontsize=8, bbox_to_anchor=(0.5, 1.06))
fig.tight_layout()
out = OUT + ("_log" if LOG else "")
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", dpi=200, bbox_inches="tight")
print(f"saved {out}.pdf / .png")
