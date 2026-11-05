"""Intro 的 teaser（單欄 3.35 in）：fig_teaser.pdf
  上：剪枝方法「先全部編碼再丟」vs FoveaTile「先看再編碼」（V* 範例圖，選擇結果來自實際執行的 v1.0，b1536）
  下：HR-Bench 8K、b1536，tile 數 7 → 188 時的 peak memory 與每張 GPU 可同時服務的請求數（實測，同 fig_scaling）
資料：method_examples_vstar_b1536.json（見 make_method_figures.py）、results/scaling/、results/concurrency/
用法：python paper/figures/make_teaser.py --idx 51
"""
import argparse, json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, FancyArrowPatch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "results", "paper"))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "dejavusans", "axes.spines.top": False, "axes.spines.right": False})
C_OURS, C_BASE, C_GREY, C_TXT = "#2a6fdb", "#eb6834", "#9aa0a6", "#333333"
VSTAR = os.path.expanduser("~/datasets/vstar_bench")

ap = argparse.ArgumentParser()
ap.add_argument("--examples", default=os.path.join(HERE, "method_examples_vstar_b1536.json"))
ap.add_argument("--idx", type=int, default=51)
args = ap.parse_args()

r = json.load(open(args.examples))[args.idx]
q = [x for x in (json.loads(l) for l in open(f"{VSTAR}/test_questions.jsonl"))
     if x["category"] == "direct_attributes"][args.idx]
img = Image.open(f"{VSTAR}/{q['image']}").convert("RGB")
W, H = img.size
dc, dr = r["detail_grid"]
selected = set(r["selected"])

FW, FH = 3.35, 3.05
fig = plt.figure(figsize=(FW, FH))


def image_ax(x, y, w, im, tiles=None, encoded=None, sel_color=None):
    """figure 座標 (x, y) 左下、寬 w（figure 比例）的影像；tiles=(cols, rows) 畫網格，
    encoded=None 全部算編碼，否則只有 encoded 裡的 tile 編碼、其他蓋灰。"""
    iw, ih = im.size
    h = w * FW / FH * ih / iw
    ax = fig.add_axes([x, y, w, h])
    ax.imshow(im, extent=(0, iw, ih, 0))
    ax.set_xlim(0, iw); ax.set_ylim(ih, 0); ax.axis("off")
    if tiles:
        c, rr = tiles
        tw, th = iw / c, ih / rr
        for t in range(c * rr):
            x0, y0 = (t % c) * tw, (t // c) * th
            if encoded is not None and t not in encoded:
                ax.add_patch(Rectangle((x0, y0), tw, th, fc="white", ec="none", alpha=0.72))
            ax.add_patch(Rectangle((x0, y0), tw, th, fc="none", ec="white", lw=0.35))
        if encoded is not None:
            for t in encoded:
                ax.add_patch(Rectangle(((t % c) * tw, (t // c) * th), tw, th, fc="none", ec=sel_color, lw=0.9))
    return h


def box(x, y, w, h, s, color, fill):
    fig.patches.append(FancyBboxPatch((x, y), w, h, transform=fig.transFigure, boxstyle="round,pad=0,rounding_size=0.012",
                                      fc=fill, ec=color, lw=0.7))
    fig.text(x + w / 2, y + h / 2, s, ha="center", va="center", fontsize=5.1, color=C_TXT, linespacing=1.15)


def arrow(x0, x1, y, color=C_TXT, label=None):
    fig.patches.append(FancyArrowPatch((x0, y), (x1, y), transform=fig.transFigure, arrowstyle="-|>",
                                       mutation_scale=5, lw=0.6, color=color))
    if label:
        fig.text((x0 + x1) / 2, y + 0.012, label, ha="center", va="bottom", fontsize=4.6, color=color)


# ── (a) 兩條流程：欄位 x 座標（figure 比例）──
IX, IW = 0.125, 0.175                    # 細節網格影像
VX, VW = 0.345, 0.17                     # ViT 框
LX, LW = 0.56, 0.19                      # LLM 框
MX, MW = 0.795, 0.195                    # 記憶體框
BH = 0.105


def pipeline(yc, title, col, fill, vit, llm, mem, encoded=None):
    fig.text(0.01, yc + 0.075, title, fontsize=5.9, color=col, fontweight="bold", va="bottom")
    image_ax(IX, yc - 0.0645, IW, img, tiles=(dc, dr), encoded=encoded, sel_color=col)
    arrow(IX + IW + 0.005, VX - 0.005, yc)
    box(VX, yc - BH / 2, VW, BH, vit, col, fill)
    arrow(VX + VW + 0.005, LX - 0.005, yc)
    box(LX, yc - BH / 2, LW, BH, llm, col, fill)
    arrow(LX + LW + 0.005, MX - 0.005, yc)
    box(MX, yc - BH / 2, MW, BH, mem, col, "white")


pipeline(0.845, "Prune after encoding (FastV, SparseVLM, IF-Prune)", C_BASE, "#fdf0e8",
         f"ViT: all\n{dc * dr} tiles", "LLM: all tokens,\nthen prune", "peak memory\ngrows with $N$")
pipeline(0.64, r"$\bf{FoveaTile}$: look before you encode", C_OURS, "#e8f0fc",
         f"ViT: {len(selected)} of\n{dc * dr} tiles", "LLM: budget\ntokens only", "peak memory\nconstant",
         encoded=selected)
thumb = img.resize((224, 224 * H // W))
image_ax(0.01, 0.64 - 0.025, 0.08, thumb)
fig.text(0.05, 0.64 - 0.032, "coverage view\n→ prior", ha="center", va="top", fontsize=4.5, color=C_OURS, linespacing=1.1)
arrow(0.093, IX - 0.004, 0.64, C_OURS)

# ── (b) 隨 tile 數的成本（實測）──
import build_paper_results as P
os.chdir(ROOT)
max_nums = [7, 25, 97, 145, 201]
sc = P.scaling_points(max_nums)
tiles = [int(sc[("foveatile", n)]["tiles"]) for n in max_nums]
STY = [("uncompressed", "Uncompressed", "#444444", "X", "--"), ("ifprune", "IF-Prune", "#eb6834", "s", "-"),
       ("fastv", "FastV", "#1f9e5a", "^", "-"), ("foveatile", "FoveaTile (ours)", C_OURS, "o", "-")]
axes = [fig.add_axes([0.1, 0.175, 0.36, 0.33]), fig.add_axes([0.61, 0.175, 0.36, 0.33])]
for ax, key, yl in [(axes[0], "mem", "Memory increase (GB)"), (axes[1], "conc", "Requests per GPU")]:
    for m, lab, c, mk, ls in STY:
        ours = m == "foveatile"
        ax.plot(tiles, [sc[(m, n)][key] for n in max_nums], color=c, marker=mk, ms=2.4 if not ours else 3.0,
                lw=1.4 if ours else 0.8, ls=ls, label=lab, zorder=3 if ours else 2)
    ax.set_xticks([t for t in tiles if t != tiles[1]])
    ax.set_xlabel("Tiles per image $N$", fontsize=5.4, labelpad=1)
    ax.set_ylabel(yl, fontsize=5.4, labelpad=1)
    ax.grid(color="#eeeeee", lw=0.5)
    ax.tick_params(labelsize=4.9, length=2, pad=1.5)
o, u, f = (sc[(m, 201)]["conc"] for m in ("foveatile", "uncompressed", "fastv"))
axes[1].annotate(f"{o / f:.1f}$\\times$ FastV\n{o / u:.0f}$\\times$ uncomp.", (tiles[-1], o), xytext=(0, -6),
                 textcoords="offset points", ha="right", va="top", fontsize=4.9, color=C_OURS, fontweight="bold")
mo, mu = sc[("foveatile", 201)]["mem"], sc[("uncompressed", 201)]["mem"]
axes[0].annotate(f"{mu / mo:.0f}$\\times$", (tiles[-1], mo + 0.25 * (mu - mo)), xytext=(-3, 0), textcoords="offset points",
                 ha="right", va="center", fontsize=5.2, color=C_OURS, fontweight="bold")
axes[0].annotate("", xy=(tiles[-1], mu), xytext=(tiles[-1], mo),
                 arrowprops=dict(arrowstyle="<->", color=C_OURS, lw=0.6, shrinkA=2, shrinkB=2))
h, l = axes[0].get_legend_handles_labels()
fig.legend(h[::-1], l[::-1], loc="lower center", ncol=4, frameon=False, fontsize=4.9, handlelength=1.6,
           columnspacing=0.9, bbox_to_anchor=(0.5, 0.045))

out = os.path.join(HERE, "fig_teaser")
fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(out + ".png", dpi=400, bbox_inches="tight", pad_inches=0.01)
print("saved", out + ".pdf")
