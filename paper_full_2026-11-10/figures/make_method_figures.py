"""Method 章節的兩張圖（數值全部來自實際執行的 FoveaTile v1.0，b1536）：
  fig_overview.pdf  ：Overview（figure*，全寬 7 in）
  fig_boundary.pdf  ：prior 排名 → 直接保留 / 打分窗口 / 丟棄，以及 q_attn 的前綴 KV 重用（單欄 3.3 in）
資料：method_examples_vstar_b1536.json（V* direct_attributes 前 60 題的 prior / 選擇結果），由
python paper/figures/extract_method_example.py --n 60 --out paper/figures 產生後改名（需要 GPU，約 3 分鐘）。
用法：python paper/figures/make_method_figures.py --idx 51
"""
import argparse, json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, FancyBboxPatch
from matplotlib.lines import Line2D
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "mathtext.fontset": "dejavusans"})
C_COV, C_DIR, C_WIN, C_PICK, C_DROP, C_TXT, C_GREY = "#2a6fdb", "#1f9e5a", "#e08a00", "#c0392b", "#9aa0a6", "#555555", "#d9d9d9"
VSTAR = os.path.expanduser("~/datasets/vstar_bench")

ap = argparse.ArgumentParser()
ap.add_argument("--examples", default=os.path.join(HERE, "method_examples_vstar_b1536.json"))
ap.add_argument("--idx", type=int, default=51)
ap.add_argument("--budget", type=int, default=1536)
args = ap.parse_args()

r = json.load(open(args.examples))[args.idx]
q = [x for x in (json.loads(l) for l in open(f"{VSTAR}/test_questions.jsonl"))
     if x["category"] == "direct_attributes"][args.idx]
img = Image.open(f"{VSTAR}/{q['image']}").convert("RGB")
bbox = json.load(open(f"{VSTAR}/{q['image'].rsplit('.', 1)[0]}.json"))["bbox"][0]
W, H = img.size
dc, dr = r["detail_grid"]; cc, cr = r["cover_grid"]
N, m, K = dc * dr, cc * cr, args.budget // 256
k = K - m
a_, b_ = r["boundary_a_b"]; S = a_ + b_
direct, window, selected = set(r["direct_kept"]), set(r["window"]), set(r["selected"])
qscore = {int(t): v for t, v in r["candidate_scores"].items()}
rank = {t: i for i, t in enumerate(r["prior_ranked"])}
prior = r["prior"]
target = int((bbox[1] + bbox[3] / 2) // (H / dr)) * dc + int((bbox[0] + bbox[2] / 2) // (W / dc))
question = q["text"].splitlines()[0]
options = {ln[1]: ln[4:] for ln in q["text"].splitlines() if ln.startswith("(")}
answer = options.get(r["pred"].strip()[:1], r["pred"])
correct = r["pred"].strip()[:1] == r["answer"]


def status(t):
    if t in direct:
        return "direct"
    if t in window:
        return "pick" if t in selected else "win"
    return "drop"


class Panel:
    """figure 座標裡的一個矩形；所有東西用面板內的相對座標 (0..1) 擺。"""

    def __init__(self, fig, x, y, w, h):
        self.fig, self.x, self.y, self.w, self.h = fig, x, y, w, h
        self.fw, self.fh = fig.get_size_inches()

    def X(self, rx):
        return self.x + rx * self.w

    def Y(self, ry):
        return self.y + ry * self.h

    def frame(self, title, color):
        self.fig.patches.append(FancyBboxPatch((self.x, self.y), self.w, self.h, transform=self.fig.transFigure,
                                               boxstyle="round,pad=0,rounding_size=0.01", fc="none", ec=color, lw=0.8))
        self.text(0.03, 0.965, title, color=color, fontsize=6.6, fontweight="bold", va="top")

    def text(self, rx, ry, s, **kw):
        kw.setdefault("fontsize", 5.6)
        return self.fig.text(self.X(rx), self.Y(ry), s, **kw)

    def image(self, rx, ry_top, rw, im, aspect=None):
        """寬 rw（面板比例）、上緣在 ry_top 的影像；回傳 (axes, 下緣的 ry)。"""
        aspect = aspect or im.size[1] / im.size[0]
        w = rw * self.w
        h = w * self.fw / self.fh * aspect
        ax = self.fig.add_axes([self.X(rx), self.Y(ry_top) - h, w, h])
        ax.imshow(im, extent=(0, im.size[0], im.size[1], 0)); ax.set_xticks([]); ax.set_yticks([])
        return ax, ry_top - h / self.h

    def rect(self, rx, ry, rw, rh, **kw):
        self.fig.patches.append(Rectangle((self.X(rx), self.Y(ry)), rw * self.w, rh * self.h,
                                          transform=self.fig.transFigure, **kw))


def tile_box(t):
    c, rr = t % dc, t // dc
    return c * W / dc, rr * H / dr, W / dc, H / dr


def spines(ax, color, lw=1.0):
    for s in ax.spines.values():
        s.set_color(color); s.set_linewidth(lw)


def arrow(fig, p0, p1, color="#444444", **kw):
    fig.patches.append(FancyArrowPatch(p0, p1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=7,
                                       lw=0.8, color=color, **kw))


# ═════════════════════════════════════════════════════════════════════════════
# Figure 1：Overview
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(7.0, 2.55))
P0 = Panel(fig, 0.000, 0.03, 0.150, 0.94)
P1 = Panel(fig, 0.165, 0.03, 0.255, 0.94)
P2 = Panel(fig, 0.435, 0.03, 0.270, 0.94)
P3 = Panel(fig, 0.718, 0.03, 0.278, 0.94)

# ── input ──
P0.text(0.5, 0.965, "Input", ha="center", va="top", fontsize=6.6, fontweight="bold")
P0.text(0.5, 0.88, f"{W}$\\times${H} image\n{N} detail tiles", ha="center", va="top", color=C_TXT)
ax, yb = P0.image(0.0, 0.70, 1.0, img)
ax.add_patch(Rectangle((bbox[0] - 30, bbox[1] - 30), bbox[2] + 60, bbox[3] + 60, fill=False, ec=C_PICK, lw=0.8))
spines(ax, "white", 0)
P0.text(0.5, yb - 0.06, f"“{question}”", ha="center", va="top", style="italic", wrap=True)
P0.text(0.5, 0.16, f"budget $B$={args.budget}\n$K$={K} tiles = {m} coverage\n+ {k} detail", ha="center",
        va="top", color=C_TXT)

# ── ① coverage tiles → prior ──
P1.frame("① Coverage tiles → prior", C_COV)
crops = [img.crop((t % cc * W / cc, t // cc * H / cr, (t % cc + 1) * W / cc, (t // cc + 1) * H / cr)).resize((448, 448))
         for t in range(m)]
cw = 0.28
for t, im in enumerate(crops):
    ax, yb = P1.image(0.5 - (m * cw + (m - 1) * 0.03) / 2 + t * (cw + 0.03), 0.89, cw, im)
    spines(ax, C_COV, 1.2)
P1.text(0.5, yb - 0.025, f"{m} coverage tiles: whole image at low\nresolution, always kept", ha="center", va="top",
        color=C_COV)
P1.text(0.5, yb - 0.135, "one partial LLM pass on [$x_b$; $V^c$; $x_a$]:\nattention from question $x_a$, layers 21–28",
        ha="center", va="top")
ax, yb2 = P1.image(0.2, yb - 0.255, 0.6, img)
for t in range(N):
    x0, y0, w0, h0 = tile_box(t)
    ax.add_patch(Rectangle((x0, y0), w0, h0, fc=(0.85, 0.15, 0.1, 0.8 * prior[t]), ec="white", lw=0.4))
spines(ax, "white", 0)
P1.text(0.5, yb2 - 0.02, "prior $\\pi_t$ per detail tile (darker = higher)", ha="center", va="top")

# ── ② selection ──
P2.frame("② Select detail tiles before the ViT", C_DIR)
ax, yb = P2.image(0.06, 0.88, 0.88, img)
for t in range(N):
    x0, y0, w0, h0 = tile_box(t)
    st = status(t)
    if st == "drop":
        ax.add_patch(Rectangle((x0, y0), w0, h0, fc=(1, 1, 1, 0.62), ec="white", lw=0.5))
        continue
    col = {"direct": C_DIR, "pick": C_DIR, "win": C_WIN}[st]
    ax.add_patch(Rectangle((x0 + 14, y0 + 14), w0 - 28, h0 - 28, fill=False, ec=col, lw=1.3,
                           ls="--" if st == "win" else "-"))
    ax.text(x0 + 30, y0 + 30, f"{rank[t] + 1}", color="white", fontsize=4.4, va="top", ha="left", fontweight="bold",
            bbox=dict(boxstyle="square,pad=0.12", fc=col, ec="none", alpha=0.9))
ax.add_patch(Rectangle((bbox[0] - 30, bbox[1] - 30), bbox[2] + 60, bbox[3] + 60, fill=False, ec=C_PICK, lw=0.8))
spines(ax, "white", 0)
legend = [(C_DIR, "-", f"kept directly: prior top $k{{-}}a$ = {k - a_}"),
          (C_WIN, "--", f"boundary window: $S$ = {S}, scored by $q_t$"),
          (C_DIR, "-", f"… of which top $a$ = {a_} are kept"),
          (None, None, f"discarded: {N - (k - a_) - S} tiles, never encoded")]
for i, (col, ls, s) in enumerate(legend):
    yy = yb - 0.075 - i * 0.068
    if col:
        fig.lines.append(Line2D([P2.X(0.05), P2.X(0.13)], [P2.Y(yy), P2.Y(yy)], transform=fig.transFigure,
                                color=col, lw=1.3, ls=ls))
    else:
        P2.rect(0.05, yy - 0.022, 0.08, 0.044, fc="#eeeeee", ec=C_DROP, lw=0.5)
    P2.text(0.16, yy, s, va="center")
P2.text(0.5, 0.03, "numbers on tiles: prior rank", ha="center", va="bottom", color=C_TXT, fontsize=5.2)

# ── ③ KV cache ──
P3.frame("③ Stream into the KV cache", C_COV)
sel = sorted(selected)
tw = 0.205
for i, t in enumerate(sel):
    x0, y0, w0, h0 = tile_box(t)
    ax, yb = P3.image(0.05 + i * (tw + 0.025), 0.82, tw, img.crop((x0, y0, x0 + w0, y0 + h0)).resize((448, 300)), )
    spines(ax, C_DIR, 1.2)
    P3.text(0.05 + i * (tw + 0.025) + tw / 2, 0.845, f"D{i + 1}" + ("*" if t in window else ""),
            ha="center", va="bottom", color=C_DIR, fontsize=5.4)
P3.text(0.5, yb - 0.03, "selected detail tiles, near-native resolution\n(* = chosen by boundary scoring)", ha="center",
        va="top", color=C_DIR)
segs = [("$x_b$", C_GREY, 0.7)] + [(f"D{i + 1}", C_DIR, 1) for i in range(len(sel))] + \
       [(f"C{i + 1}", C_COV, 1) for i in range(m)] + [("$x_a$", C_GREY, 0.7)]
unit = 0.92 / sum(s[2] for s in segs)
x, y0 = 0.04, yb - 0.27
for lab, col, wd in segs:
    P3.rect(x, y0, unit * wd - 0.006, 0.10, fc=col, ec="white", lw=0.5)
    P3.text(x + (unit * wd - 0.006) / 2, y0 + 0.05, lab, ha="center", va="center", fontsize=5.4,
            color="black" if col == C_GREY else "white")
    x += unit * wd
P3.text(0.5, y0 - 0.04, f"KV cache: {K} tiles $\\times$ 256 = {K * 256:,} visual tokens\n"
        "(detail tiles in raster order, then coverage tiles)", ha="center", va="top")
P3.text(0.5, y0 - 0.19, f"ViT encodes {m + k - a_ + S} of {N + m} tiles;\nno cost term grows with $N$",
        ha="center", va="top", color=C_TXT)
P3.text(0.5, 0.035, f"Answer: “{answer}”" + (" (correct)" if correct else ""), ha="center", va="bottom",
        fontsize=6.2, fontweight="bold")

for pa, pb in [(P0, P1), (P1, P2), (P2, P3)]:
    yy = P1.Y(0.55)
    arrow(fig, (pa.x + pa.w + 0.001, yy), (pb.x - 0.001, yy))
fig.savefig(os.path.join(HERE, "fig_overview.pdf"))
fig.savefig(os.path.join(HERE, "fig_overview.png"), dpi=400)
plt.close(fig)

# ═════════════════════════════════════════════════════════════════════════════
# Figure 2：boundary scoring
# ═════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(3.3, 2.6))
order = r["prior_ranked"]
vals = [prior[t] for t in order]
colors = [{"direct": C_DIR, "pick": C_WIN, "win": C_WIN, "drop": C_DROP}[status(t)] for t in order]
ax = fig.add_axes([0.12, 0.47, 0.85, 0.40])
bars = ax.bar(range(N), vals, color=colors, width=0.78)
for i, t in enumerate(order):
    if t in window:
        ax.text(i, vals[i] + 0.03, f"{qscore[t]:.3f}", ha="center", va="bottom", rotation=90, fontsize=4.8,
                color=C_PICK if t in selected else C_TXT, fontweight="bold" if t in selected else "normal")
    if t in selected and t in window:
        bars[i].set_edgecolor(C_PICK); bars[i].set_linewidth(0.9)
    if t == target:
        ax.plot(i, -0.19, marker="^", color=C_PICK, ms=3.2, clip_on=False)
ax.set_xlim(-0.6, N - 0.4); ax.set_ylim(0, 1.05)
ax.set_xticks(range(N)); ax.set_xticklabels([str(i + 1) for i in range(N)], fontsize=4.4)
ax.tick_params(axis="x", pad=4.5, length=0); ax.tick_params(axis="y", labelsize=4.8, length=2)
ax.set_xlabel("detail tiles sorted by prior (rank);  ▲ = tile containing the target", fontsize=5.2, labelpad=6)
ax.set_ylabel("prior $\\pi_t$", fontsize=5.4, labelpad=1)
ax.spines[["top", "right"]].set_visible(False)
leg = [(C_DIR, f"kept directly (top $k{{-}}a$ = {k - a_})"),
       (C_WIN, f"boundary window ($S$ = {S}), labeled with $q_t$;\ntop $a$ = {a_} kept (bold)"),
       (C_DROP, f"discarded before the ViT ({N - (k - a_) - S} tiles)")]
legend_y = [0.93, 0.76, 0.59]
for i, (col, s) in enumerate(leg):
    ax.add_patch(Rectangle((7.5, legend_y[i]), 0.7, 0.07, color=col, clip_on=False))
    ax.text(8.5, legend_y[i] + 0.035, s, va="center", fontsize=4.9, linespacing=1.15)
fig.text(0.02, 0.965, f"(a) Ranking and boundary scoring  ($K$={K}, $m$={m}, $k$={k}, $S$={S}, $a$={a_})",
         fontsize=5.8, fontweight="bold", va="top")

fig.text(0.02, 0.30, "(b) Scoring a boundary tile $t$ (Eq. 3)", fontsize=5.8, fontweight="bold", va="top")
blocks = [("$x_b$", 0.10, C_GREY, True), ("coverage $V^c$", 0.28, C_COV, True),
          ("tile $V_t$", 0.20, C_WIN, False), ("question $x_a$", 0.22, C_GREY, False)]
x, y0, hh = 0.08, 0.10, 0.075
xs = []
for lab, wd, col, reused in blocks:
    fig.patches.append(Rectangle((x, y0), wd - 0.008, hh, transform=fig.transFigure, fc=col,
                                 alpha=0.45 if reused else 1.0, hatch="/////" if reused else None,
                                 ec="white" if reused else "black", lw=0.4))
    fig.text(x + (wd - 0.008) / 2, y0 + hh / 2, lab, ha="center", va="center", fontsize=5.2,
             color="white" if (col in (C_COV, C_WIN) and not reused) else "black")
    xs.append((x, x + wd - 0.008)); x += wd
fig.text((xs[0][0] + xs[1][1]) / 2, y0 + hh + 0.015, "prefix KV, layers $1..l_q$:\nreused from the prior pass",
         ha="center", va="bottom", fontsize=4.9, color=C_COV)
fig.text((xs[2][0] + xs[3][1]) / 2, y0 + hh + 0.015, "computed per candidate:\n$n{+}|x_a|$ tokens, $l_q$ layers",
         ha="center", va="bottom", fontsize=4.9)
fig.patches.append(FancyArrowPatch(((xs[3][0] + xs[3][1]) / 2, y0 - 0.005), ((xs[2][0] + xs[2][1]) / 2, y0 - 0.005),
                                   transform=fig.transFigure, connectionstyle="arc3,rad=-0.45", arrowstyle="-|>",
                                   mutation_scale=6, lw=0.8, color=C_PICK))
fig.text(0.60, 0.005, "$q_t$: attention from $x_a$ to $V_t$ at layer $l_q$", ha="center", va="bottom", fontsize=5.0,
         color=C_PICK)
fig.savefig(os.path.join(HERE, "fig_boundary.pdf"))
fig.savefig(os.path.join(HERE, "fig_boundary.png"), dpi=400)
print("saved fig_overview / fig_boundary in", HERE)
