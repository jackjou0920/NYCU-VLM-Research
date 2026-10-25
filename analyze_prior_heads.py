"""
prior 變體的候選召回率（docs/04_calibration_protocol.md §7c），讀 diagnose_prior_heads.py 的輸出。

候選召回率：每個目標至少一塊重疊的細節 tile 落在 prior 前 C 名（C = slots - a + S，直接保留 + 打分窗口）。
變體：v1.0（21..28 層全部 head）、A1（fit 上單 head 召回率前 k 名）、A2（問題無關基準 nl / gen 的 ratio / sub 校正）。

用法：python analyze_prior_heads.py [--dir results/diagnostics/prior_heads/needle_calib] [--check]
"""
from __future__ import annotations

import os
import argparse
import functools
import statistics as st

import torch

from svma_config import default_coverage_tiles, default_qattn_tiles, split_slots
from diagnose_prior_heads import LAYERS

BUDGETS = (1024, 1536, 2560)
V10_LAYERS = tuple(range(21, 29))
EPS = 1e-4


def n_candidates(b):
    k, m = b // 256, default_coverage_tiles(b)
    slots = k - m
    a, bb = split_slots(slots, default_qattn_tiles(b, m))
    return slots - a + a + bb


@functools.lru_cache(maxsize=None)
def _patch_to_tile(cover_grid, detail_grid, m, side=16):
    cc, cr = cover_grid
    cols, rows = detail_grid
    idx = []
    for t in range(m):
        tr, tc = divmod(t, cc)
        for pr in range(side):
            for pc in range(side):
                gc = min(cols - 1, int((tc * side + pc + 0.5) / (cc * side) * cols))
                gr = min(rows - 1, int((tr * side + pr + 0.5) / (cr * side) * rows))
                idx.append(gr * cols + gc)
    idx = torch.tensor(idx)
    n = cols * rows
    c = torch.zeros(n, dtype=torch.float64).index_add_(0, idx, torch.ones_like(idx, dtype=torch.float64))
    return idx, n, c.clamp(min=1)


def tile_mean(sal: torch.Tensor, r) -> torch.Tensor:
    """[..., m*256] patch 分數 → [..., n_detail] tile 平均（同 coverage_prior：patch 中心落在哪塊細節 tile）。"""
    idx, n, c = _patch_to_tile(tuple(r["cover_grid"]), tuple(r["detail_grid"]), r["m"])
    s = torch.zeros(*sal.shape[:-1], n, dtype=torch.float64).index_add_(-1, idx, sal.double())
    return s / c


def hit(score: torch.Tensor, r, C: int) -> bool:
    """score：[n_detail]；排序同 pipeline（分數高到低，平手 index 小的優先）。"""
    order = sorted(range(score.shape[0]), key=lambda i: (-float(score[i]), i))[:C]
    return all(set(t) & set(order) for t in r["targets"])


def patch_scores(r, heads, query="qa", base=None, mode=None):
    """heads：[(layer, head)] 或 None（= v1.0：21..28 層全部 head）。回傳 [m*256] patch 分數。"""
    def sel(key):
        x = r[key].float()                                     # [n_layers, H, span]
        if heads is None:
            li = [LAYERS.index(l) for l in V10_LAYERS]
            return x[li].mean(dim=(0, 1))
        return torch.stack([x[LAYERS.index(l), h] for l, h in heads]).mean(dim=0)
    q = sel(query)
    if base is None:
        return q
    p, pb = q / q.sum(), sel(base) / sel(base).sum()
    return p / (pb + EPS) if mode == "ratio" else p - pb


def recall(recs, C, **kw):
    hs = [hit(tile_mean(patch_scores(r, **kw), r), r, C) for r in recs if r["detail_grid"][0] > 0]
    return 100 * st.mean(hs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/diagnostics/prior_heads/needle_calib")
    ap.add_argument("--check", action="store_true", help="只驗證 v1.0 重建跟 pipeline 的 prior 排序一致")
    ap.add_argument("--budgets", type=int, nargs="+", default=list(BUDGETS))
    args = ap.parse_args()
    data = {b: torch.load(os.path.join(args.dir, f"b{b}.pt")) for b in args.budgets}

    # v1.0 重建 vs 正式 pipeline 的 prior（排序要一致）
    for b, recs in data.items():
        chk = [r for r in recs if "v10_prior" in r]
        same = sum(
            sorted(range(len(r["v10_prior"])), key=lambda i: (-r["v10_prior"][i], i))
            == sorted(range(len(r["v10_prior"])), key=lambda i: (-float(tile_mean(patch_scores(r, None), r)[i]), i))
            for r in chk)
        print(f"[check] b{b}: v1.0 重建排序與 pipeline 一致 {same}/{len(chk)}")
    if args.check:
        return

    fit = {b: [r for r in recs if r["split"] == "fit"] for b, recs in data.items()}
    val = {b: [r for r in recs if r["split"] == "val"] for b, recs in data.items()}
    C = {b: n_candidates(b) for b in data}
    print("候選數 C：", C)

    # A1：fit 上逐 head 的召回率（3 個 budget 平均）
    H = data[args.budgets[0]][0]["qa"].shape[1]
    single = {(l, h): [] for l in LAYERS for h in range(H)}
    for b in data:
        hits = torch.zeros(len(LAYERS), H)
        rs = [r for r in fit[b] if r["detail_grid"][0] > 0]
        for r in rs:
            T = tile_mean(r["qa"].float(), r)                               # [n_layers, H, n_detail]
            # 排序同 pipeline：分數高到低、平手 index 小的優先 → 用 (-score, index) 的穩定排序
            order = torch.sort(-T, dim=-1, stable=True).indices[..., :C[b]]
            for t in r["targets"]:
                hits += torch.isin(order, torch.tensor(t)).any(dim=-1).float() if t is r["targets"][0] else 0
            if len(r["targets"]) > 1:
                raise NotImplementedError("多目標只在 V* relative 才有")
        for li, l in enumerate(LAYERS):
            for h in range(H):
                single[(l, h)].append(100 * float(hits[li, h]) / len(rs))
    single = {k: st.mean(v) for k, v in single.items()}
    ranked = sorted(single, key=lambda x: -single[x])
    print("fit 上單 head 召回率前 10：", [(l, h, round(single[(l, h)], 1)) for l, h in ranked[:10]])
    v10_fit = st.mean(recall(fit[b], C[b], heads=None) for b in data)
    print(f"fit 上 v1.0 召回率 {v10_fit:.1f}；單 head 比 v1.0 好的有 {sum(v > v10_fit for v in single.values())} 個")

    variants = {"v1.0": dict(heads=None)}
    for base in ("nl", "gen"):
        for mode in ("ratio", "sub"):
            variants[f"A2-{base}-{mode}"] = dict(heads=None, base=base, mode=mode)
    for k in (2, 4, 8, 16, 32):
        variants[f"A1-top{k}"] = dict(heads=ranked[:k])
        for base in ("nl", "gen"):
            for mode in ("ratio", "sub"):
                variants[f"A1-top{k}+A2-{base}-{mode}"] = dict(heads=ranked[:k], base=base, mode=mode)

    print(f"\n{'variant':26s} " + " ".join(f"{'val b' + str(b):>10s}" for b in data) + f"{'val avg':>9s}{'fit avg':>9s}")
    rows = []
    for name, kw in variants.items():
        v = [recall(val[b], C[b], **kw) for b in data]
        f = st.mean(recall(fit[b], C[b], **kw) for b in data)
        rows.append((name, v, st.mean(v), f))
        print(f"{name:26s} " + " ".join(f"{x:10.1f}" for x in v) + f"{st.mean(v):9.1f}{f:9.1f}")
    best = max(rows, key=lambda x: x[2])
    print(f"\nval 平均最高：{best[0]} {best[2]:.1f}（v1.0 {rows[0][2]:.1f}，差 {best[2] - rows[0][2]:+.1f}）")


if __name__ == "__main__":
    main()
