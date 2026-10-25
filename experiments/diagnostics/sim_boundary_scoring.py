"""
邊界打分（boundary scoring）模擬：只用 diag_cascade_stages.py 的分數（每張圖所有 tile 都有 q_attn 分數），
不跑模型。比較高 budget（K 大）時，各種「只對部分 tile 打分」的選法：

  full      : 所有 grid tile 都跑 q_attn，取前 K-1（打分完全不省的參考答案）
  casc m    : 現行 cascade：prior 前 M = ceil(m*(K-1)) 塊跑 ViT + q_attn，取前 K-1
  bnd a,b   : 邊界打分：prior 排名前 (K-1-a) 塊直接保留（只跑 ViT、不打分）；
              接下來 a+b 塊跑 ViT + q_attn，從中取 a 塊；其餘直接丟（不跑 ViT）
              → q_attn 只跑 a+b 塊，不隨 K 增加

指標（每張圖平均）：
  與 full 重疊 : |選到的 ∩ full 選到的| / (K-1)
  留住最佳     : oracle 最佳 tile（per-tile oracle 分數最高者）有沒有被選進來
                 （oracle 是在 b1024 的設定下算的「單塊最有用的 tile」，在大 K 只能當參考）
  第二階段成本 : 跑 ViT 的塊數 × VIT_MS + 打分的塊數 × 打分 ms/tile

用法：
    python sim_boundary_scoring.py --diag_json out_diag_cascade_v2_infovqa_b1024.json --sig qattn_ctx_kv_L20
"""
from __future__ import annotations

import argparse
import json
import math

VIT_MS = 12.0     # 每塊 tile 的 ViT + 其他開銷（由 TTFT 對候選數的斜率推估）


def rank_desc(d):
    return sorted(d, key=lambda t: (-d[t], t))


def select(policy, prior, score, slots):
    """回傳 (選到的 grid tile, 跑 ViT 的塊數, 打分的塊數)。"""
    order = rank_desc(prior)
    n = len(order)
    kind = policy[0]
    if n <= slots:                                         # tile 不夠多，全部保留，不必打分
        return set(order), n, 0
    if kind == "full":
        return set(rank_desc(score)[:slots]), n, n
    if kind == "casc":
        M = min(n, max(slots, math.ceil(policy[1] * slots)))
        if M <= slots:
            return set(order[:M]), M, 0
        return set(rank_desc({t: score[t] for t in order[:M]})[:slots]), M, M
    if kind == "bnd":
        a, b = policy[1], policy[2]
        a = min(a, slots)
        keep = order[: slots - a]
        window = order[slots - a: slots + b]
        pick = rank_desc({t: score[t] for t in window})[:a]
        return set(keep) | set(pick), slots + b if slots + b < n else n, len(window)
    raise ValueError(policy)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag_json", default="out_diag_cascade_v2_infovqa_b1024.json")
    ap.add_argument("--sig", default="qattn_ctx_kv_L20", help="第二階段用哪個打分訊號")
    ap.add_argument("--prior", default="prior_L21-28q")
    ap.add_argument("--budgets", default="1024,2048,3072")
    args = ap.parse_args()

    out = json.load(open(args.diag_json, encoding="utf-8"))
    recs = [r for r in out["records"] if "skip" not in r]
    ik = lambda d: {int(k): v for k, v in d.items()}
    for r in recs:
        r["oracle"] = ik(r["oracle"])
        r["prior"] = ik(r["priors"][args.prior])
        r["score"] = ik(r["scores"][args.sig])
    ms_list = [r.get("sig_ms_per_tile", {}).get(args.sig) for r in recs[1:]]
    ms_list = [x for x in ms_list if x is not None]
    score_ms = sum(ms_list) / len(ms_list) if ms_list else 29.4

    print(f"{len(recs)} 張圖；stage-1 = {args.prior}，stage-2 = {args.sig}（{score_ms:.1f} ms/tile），"
          f"ViT 約 {VIT_MS:.0f} ms/tile")
    for B in [int(x) for x in args.budgets.split(",")]:
        slots = B // 256 - 1
        policies = [("full",)] + [("casc", m) for m in (1.0, 1.25, 1.5, 2, 3)]
        policies += [("bnd", a, b) for a, b in [(1, 1), (2, 2), (2, 4), (3, 3), (4, 4), (3, 6), (6, 6)]
                     if a <= slots]
        rows = []
        for pol in policies:
            ov, best_kept, vit, sco = [], [], [], []
            for r in recs:
                full_sel, _, _ = select(("full",), r["prior"], r["score"], slots)
                sel, nv, ns = select(pol, r["prior"], r["score"], slots)
                ov.append(len(sel & full_sel) / min(slots, len(full_sel)) if full_sel else 1.0)
                best = max(r["oracle"], key=lambda t: r["oracle"][t])
                best_kept.append(best in sel)
                vit.append(nv)
                sco.append(ns)
            f = lambda a: sum(a) / len(a)
            name = {"full": "full（全部打分）", "casc": f"casc m={pol[1]:g}" if pol[0] == "casc" else "",
                    "bnd": f"bnd a={pol[1]},b={pol[2]}" if pol[0] == "bnd" else ""}[pol[0]]
            cost = f(vit) * VIT_MS + f(sco) * score_ms
            rows.append((name, f(vit), f(sco), cost, 100 * f(ov), 100 * f(best_kept)))
        print(f"\n== b{B}（K-1 = {slots}）")
        print(f"  {'選法':18s} {'ViT塊數':>7s} {'打分塊數':>8s} {'成本ms':>7s} {'與full重疊':>10s} {'留住最佳':>8s}")
        for name, nv, ns, cost, ov, bk in rows:
            print(f"  {name:18s} {nv:7.1f} {ns:8.1f} {cost:7.0f} {ov:9.1f}% {bk:7.1f}%")


if __name__ == "__main__":
    main()
