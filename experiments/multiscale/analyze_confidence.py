"""
依信心度切換模式（confidence-gated cascade）的離線分析。

前提：同一批題目用兩種多尺度設定各跑一次（run_multiscale.sh，M_LIST="1 3K/4"），JSON 的
tile_stats[tag][i]["conf"] 有答案信心度（internvl_stream.py 的 decode 迴圈記錄）：
  p_first      第一個答案 token 的機率
  margin_first 第一個 token 的 p_top1 - p_top2
  mean_logp    整段答案每個 token 的平均 log 機率（不含 EOS）
  min_logp     整段答案裡最低的 token log 機率

策略（每題獨立決定）：
  先跑 fast 模式；信心度 >= tau 就直接輸出，否則再跑 slow 模式，
    switch：改用 slow 的答案；pick：取兩次裡信心度較高的答案。
  對照：只跑 fast / 只跑 slow / 兩個都跑再依信心度挑（rerun 100%）/ oracle（每題選對）。

門檻：把題目隨機分成校準 / 測試兩半（HR-Bench 以 4 題一組為單位），在校準半上替每個 budget 選
「所有 dataset 共用的單一 tau」（平均準確率最高、平均重跑比例 <= --max_rerun），再到測試半報告。
平均 TTFT 用 JSON 的 ttft（batch 內共用第一張圖的值，當作每張圖 TTFT 的估計）：
  E[TTFT] ≈ TTFT_fast + rerun_rate × TTFT_slow（第二次沒有重用任何 KV 的保守估計）。

用法：
  python analyze_confidence.py                       # 預設 --prefix out_conf、fast = 3K/4、slow = 1
  python analyze_confidence.py --prefix out_conf --budgets 1024 2048 --max_rerun 0.5
"""
from __future__ import annotations

import os
import re
import sys
import json
import random
import argparse
import statistics as st

from evaluate_anls import calculate_single_anls
from evaluate_hrbench import canon_pred, gold_letter

MODEL_TAG = "InternVL3_5-8B"
SIGNALS = ["p_first", "margin_first", "mean_logp", "min_logp"]


def tag_key(tag):
    """tag → (m, detail_scale)；沒有 _m / _ds 後綴就是 m=1、官方網格。"""
    m = re.search(r"_m(\d+)", tag)
    ds = re.search(r"_ds([\d.]+)$", tag)
    return int(m.group(1)) if m else 1, ds.group(1) if ds else "0"


def m_value(spec, K):
    """'1' / '3K/4' / 'K' → 整數 m。"""
    if "K" not in spec:
        return int(spec)
    num, _, den = spec.partition("K")
    return K * int(num or 1) // int(den.lstrip("/") or 1)


def load_units(path, m_fast, m_slow, hr):
    """回傳 (units, ttft_fast, ttft_slow)。unit = 一題（docs）或一組 4 題（HR-Bench）；
    每個 unit 是 list[member]，member = {"fast": (score, conf), "slow": (score, conf)}。"""
    d = json.load(open(path))
    tags = {tag_key(t): t for t in d["candidates"]}
    tf, ts = tags.get((m_fast, "0")), tags.get((m_slow, "0"))
    if tf is None or ts is None:
        raise KeyError(f"{path} 缺少 m={m_fast} 或 m={m_slow} 的 tag（有：{sorted(tags)}）")
    meta = d["meta"]
    n = min(len(d["candidates"][tf]), len(d["candidates"][ts]), len(meta))

    def member(i):
        out = {}
        for name, t in (("fast", tf), ("slow", ts)):
            pred = str(d["candidates"][t][i])
            if hr:
                pc = canon_pred(pred, meta[i], "letter")
                score = 1.0 if (pc is not None and pc == gold_letter(meta[i])) else 0.0
            else:
                score = calculate_single_anls(pred, meta[i]["answers"])
            conf = d["tile_stats"][t][i].get("conf")
            if conf is None:
                raise KeyError(f"{path} 的 {t} 沒有 conf（要用新版 internvl_stream.py 重跑）")
            out[name] = (score, conf)
        return out

    if hr:   # HR-Bench：id // 4 一組（4 種選項排列），全部答對才算對
        groups = {}
        for i in range(n):
            groups.setdefault(int(meta[i]["id"]) // 4, []).append(i)
        units = [[member(i) for i in g] for g in groups.values() if len(g) == 4]
    else:
        units = [[member(i)] for i in range(n)]
    mean_ttft = lambda t: st.mean(x for x in d.get("ttft", {}).get(t, [float("nan")]) if x is not None)
    return units, mean_ttft(tf), mean_ttft(ts)


def unit_score(unit, hr, choose):
    """choose(member) → 'fast' / 'slow'。HR：全部答對才 1；docs：ANLS。"""
    scores = [mem[choose(mem)][0] for mem in unit]
    return float(all(s >= 1.0 for s in scores)) if hr else scores[0]


def policy(sig, tau, mode):
    """回傳 choose(member) 與 rerun(member)。"""
    def rerun(mem):
        return mem["fast"][1][sig] < tau

    def choose(mem):
        if not rerun(mem):
            return "fast"
        if mode == "switch":
            return "slow"
        return "fast" if mem["fast"][1][sig] >= mem["slow"][1][sig] else "slow"
    return choose, rerun


def evaluate(units, hr, choose, rerun=None):
    acc = 100 * st.mean(unit_score(u, hr, choose) for u in units)
    rr = st.mean(float(rerun(m)) for u in units for m in u) if rerun else 0.0
    return acc, rr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prefix", default="out_conf")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1024, 2048])
    ap.add_argument("--fast", default="3K/4", help="先跑的模式（m，可寫 K 的比例）")
    ap.add_argument("--slow", default="1", help="信心度低時改跑的模式")
    ap.add_argument("--num_images", type=int, default=500, help="docvqa / infovqa 只取前 N 題（跟其他結果對齊）")
    ap.add_argument("--max_rerun", type=float, default=0.5, help="選門檻時平均重跑比例的上限")
    ap.add_argument("--calib_frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    files = {
        "docvqa": f"{args.prefix}_short_{MODEL_TAG}_docvqa_b{{b}}.json",
        "infovqa": f"{args.prefix}_short_{MODEL_TAG}_infovqa_b{{b}}.json",
        "hrbench4k": f"{args.prefix}_4k_letter_{MODEL_TAG}_hrbench_b{{b}}.json",
        "hrbench8k": f"{args.prefix}_8k_letter_{MODEL_TAG}_hrbench_b{{b}}.json",
    }
    for b in args.budgets:
        K = b // 256
        mf, ms = m_value(args.fast, K), m_value(args.slow, K)
        data = {}
        for name, pat in files.items():
            path = pat.format(b=b)
            if not os.path.exists(path):
                print(f"[skip] {path} 不存在")
                continue
            units, t_f, t_s = load_units(path, mf, ms, hr=name.startswith("hrbench"))
            if not name.startswith("hrbench"):
                units = units[:args.num_images]
            data[name] = (units, t_f, t_s)
        if not data:
            continue
        hr_of = lambda name: name.startswith("hrbench")

        print(f"\n{'=' * 100}\n budget={b} (K={K})  fast = m{mf}  slow = m{ms}\n{'=' * 100}")
        # ---- 1. 對照 + 信號品質（兩個都跑、依信心度挑）----
        print(f"\n{'dataset':10s} {'fast':>6s} {'slow':>6s} {'oracle':>7s} | both-run, pick higher conf: "
              + "  ".join(f"{s:>12s}" for s in SIGNALS) + f" | TTFT fast/slow")
        for name, (units, t_f, t_s) in data.items():
            hr = hr_of(name)
            fast = evaluate(units, hr, lambda m: "fast")[0]
            slow = evaluate(units, hr, lambda m: "slow")[0]
            orac = 100 * st.mean(
                float(all(max(m["fast"][0], m["slow"][0]) >= 1.0 for m in u)) if hr
                else max(u[0]["fast"][0], u[0]["slow"][0]) for u in units)
            picks = [evaluate(units, hr, policy(s, float("inf"), "pick")[0])[0] for s in SIGNALS]
            print(f"{name:10s} {fast:6.1f} {slow:6.1f} {orac:7.1f} | {'':27s}"
                  + "  ".join(f"{p:12.1f}" for p in picks) + f" | {t_f:.2f}/{t_s:.2f}s")

        # ---- 2. 依信心度的 cascade：校準半選共用門檻、測試半報告 ----
        rng = random.Random(args.seed)
        split = {}
        for name, (units, _, _) in data.items():
            idx = list(range(len(units)))
            rng.shuffle(idx)
            c = int(len(idx) * args.calib_frac)
            split[name] = ([units[i] for i in idx[:c]], [units[i] for i in idx[c:]])
        print(f"\n共用門檻（校準 {args.calib_frac:.0%}、平均重跑 <= {args.max_rerun:.0%}），測試半的結果：")
        print(f"{'signal':13s} {'mode':6s} {'tau':>9s} | " + "  ".join(f"{n:>22s}" for n in data)
              + f" | {'mean Δ vs fast':>14s}")
        for sig in SIGNALS:
            allv = sorted({m["fast"][1][sig] for cal, _ in split.values() for u in cal for m in u})
            taus = [allv[int(q * (len(allv) - 1))] for q in [i / 40 for i in range(41)]] + [float("inf")]
            for mode in ("switch", "pick"):
                best = None
                for tau in taus:
                    ch, rr = policy(sig, tau, mode)
                    res = [evaluate(cal, hr_of(n), ch, rr) for n, (cal, _) in split.items()]
                    acc, rate = st.mean(r[0] for r in res), st.mean(r[1] for r in res)
                    if rate <= args.max_rerun and (best is None or acc > best[0]):
                        best = (acc, tau)
                tau = best[1]
                ch, rr = policy(sig, tau, mode)
                cells, deltas = [], []
                for name, (_, test) in split.items():
                    hr = hr_of(name)
                    acc, rate = evaluate(test, hr, ch, rr)
                    base = evaluate(test, hr, lambda m: "fast")[0]
                    _, t_f, t_s = data[name]
                    deltas.append(acc - base)
                    cells.append(f"{acc:5.1f} ({acc - base:+4.1f}) r{rate:3.0%} {t_f + rate * t_s:4.2f}s")
                print(f"{sig:13s} {mode:6s} {tau:9.4f} | " + "  ".join(f"{c:>22s}" for c in cells)
                      + f" | {st.mean(deltas):+14.2f}")
        print("  格式：測試半準確率（相對只跑 fast 的差）重跑比例 估計平均 TTFT")


if __name__ == "__main__":
    sys.exit(main())
