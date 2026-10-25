"""
從全量結果 json 沿用（不重跑）某個 S 的結果到目標 json（全量或隨機抽樣的校準集都可以）。

可沿用的來源（都要是全量、每題都有）：
  1. 全量 json 裡已經有同一個新 tag：budget=<B>_svma_S<S>[_qa<qa>]
  2. 舊 cascade 的等價設定：S//2 >= K-1 且 S 是 (K-1) 的倍數時，邊界打分 = 舊 cascade m = S/(K-1)，
     tag budget=<B>_info_density_delay0_tp21-28q_casc<m>_qa<qa>（數學上等價，選到的 tile 相同）

目標是抽樣集時（--subset_seed），用 svma_subset.random_subset 在全量 meta 上算出同一批題目，
依題目順序切片；目標 json 不存在就建立（含 meta 與 subset 資訊，internvl_stream.py 之後可以接著跑）。

用法（run_svma_v2.sh 會自動呼叫）：
    python reuse_results.py --full_json out_v2_..._b1024.json --target_json out_v2_..._b1024.json \
        --budget 1024 --S 12 --qa 20ctx
    python reuse_results.py --full_json out_v2_..._b1024.json --target_json out_calib_..._b1024.json \
        --budget 1024 --S 12 --qa 20ctx --subset_seed 0 --subset_n 600 --dataset infovqa
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from svma_subset import random_subset, hrbench_group


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full_json", required=True)
    ap.add_argument("--target_json", required=True)
    ap.add_argument("--budget", type=int, required=True)
    ap.add_argument("--S", type=int, required=True)
    ap.add_argument("--qa", default="20ctx")
    ap.add_argument("--subset_seed", type=int, default=None)
    ap.add_argument("--subset_n", type=int, default=None)
    ap.add_argument("--subset_complement", action="store_true")
    ap.add_argument("--dataset", default="infovqa")
    args = ap.parse_args()

    B, S = args.budget, args.S
    new_tag = f"budget={B}_svma_S{S}" + (f"_qa{args.qa}" if S > 0 else "")
    if not os.path.exists(args.full_json):
        return
    full = json.load(open(args.full_json, encoding="utf-8"))
    meta = full.get("meta", [])
    n = len(meta)
    cands = full.get("candidates", {})

    slots = B // 256 - 1
    sources = [new_tag]
    if S > 0 and slots > 0 and S // 2 >= slots and S % slots == 0:
        sources.append(f"budget={B}_info_density_delay0_tp21-28q_casc{S // slots}_qa{args.qa}")
    src = next((t for t in sources if n and len(cands.get(t, [])) == n), None)
    if src is None:
        return

    same_file = os.path.abspath(args.full_json) == os.path.abspath(args.target_json)
    if args.subset_seed is None:
        if not same_file:
            sys.exit("[reuse] 沒有 --subset_seed 時，target_json 必須就是 full_json")
        if src == new_tag:
            return
        for k in ("candidates", "ttft", "tile_selection"):
            if src in full.get(k, {}):
                full[k][new_tag] = full[k][src]
        json.dump(full, open(args.target_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        print(f"[reuse] {src} -> {new_tag}（S={S} 與舊 cascade 數學上等價）")
        return

    # 抽樣集：在全量 meta 上算出同一批題目（題目順序 = 資料集原順序，與 internvl_stream.py 相同）
    group_of = hrbench_group if args.dataset == "hrbench" else None
    idx_rows = random_subset([dict(m, _i=i) for i, m in enumerate(meta)], args.subset_n,
                             args.subset_seed, complement=args.subset_complement, group_of=group_of)
    idx = [r["_i"] for r in idx_rows]
    ids = [str(meta[i].get("id", i)) for i in idx]

    if os.path.exists(args.target_json):
        tgt = json.load(open(args.target_json, encoding="utf-8"))
        if (tgt.get("subset") or {}).get("ids") != ids:
            sys.exit(f"[reuse] {args.target_json} 的題目集合跟這次的抽樣不同，不沿用")
    else:
        tgt = {"references": [], "candidates": {}, "meta": [meta[i] for i in idx],
               "subset": {"seed": args.subset_seed, "n": args.subset_n,
                          "complement": bool(args.subset_complement), "ids": ids}}
    if len(tgt.get("candidates", {}).get(new_tag, [])) == len(idx):
        return
    for k in ("candidates", "ttft", "tile_selection"):
        if src in full.get(k, {}) and len(full[k][src]) == n:
            tgt.setdefault(k, {})[new_tag] = [full[k][src][i] for i in idx]
    json.dump(tgt, open(args.target_json, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"[reuse] {args.full_json} :: {src} 依抽樣切出 {len(idx)} 筆 -> {args.target_json} :: {new_tag}")


if __name__ == "__main__":
    main()
