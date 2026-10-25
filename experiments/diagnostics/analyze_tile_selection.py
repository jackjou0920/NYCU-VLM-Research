"""
比較「greedy oracle 選的 tile」跟「SVMA 實際選的 tile」（不載入模型，純讀檔分析）。

輸入：
  --oracle_json : oracle_tile_diagnostic.py 跑出來的 json（要用新版腳本重跑，才有
                  tile_selection；舊 json 沒存 chosen_tiles）。
  --svma_log    : SVMA 的 run log（results/svma_v1/internvl_svma_docvqa/output_short_docvqa.log 這類）。
                  SVMA 沒把選到的 tile 存進 json，只 print 成
                  `[Image i] tiles=N ...` / `[Image i] selected_tile_indices=[...]`，
                  這裡直接 parse 該 budget 的 --run_stream 區段，依出現順序當作第 0,1,2...張圖。
  --budget      : 要看哪個 budget 的區段（預設 1024）。

回答三個問題（最後一個是關鍵）：
  1) oracle 選的 grid tile 是不是空間上相鄰成群（相對於「隨機選同樣數量」）？
  2) SVMA 跟 oracle 的重疊率（oracle 的 grid tile 有多少被 SVMA 選到）？
  3) 位置偏差：兩者選的 tile 平均落在頁面上方還是下方（row 正規化到 [0,1]）？

注意：thumbnail（最後一個 tile）永遠被保留，分析只看 grid tile（index < num_tiles-1）。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics as st


def parse_svma_log(path: str, budget: int):
    """回傳 [(num_tiles, selected_indices), ...]，依 log 出現順序。"""
    header_re = re.compile(r"^====\s.*\sb(\d+)\s.*--run_stream")
    any_header_re = re.compile(r"^====\s")
    tiles_re = re.compile(r"\[Image \d+\] tiles=(\d+)")
    sel_re = re.compile(r"\[Image \d+\] selected_tile_indices=\[([^\]]*)\]")

    out, in_section, cur_tiles = [], False, None
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            if any_header_re.match(line) and not line.startswith("====="):
                m = header_re.match(line)
                in_section = bool(m and int(m.group(1)) == budget)
                continue
            if not in_section:
                continue
            m = tiles_re.search(line)
            if m:
                cur_tiles = int(m.group(1))
                continue
            m = sel_re.search(line)
            if m and cur_tiles is not None:
                idx = [int(x) for x in m.group(1).split(",") if x.strip()]
                out.append((cur_tiles, idx))
                cur_tiles = None
    return out


def grid_neighbors(a: int, b: int, cols: int) -> bool:
    """8-鄰域相鄰（含對角）。"""
    ra, ca, rb, cb = a // cols, a % cols, b // cols, b % cols
    return a != b and abs(ra - rb) <= 1 and abs(ca - cb) <= 1


def n_components(tiles: list[int], cols: int) -> int:
    tiles = list(tiles)
    seen, comps = set(), 0
    for t in tiles:
        if t in seen:
            continue
        comps += 1
        stack = [t]
        seen.add(t)
        while stack:
            u = stack.pop()
            for v in tiles:
                if v not in seen and grid_neighbors(u, v, cols):
                    seen.add(v)
                    stack.append(v)
    return comps


def mean(x):
    return sum(x) / len(x) if x else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle_json", required=True)
    ap.add_argument("--svma_log", required=True)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--n_random", type=int, default=200, help="每張圖的隨機基準抽樣次數")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    with open(args.oracle_json, encoding="utf-8") as f:
        d = json.load(f)
    sel_dict = d.get("tile_selection") or {}
    if not sel_dict:
        raise SystemExit("oracle json 沒有 tile_selection：請用新版 oracle_tile_diagnostic.py 重跑")
    tag = next(iter(sel_dict))
    oracle = sel_dict[tag]
    svma = parse_svma_log(args.svma_log, args.budget)
    n = min(len(oracle), len(svma))
    print(f"oracle tag={tag}  oracle n={len(oracle)}  svma parsed n={len(svma)}  用前 {n} 張")

    rows = []
    misaligned = 0
    for i in range(n):
        o, (s_tiles, s_sel) = oracle[i], svma[i]
        if o["num_tiles"] != s_tiles:
            misaligned += 1
            continue
        thumb = o["num_tiles"] - 1
        cols, grid_n = o["cols"], max(o["num_tiles"] - 1, 1)
        og = sorted(t for t in o["chosen_tiles"] if t != thumb)
        sg = sorted(t for t in s_sel if t != thumb)
        if not og:
            continue
        rows.append((cols, grid_n, og, sg))
    if misaligned:
        print(f"[WARN] {misaligned}/{n} 張的 tile 數對不上（log 與 oracle 順序可能位移），已跳過；"
              "若比例很高，代表兩邊圖片順序或跳圖不同，結果不可信")
    print(f"可分析張數 = {len(rows)}\n")

    # 1) 空間相鄰（connected components 越少 = 越成群）
    comp_o, comp_s, comp_r, single_o, single_r = [], [], [], 0, 0
    rand_frac_adj = []
    for cols, grid_n, og, sg in rows:
        k = len(og)
        comp_o.append(n_components(og, cols))
        comp_s.append(n_components(sg, cols) if sg else 0)
        single_o += comp_o[-1] == 1
        rc = []
        for _ in range(args.n_random):
            r = rng.sample(range(grid_n), min(k, grid_n))
            rc.append(n_components(r, cols))
        comp_r.append(mean(rc))
        single_r += mean([c == 1 for c in rc])
    print("[1] 空間相鄰（8 鄰域 connected components 數；1 = 全部連成一群）")
    print(f"    oracle  平均 components = {mean(comp_o):.2f}   全部連成一群的比例 = {single_o/len(rows):.1%}")
    print(f"    SVMA    平均 components = {mean(comp_s):.2f}")
    print(f"    隨機    平均 components = {mean(comp_r):.2f}   全部連成一群的比例 = {single_r/len(rows):.1%}")

    # 2) 重疊率
    rec, jac, hit_any = [], [], 0
    rand_rec = []
    for cols, grid_n, og, sg in rows:
        so, ss = set(og), set(sg)
        rec.append(len(so & ss) / len(so))
        jac.append(len(so & ss) / max(len(so | ss), 1))
        hit_any += bool(so & ss)
        rand_rec.append(len(sg) / grid_n if grid_n else 0)   # 隨機選 |sg| 塊時，命中 oracle 單塊的期望率
    print("\n[2] SVMA vs oracle 的 grid tile 重疊")
    print(f"    oracle tile 被 SVMA 選中的比例（recall） = {mean(rec):.1%}")
    print(f"    完全沒重疊的圖比例                       = {1 - hit_any/len(rows):.1%}")
    print(f"    Jaccard                                  = {mean(jac):.3f}")
    print(f"    （隨機選同數量的期望 recall               ≈ {mean(rand_rec):.1%}）")

    # 3) 位置偏差（row 正規化到 [0,1]，0=頁面上方）
    def norm_rows(sel_list, cols, grid_n):
        rws = grid_n // cols
        return [ (t // cols) / max(rws - 1, 1) for t in sel_list ]
    ro, rs = [], []
    for cols, grid_n, og, sg in rows:
        if grid_n // cols <= 1:
            continue    # 只有一列，沒有上下可言
        ro += norm_rows(og, cols, grid_n)
        rs += norm_rows(sg, cols, grid_n)
    print("\n[3] 位置偏差（row 正規化，0=最上、1=最下；只看多列的圖）")
    print(f"    oracle 平均 row = {mean(ro):.3f}   SVMA 平均 row = {mean(rs):.3f}   （均勻分布應為 0.5）")

    # 4) SVMA 只錯在哪裡：oracle 第一個選的 tile（=最像答案所在）有沒有被 SVMA 選到
    print("\n[4] 未在 SVMA 選擇中的 oracle tile，是否緊鄰 SVMA 選的 tile（8 鄰域）")
    miss_total, miss_adj = 0, 0
    for cols, grid_n, og, sg in rows:
        for t in og:
            if t in sg:
                continue
            miss_total += 1
            miss_adj += any(grid_neighbors(t, u, cols) for u in sg)
    if miss_total:
        print(f"    被 SVMA 漏掉的 oracle tile 共 {miss_total} 塊，其中緊鄰 SVMA 已選 tile 的有 {miss_adj/miss_total:.1%}")
        print("    （比例高 → 「選中 tile 後連鄰居一起保留」能撿回一大塊；比例低 → 是位置本身選錯）")


if __name__ == "__main__":
    main()
