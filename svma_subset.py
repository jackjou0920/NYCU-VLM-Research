"""
固定 seed 的隨機抽樣（只用標準函式庫）。internvl_stream.py 的 --subset_seed 與 reuse_results.py
共用同一個實作，讓「不載入模型」也能算出某次抽樣會抽到哪些題目（用來從全量結果切片沿用）。
"""
from __future__ import annotations

import random


def hrbench_group(d):
    """HR-Bench 每 4 筆是同一題的四種選項輪換（CircularEval 以 id // 4 為一組），抽樣要整組抽。"""
    raw = str(d.get("id", "")).strip()
    return int(raw) // 4 if raw.isdigit() else d.get("question", "")


def random_subset(datasets, n: int, seed: int, complement: bool = False, group_of=None):
    """固定 seed 隨機抽 n 筆（保持原本順序）；complement=True 回傳「沒被抽到」的其餘部分。

    group_of(d) 給定時以「組」為單位抽，n 解讀為組數。同一個 (n, seed) 的抽樣與其 complement
    互斥且合起來是全部，可分別當校準集與測試集。datasets 只需要是 dict 的 list（有 id 即可），
    所以結果 json 的 meta 也能直接丟進來算出同一批題目。"""
    keys, key_of, seen = [], {}, set()
    for i, d in enumerate(datasets):
        g = i if group_of is None else group_of(d)
        if g not in seen:
            seen.add(g)
            keys.append(g)
        key_of[i] = g
    if n > len(keys):
        raise ValueError(f"subset 要抽 {n} 筆，但只有 {len(keys)} 筆（組）")
    picked = set(random.Random(seed).sample(keys, n))
    return [d for i, d in enumerate(datasets) if (key_of[i] in picked) != complement]
