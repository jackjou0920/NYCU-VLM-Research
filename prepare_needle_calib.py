"""
合成大圖 bbox 校準集（prior 命中率用；docs/04_calibration_protocol.md §7c）：
  COCO val2017 的圖填滿裁成 3:2，拼成 4x4 → 2248x1500（≈ V* 中位數 2250x1500）。
  每張大圖挑一個目標物件：該類別在整張大圖只出現這一次（含 iscrowd），bbox 完整在裁切範圍內，
  拼圖後短邊 MIN_SHORT~MAX_SHORT px（V* 有 47% 的目標短邊 < 30 px）。
  題目照 V* direct_attributes 的格式問顏色，選項是隨機顏色——只用來量 prior 命中率，不評答案。
  跟所有測試集不重疊（COCO 圖片；V* 來自 SA-1B，HR-Bench 為自有 8K 圖）。

輸出（NEEDLE_ROOT，預設 ~/datasets/needle_calib）：
  imgs/<i>.jpg、questions.jsonl（image / question / bbox [x, y, w, h] / target / split）
  split：前一半 "fit"（挑 head 等設定）、後一半 "val"（驗證，避免在同一批圖上挑又評）

用法：python prepare_needle_calib.py [--n 300] [--seed 0]
"""
from __future__ import annotations

import os
import json
import random
import argparse
from collections import defaultdict

from PIL import Image

COCO = os.path.expanduser("~/datasets/coco")
ROOT = os.path.expanduser(os.environ.get("NEEDLE_ROOT", "~/datasets/needle_calib"))
CELL_W, CELL_H, COLS, ROWS = 562, 375, 4, 4
MIN_SHORT, MAX_SHORT = 6, 60
COLORS = ["red", "blue", "green", "yellow", "white", "black", "gray", "brown", "orange", "pink", "purple", "silver"]


def crop_box(w, h):
    """填滿 CELL_W:CELL_H 的置中裁切範圍（原圖座標）。"""
    r = CELL_W / CELL_H
    if w / h > r:
        cw, ch = h * r, h
    else:
        cw, ch = w, w / r
    return (w - cw) / 2, (h - ch) / 2, cw, ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    d = json.load(open(f"{COCO}/annotations/instances_val2017.json"))
    cats = {c["id"]: c["name"] for c in d["categories"]}
    imgs = {im["id"]: im for im in d["images"]}
    anns = defaultdict(list)
    for a in d["annotations"]:
        anns[a["image_id"]].append(a)
    ids = sorted(i for i in imgs if anns[i])
    os.makedirs(f"{ROOT}/imgs", exist_ok=True)

    out, tries = [], 0
    while len(out) < args.n and tries < args.n * 200:
        tries += 1
        cell_ids = rng.sample(ids, COLS * ROWS)
        # 每格裁切後還看得到的物件（bbox 中心在裁切範圍內就算看得到）
        visible = []
        for k, iid in enumerate(cell_ids):
            im = imgs[iid]
            x0, y0, cw, ch = crop_box(im["width"], im["height"])
            s = CELL_W / cw
            for a in anns[iid]:
                x, y, w, h = a["bbox"]
                cx, cy = x + w / 2, y + h / 2
                if x0 <= cx <= x0 + cw and y0 <= cy <= y0 + ch:
                    inside = x >= x0 and y >= y0 and x + w <= x0 + cw and y + h <= y0 + ch
                    mx = (k % COLS) * CELL_W + (x - x0) * s
                    my = (k // COLS) * CELL_H + (y - y0) * s
                    visible.append((a["category_id"], a["iscrowd"], inside, [mx, my, w * s, h * s]))
        count = defaultdict(int)
        for c, *_ in visible:
            count[c] += 1
        cand = [(c, b) for c, crowd, inside, b in visible
                if count[c] == 1 and not crowd and inside and MIN_SHORT <= min(b[2], b[3]) <= MAX_SHORT]
        if not cand:
            continue
        c, bbox = rng.choice(cand)
        mosaic = Image.new("RGB", (COLS * CELL_W, ROWS * CELL_H))
        for k, iid in enumerate(cell_ids):
            im = imgs[iid]
            x0, y0, cw, ch = crop_box(im["width"], im["height"])
            tile = Image.open(f"{COCO}/val2017/{im['file_name']}").convert("RGB")
            tile = tile.crop((round(x0), round(y0), round(x0 + cw), round(y0 + ch))).resize((CELL_W, CELL_H))
            mosaic.paste(tile, ((k % COLS) * CELL_W, (k // COLS) * CELL_H))
        i = len(out)
        rel = f"imgs/{i:04d}.jpg"
        mosaic.save(f"{ROOT}/{rel}", quality=95)
        opts = rng.sample(COLORS, 4)
        q = (f"What is the color of the {cats[c]}?\n" + "\n".join(f"({L}) {o}" for L, o in zip("ABCD", opts))
             + "\nAnswer with the option's letter from the given choices directly.")
        out.append({"id": str(i), "image": rel, "question": q, "target": cats[c],
                    "bbox": [round(v, 1) for v in bbox], "cells": cell_ids,
                    "split": "fit" if i < args.n // 2 else "val"})
    with open(f"{ROOT}/questions.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    shorts = sorted(min(r["bbox"][2], r["bbox"][3]) for r in out)
    print(f"{len(out)} 張（嘗試 {tries} 次）→ {ROOT}；目標短邊中位數 {shorts[len(shorts) // 2]:.0f} px，"
          f"< 30 px 佔 {100 * sum(s < 30 for s in shorts) / len(shorts):.0f}%")


if __name__ == "__main__":
    main()
