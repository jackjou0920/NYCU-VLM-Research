"""
MME-RealWorld 資料準備：held-out 校準集（完整版扣掉 Lite）+ Lite 測試集。

  校準集 calib600：MME_RealWorld.json（英文版 23,609 題）先扣掉 Lite 的 1,919 題，
    再扣掉「圖片出現在 Lite 裡」的所有題目（校準 / 測試連圖片都不重疊），
    之後每個 Subtask（5 類）用 seed 0 各抽 120 題 → 600 題。
  測試集 lite：MME-RealWorld-Lite 的 1,919 題（data.zip 解壓）。

  圖片只解壓用得到的：校準集只從 tar 分割檔串流抽出那 600 題的圖（不落地整個 88GB），
  Lite 直接解壓 data.zip（約 8.8GB）。

輸出（MME_RW_ROOT，預設 ~/datasets/MME-RealWorld）：
  calib600.json / lite.json      題目清單（Image 欄位是相對於 MME_RW_ROOT 的路徑）
  <AutonomousDriving|ocr_cc|...>/  校準集的圖
  lite/imgs/                      Lite 的圖

用法：
  python prepare_mme_realworld.py                 # 校準集 + Lite
  python prepare_mme_realworld.py --skip_lite     # 只做校準集
"""
from __future__ import annotations

import os
import glob
import json
import random
import argparse
import subprocess
import zipfile
from collections import defaultdict

from huggingface_hub import hf_hub_download

FULL_REPO = "yifanzhang114/MME-RealWorld"
LITE_REPO = "yifanzhang114/MME-RealWorld-Lite"
DEFAULT_ROOT = os.path.expanduser(os.environ.get("MME_RW_ROOT", "~/datasets/MME-RealWorld"))

# JSON 的 Image 第一層資料夾 → tar 檔名（分割檔 <name>.tar.gz.part_*，AutonomousDriving 沒有分割）
ARCHIVES = {
    "AutonomousDriving": "AutonomousDriving",
    "diagram_and_table": "diagram_and_table",
    "monitoring_images": "monitoring_images",
    "ocr_cc": "ocr_cc",
    "remote_sensing": "remote_sensing",
}


def _key(d):
    """Lite 跟完整版的 Question_id / 圖片路徑格式不同，用題目 + 選項 + 圖片檔名對應。"""
    return (d["Text"].strip().lower(), tuple(c.strip().lower() for c in d["Answer choices"]),
            os.path.basename(d["Image"]).lower())


def build_calib(full: list[dict], lite: list[dict], per_subtask: int, seed: int) -> list[dict]:
    lite_keys = {_key(d) for d in lite}
    lite_imgs = {os.path.basename(d["Image"]).lower() for d in lite}
    pool = defaultdict(list)
    n_q, n_img = 0, 0
    for d in full:
        if _key(d) in lite_keys:
            n_q += 1
            continue
        if os.path.basename(d["Image"]).lower() in lite_imgs:
            n_img += 1
            continue
        pool[d["Subtask"]].append(d)
    print(f"完整版 {len(full)} 題：扣掉 Lite 本身 {n_q} 題、圖片跟 Lite 重疊 {n_img} 題 → "
          f"可用 {sum(map(len, pool.values()))} 題")
    rng = random.Random(seed)
    calib = []
    for sub in sorted(pool):
        items = sorted(pool[sub], key=lambda d: d["Question_id"])
        rng.shuffle(items)
        calib += items[:per_subtask]
        print(f"  {sub:28s} 可用 {len(items):5d} 題，抽 {min(per_subtask, len(items))} 題")
    return calib


def extract_from_tars(snapshot: str, root: str, images: list[str]):
    """從 tar 分割檔串流抽出指定的圖（cat parts | tar -xz -T list），已存在的跳過。"""
    by_arch = defaultdict(list)
    for p in sorted(set(images)):
        if not os.path.exists(os.path.join(root, p)):
            by_arch[ARCHIVES[p.split("/")[0]]].append(p)
    for arch, members in by_arch.items():
        parts = sorted(glob.glob(os.path.join(snapshot, f"{arch}.tar.gz.part_*"))) or \
                [os.path.join(snapshot, f"{arch}.tar.gz")]
        lst = os.path.join(root, f".extract_{arch}.txt")
        with open(lst, "w") as f:
            f.write("\n".join(members) + "\n")
        print(f"[extract] {arch}: {len(members)} 張圖（掃過 {len(parts)} 個分割檔）")
        cat = subprocess.Popen(["cat", *parts], stdout=subprocess.PIPE)
        subprocess.run(["tar", "-xz", "-C", root, "-T", lst], stdin=cat.stdout, check=False)
        cat.stdout.close()
        cat.wait()
        os.remove(lst)
    missing = [p for p in images if not os.path.exists(os.path.join(root, p))]
    if missing:
        raise SystemExit(f"有 {len(missing)} 張圖沒抽到，例如 {missing[:3]}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=DEFAULT_ROOT)
    ap.add_argument("--per_subtask", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip_lite", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.root, exist_ok=True)

    full_json = hf_hub_download(FULL_REPO, "MME_RealWorld.json", repo_type="dataset")
    snapshot = os.path.dirname(full_json)
    full = json.load(open(full_json))
    lite_zip = hf_hub_download(LITE_REPO, "data.zip", repo_type="dataset")
    zf = zipfile.ZipFile(lite_zip)
    lite = json.loads(zf.read("data/MME-RealWorld-Lite.json"))

    calib = build_calib(full, lite, args.per_subtask, args.seed)
    extract_from_tars(snapshot, args.root, [d["Image"] for d in calib])
    with open(os.path.join(args.root, "calib600.json"), "w") as f:
        json.dump(calib, f, ensure_ascii=False, indent=1)
    print(f"[calib] {len(calib)} 題 → {args.root}/calib600.json")

    if not args.skip_lite:
        lite_dir = os.path.join(args.root, "lite")
        names = [n for n in zf.namelist() if n.startswith("data/imgs/") and not n.endswith("/")]
        todo = [n for n in names if not os.path.exists(os.path.join(lite_dir, "imgs", os.path.basename(n)))]
        print(f"[lite] 解壓 {len(todo)}/{len(names)} 張圖 → {lite_dir}/imgs")
        os.makedirs(os.path.join(lite_dir, "imgs"), exist_ok=True)
        for n in todo:
            with zf.open(n) as src, open(os.path.join(lite_dir, "imgs", os.path.basename(n)), "wb") as dst:
                while chunk := src.read(1 << 24):
                    dst.write(chunk)
        lite_out = [{**d, "Image": os.path.join("lite", "imgs", d["Image"])} for d in lite]
        with open(os.path.join(args.root, "lite.json"), "w") as f:
            json.dump(lite_out, f, ensure_ascii=False, indent=1)
        print(f"[lite] {len(lite_out)} 題 → {args.root}/lite.json")


if __name__ == "__main__":
    main()
