"""
SparseVLM 的 budget 規則補跑：原圖 token 數 <= budget 的圖不剪枝（見 sparsevlm_internvl.budget_lam）。

規則加入前（2026-10-14 之前）跑的 SparseVLM 結果，小圖也被固定 λ 剪過。這支腳本：
  run   ：對原圖 token 數 <= --max_budget 的圖跑一次 λ = 0 的 SparseVLM（同一條程式路徑、不剪枝），
          答案跟 budget 無關，存一份 {index: [raw_tokens, answer]}（可中斷續跑）
  patch ：對每個 budget 的結果檔，把 raw_tokens <= budget 那些題目的答案換成 λ = 0 的答案，
          原檔先備份成 <name>.pre_budget_rule.json，寫回原檔名，之後重新評分

用法：
  python sparsevlm_small_images.py run --dataset docvqa --out results/baselines_v2/internvl_sparsevlm_docvqa/lam0_small.json
  python sparsevlm_small_images.py patch --small results/.../lam0_small.json --jsons results/.../out_sparsevlm_*_b*.json
"""
from __future__ import annotations

import os
import re
import sys
import json
import shutil
import argparse

from internvl_stream import (
    DEVICE, build_model, load_image_tiles, DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
)
from sparsevlm_internvl import generate_sparsevlm


def _load(dataset):
    from process_common import load_docvqa, load_infovqa, load_hrbench
    if dataset == "docvqa":
        return load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                           num_image=None, prompt_mode=DOCVQA_PROMPT)
    if dataset == "infovqa":
        return load_infovqa(dataset="Ahren09/InfoVQA", split="val", num_image=None, prompt_mode=INFOVQA_PROMPT)
    return load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT, num_image=None, prompt_mode=HRBENCH_PROMPT)


def run(args):
    import torch
    out = json.load(open(args.out)) if os.path.exists(args.out) else {"dataset": args.dataset, "answers": {}}
    tok, model = build_model(args.model_name, dtype=torch.bfloat16, device=DEVICE)
    data = _load(args.dataset)
    n_small = 0
    for i, d in enumerate(data):
        pv = load_image_tiles(d["image"], input_size=448, max_num=max(1, args.max_num - 1))
        raw = pv.shape[0] * model.num_image_token
        if raw > args.max_budget:
            continue
        n_small += 1
        if str(i) in out["answers"]:
            continue
        ans, _ = generate_sparsevlm(model, tok, pv, d["question"], budget=raw, lam=0.0)
        out["answers"][str(i)] = [raw, ans]
        if len(out["answers"]) % 20 == 0:
            json.dump(out, open(args.out, "w"), ensure_ascii=False)
            print(f"[{i + 1}/{len(data)}] small images done: {len(out['answers'])}", flush=True)
    json.dump(out, open(args.out, "w"), ensure_ascii=False)
    print(f"done: {n_small} 張圖的原圖 token <= {args.max_budget}（共 {len(data)} 題）→ {args.out}")


def patch(args):
    small = {int(k): v for k, v in json.load(open(args.small))["answers"].items()}
    for path in args.jsons:
        b = int(re.search(r"_b(\d+)\.json$", path).group(1))
        d = json.load(open(path))
        tag = list(d["candidates"])[0]
        cand = d["candidates"][tag]
        if len(cand) != len(d["meta"]):
            sys.exit(f"{path}: candidates {len(cand)} 筆 != meta {len(d['meta'])} 筆，還沒跑完或錯位，先不要 patch")
        bak = path[:-5] + ".pre_budget_rule.json"
        if not os.path.exists(bak):
            shutil.copy(path, bak)
        n = 0
        for i, (raw, ans) in small.items():
            if raw <= b:
                cand[i] = ans
                n += 1
        d["protocol_note"] = ("原圖 token <= budget 的題目改用 λ=0（不剪枝）的答案（sparsevlm_small_images.py，"
                              f"budget 規則見 sparsevlm_internvl.budget_lam）；換掉 {n} 題")
        json.dump(d, open(path, "w"), indent=2, ensure_ascii=False)
        print(f"{path}: b{b} 換掉 {n} 題（備份 {bak}）")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--dataset", required=True, choices=["docvqa", "infovqa", "hrbench"])
    r.add_argument("--out", required=True)
    r.add_argument("--max_budget", type=int, default=5120)
    r.add_argument("--max_num", type=int, default=48)
    r.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    p = sub.add_parser("patch")
    p.add_argument("--small", required=True)
    p.add_argument("--jsons", nargs="+", required=True)
    args = ap.parse_args()
    run(args) if args.cmd == "run" else patch(args)


if __name__ == "__main__":
    main()
