"""
prior 的逐 head attention 量測（docs/04_calibration_protocol.md §7c）：只跑 prior 那次 forward，不生成答案。

每張圖、每個 budget：[text_before][m 塊覆蓋 tile][text_after] 跑到第 max(LAYERS) 層，記錄每層每個 head
text_after 對覆蓋 patch 的 attention（三種 query 列）：
  qa  ：text_after 全部列平均（= v1.0 prior 用的列）
  nl  ：text_after 第一個 token（`\n`，在問題之前 → 跟問題無關的基準，不多算）
  gen ：同一張圖配通用問題「Describe this image.」的全部列平均（多一次 forward 的基準）
另存：覆蓋 / 細節網格、目標 tile（與 bbox 重疊的細節 tile）、正式 pipeline 的 v1.0 prior（驗證重建一致）。

輸出：<out>/b<B>.pt（list[dict]，attention 存 fp16，[n_layers, n_heads, m*256]）
用法：python diagnose_prior_heads.py --budgets 1024 1536 2560 [--dataset needle_calib] [--num_images N]
"""
from __future__ import annotations

import os
import json
import math
import argparse

import torch
from PIL import Image

import internvl_stream as S
from internvl_preprocess import build_model, load_image_multiscale
from svma_config import default_coverage_tiles

LAYERS = tuple(range(12, 33))          # 1-indexed；v1.0 prior 用 21..28
GENERIC_Q = "Describe this image."
NEEDLE_ROOT = os.path.expanduser(os.environ.get("NEEDLE_ROOT", "~/datasets/needle_calib"))


def _head_attn_rows(decoder_layer, hidden_states, position_embeddings, q_start: int) -> torch.Tensor:
    """同 internvl_stream._head_mean_attn_rows，但不對 head 平均：回傳 [B, H, nq, L]。"""
    sa = decoder_layer.self_attn
    normed = decoder_layer.input_layernorm(hidden_states)
    B, L, _ = normed.shape
    hd = sa.head_dim
    q = sa.q_norm(sa.q_proj(normed[:, q_start:]).view(B, L - q_start, -1, hd)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(normed).view(B, L, -1, hd)).transpose(1, 2)
    cos, sin = position_embeddings
    q = q * cos[:, q_start:].unsqueeze(1) + S._rotate_half(q) * sin[:, q_start:].unsqueeze(1)
    k = k * cos.unsqueeze(1) + S._rotate_half(k) * sin.unsqueeze(1)
    k = S._repeat_kv(k, sa.num_key_value_groups)
    logits = (q.float() @ k.float().transpose(-2, -1)) * sa.scaling
    q_pos = torch.arange(q_start, L, device=logits.device).view(1, 1, -1, 1)
    k_pos = torch.arange(L, device=logits.device).view(1, 1, 1, -1)
    return logits.masked_fill(k_pos > q_pos, float("-inf")).softmax(dim=-1)


@torch.no_grad()
def per_head_patch_attn(model, h, span_start: int, q_start: int):
    """跑到 max(LAYERS) 層，回傳 (qa, nl)：[n_layers, H, span]（text_after 全部列平均 / 第一列）。"""
    from transformers.masking_utils import create_causal_mask
    lm = model.language_model.model
    B, L, _ = h.shape
    pos_ids = torch.arange(L, device=h.device).unsqueeze(0)
    cache_pos = torch.arange(L, device=h.device)
    mask = create_causal_mask(config=lm.config, input_embeds=h, attention_mask=None, cache_position=cache_pos,
                              past_key_values=None, position_ids=pos_ids)
    pos_emb = lm.rotary_emb(h, pos_ids)
    qa, nl = [], []
    for li, layer in enumerate(lm.layers[:max(LAYERS)], start=1):
        if li in LAYERS:
            a = _head_attn_rows(layer, h, pos_emb, q_start)[0, :, :, span_start:q_start]     # [H, nq, span]
            qa.append(a.mean(dim=1))
            nl.append(a[:, 0])
        if li < max(LAYERS):
            h = layer(h, attention_mask=mask, position_ids=pos_ids, past_key_values=None, use_cache=False,
                      cache_position=cache_pos, position_embeddings=pos_emb)
            h = h[0] if isinstance(h, tuple) else h
    return torch.stack(qa).half().cpu(), torch.stack(nl).half().cpu()


def target_tiles(bboxes, W, H, grid):
    dc, dr = grid
    out = []
    for x, y, w, h in bboxes:
        c0, c1 = int(x / W * dc), min(dc - 1, int((x + w - 1e-6) / W * dc))
        r0, r1 = int(y / H * dr), min(dr - 1, int((y + h - 1e-6) / H * dr))
        out.append(sorted(r * dc + c for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)))
    return out


def load_items(dataset, n):
    if dataset == "needle_calib":
        items = [json.loads(l) for l in open(f"{NEEDLE_ROOT}/questions.jsonl")]
        items = [dict(id=d["id"], path=f"{NEEDLE_ROOT}/{d['image']}", question=d["question"], bboxes=[d["bbox"]],
                      split=d["split"]) for d in items]
    else:
        raise ValueError(dataset)
    return items[:n] if n and n > 0 else items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="needle_calib")
    ap.add_argument("--budgets", type=int, nargs="+", default=[1024, 1536, 2560])
    ap.add_argument("--num_images", type=int, default=-1)
    ap.add_argument("--out", default="results/diagnostics/prior_heads")
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    args = ap.parse_args()
    out_dir = os.path.join(args.out, args.dataset)
    os.makedirs(out_dir, exist_ok=True)
    tok, model = build_model(args.model_name, dtype=torch.bfloat16, device=S.DEVICE)
    adapter = S.InternVLAdapter(tok, model, budget=args.budgets[0])
    emb = adapter.get_input_embeddings(model)
    items = load_items(args.dataset, args.num_images)

    for b in args.budgets:
        m = default_coverage_tiles(b)
        n_tiles = math.ceil(b / model.num_image_token)
        recs = []
        for k, d in enumerate(items):
            img = Image.open(d["path"]).convert("RGB")
            inp = load_image_multiscale(img, input_size=448, max_num_grid=47, coverage_tiles=m)
            cover = torch.cat([S.encode_tile(model, inp["cover"][i:i + S.VIT_BATCH].to(S.DEVICE, dtype=torch.bfloat16),
                                             dtype=torch.bfloat16) for i in range(0, inp["cover"].shape[0], S.VIT_BATCH)])
            rec = {"id": d["id"], "split": d.get("split"), "cover_grid": inp["cover_grid"],
                   "detail_grid": inp["detail_grid"], "m": cover.shape[0],
                   "targets": target_tiles(d["bboxes"], *img.size, inp["detail_grid"])}
            for name, q in (("qa", d["question"]), ("gen", GENERIC_Q)):
                tb, ta = S.split_prompt_at_vision(S.build_prompt(tok, model, q, n_tiles), tok)
                eb = emb(tok([tb], return_tensors="pt").input_ids.to(S.DEVICE))
                ea = emb(tok([ta], return_tensors="pt").input_ids.to(S.DEVICE))
                h = torch.cat([eb, cover.reshape(1, -1, cover.shape[-1]).to(eb.dtype), ea], dim=1)
                qa, nl = per_head_patch_attn(model, h, eb.shape[1], eb.shape[1] + cover.shape[0] * cover.shape[1])
                if name == "qa":
                    rec["qa"], rec["nl"] = qa, nl
                    if k < 20 and inp["detail"].shape[0] > 0:      # 正式 pipeline 的 v1.0 prior，驗證重建
                        pr, _ = adapter.coverage_prior(model, cover, inp["cover_grid"], inp["detail_grid"], tb, ta)
                        rec["v10_prior"] = [pr[i] for i in range(len(pr))]
                else:
                    rec["gen"] = qa
            recs.append(rec)
            if k % 50 == 0:
                print(f"[b{b}] {k + 1}/{len(items)}", flush=True)
        torch.save(recs, os.path.join(out_dir, f"b{b}.pt"))
        print(f"[b{b}] saved {len(recs)} → {out_dir}/b{b}.pt", flush=True)


if __name__ == "__main__":
    main()
