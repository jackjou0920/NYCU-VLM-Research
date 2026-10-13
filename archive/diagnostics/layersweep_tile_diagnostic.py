"""
淺層打分能不能保住 accuracy？——「ANLS vs 層數」掃描（診斷用，不動任何既有檔案）

背景：selfscore_tile_diagnostic.py 證實「每塊 tile 各跑一次 LLM、看作答信心」在 InfoVQA
b1024 能比 SVMA 高 ~9 分，但每塊 tile 跑完整 36 層，TTFT +2.5~2.9s，不實際。這裡測：
只跑前 L 層能不能保住那個增益，以及哪一種淺層訊號最有效。

每塊 grid tile 獨立打分（相容 SVMA 的串流：tile 到貨就能算，不看其他 tile），context：
    [text_before] + [這塊 tile 256 token] + [text_after(問題 + assistant 開頭)]
（預設「不放 thumbnail」，每候選 token 從 ~600 降到 ~340；--with_thumb 可加回，跟
selfscore_tile_diagnostic.py 同 context，用來分離「拿掉 thumbnail」與「變淺」各自的影響。）

診斷時每個候選跑完整 36 層一次，在每個 probe 層 L（= 只算前 L 層就拿得到的訊號）同時記錄：
    last_attn : 最後位置（答案第一個 token 的位置）對這塊 tile 256 token 的 attention 總和
                （第 L 層、head 平均）
    q_attn    : text_after 所有位置（問題 + 作答指示 + assistant 開頭）對 tile 的 attention
                總和，再對這些位置平均
    lens_top1 : 第 L 層輸出的最後位置 hidden state 接 final norm + lm_head（logit lens）
                的 top-1 機率；L=36 就是「不含 thumbnail 的完整 self score（top1_prob）」
attention 只手算需要的那幾列（同 fastv_internvl.py 的 _layer_forward_capture_last_row），
真正的 layer 輸出仍走原本的 attention kernel，所以 flash-attn 相容、不會材料化 [H,L,L]。

每個 (訊號, L) 各自選 top-(K-1) + thumbnail，正常 generate（選到同一組 tile 的共用答案），
輸出一份 json，evaluate_anls.py 一次印出全部 tag 的 ANLS。

成本：log 會印每張圖「打分跑到第 L 層」的實測累積秒數（含 probe 本身的額外計算，略高估），
json 裡也存一份；--report_only 印出各 L 的平均秒數。這就是把這個 L 接進 SVMA 時 TTFT
大約要多付的時間（不含 ViT，ViT 本來就會跑）。

用法：
    python layersweep_tile_diagnostic.py --use_ds --dataset infovqa --num_images 200 --budget 1024 \
        --reuse_from out_oracle_infovqa_b1024.json --save --output_json out_layersweep_infovqa_b1024.json

    python evaluate_anls.py --input_json out_layersweep_infovqa_b1024.json \
        --baseline_tag layersweep_lens_top1_L36_budget=1024

    python layersweep_tile_diagnostic.py --report_only --output_json out_layersweep_infovqa_b1024.json
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch

SIGNALS = ["last_attn", "q_attn", "lens_top1"]
DEFAULT_LAYERS = "4,8,12,16,20,24,28,36"


def tag_of(signal: str, layer: int, budget: int) -> str:
    return f"layersweep_{signal}_L{layer}_budget={budget}"


# ──────────────────────────────────────────────────────────────────────────
# probe：只手算需要的 attention 列（不影響 layer 本身的輸出）
# ──────────────────────────────────────────────────────────────────────────
def probe_tile_attention(decoder_layer, hidden_states, position_embeddings, q_start: int,
                         tile_start: int, tile_end: int):
    """回傳 [B, nq]：text_after 每個位置（q_start..L-1）對 tile 區段的 attention 總和（head 平均）。

    hidden_states 是這一層的『輸入』，跟 Qwen3DecoderLayer.forward 一樣先 input_layernorm。
    RoPE 分別套在 query 列與全部 key（apply_rotary_pos_emb 要求 q/k 同長度，所以這裡自己套）。
    """
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv, rotate_half

    sa = decoder_layer.self_attn
    normed = decoder_layer.input_layernorm(hidden_states)
    B, L, _ = normed.shape
    hd = sa.head_dim
    q = sa.q_norm(sa.q_proj(normed[:, q_start:]).view(B, L - q_start, -1, hd)).transpose(1, 2)  # [B,H,nq,d]
    k = sa.k_norm(sa.k_proj(normed).view(B, L, -1, hd)).transpose(1, 2)                           # [B,Hkv,L,d]

    cos, sin = position_embeddings                                     # [B, L, d]
    cq, sq = cos[:, q_start:].unsqueeze(1), sin[:, q_start:].unsqueeze(1)
    ck, sk = cos.unsqueeze(1), sin.unsqueeze(1)
    q = q * cq + rotate_half(q) * sq
    k = k * ck + rotate_half(k) * sk
    k = repeat_kv(k, sa.num_key_value_groups)                          # [B,H,L,d]

    logits = (q.float() @ k.float().transpose(-2, -1)) * sa.scaling    # [B,H,nq,L]
    q_pos = torch.arange(q_start, L, device=logits.device).view(1, 1, -1, 1)
    k_pos = torch.arange(L, device=logits.device).view(1, 1, 1, -1)
    logits = logits.masked_fill(k_pos > q_pos, float("-inf"))          # causal
    attn = logits.softmax(dim=-1)
    mass = attn[..., tile_start:tile_end].sum(dim=-1).mean(dim=1)       # [B, nq]
    return mass


@torch.no_grad()
def score_candidates(model, inputs_embeds, tile_start: int, tile_end: int, q_start: int,
                     probe_layers: list[int], signals: list[str], device):
    """inputs_embeds: [B, L, D]。跑完 max(probe_layers) 層（之後的層不需要），回傳
    ({(signal, L): [B] 分數}, {L: 從開始到算完第 L 層的累積秒數})。"""
    from transformers.masking_utils import create_causal_mask

    lm = model.language_model
    lm_model = lm.model
    B, L, _ = inputs_embeds.shape
    position_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(L, device=device)
    h = inputs_embeds
    causal_mask = create_causal_mask(config=lm_model.config, input_embeds=h, attention_mask=None,
                                     cache_position=cache_position, past_key_values=None,
                                     position_ids=position_ids)
    pos_emb = lm_model.rotary_emb(h, position_ids)
    want = set(probe_layers)
    last = max(probe_layers)

    out, times = {}, {}
    torch.cuda.synchronize(device)
    t0 = time.time()
    for idx, layer in enumerate(lm_model.layers[:last]):
        Lnum = idx + 1                                    # 算完這層 = 已跑 Lnum 層
        if Lnum in want and ("last_attn" in signals or "q_attn" in signals):
            mass = probe_tile_attention(layer, h, pos_emb, q_start, tile_start, tile_end)
            if "last_attn" in signals:
                out[("last_attn", Lnum)] = mass[:, -1].float()
            if "q_attn" in signals:
                out[("q_attn", Lnum)] = mass.mean(dim=-1).float()
            del mass
        h = layer(h, attention_mask=causal_mask, position_ids=position_ids, past_key_values=None,
                  use_cache=False, cache_position=cache_position, position_embeddings=pos_emb)
        if Lnum in want:
            if "lens_top1" in signals:
                logits = lm.lm_head(lm_model.norm(h[:, -1:, :]))[:, -1, :].float()
                out[("lens_top1", Lnum)] = logits.softmax(dim=-1).max(dim=-1).values
                del logits
            torch.cuda.synchronize(device)
            times[Lnum] = time.time() - t0
    del h
    return out, times


# ──────────────────────────────────────────────────────────────────────────
# 報表（不需要模型）
# ──────────────────────────────────────────────────────────────────────────
def report(output_json: str):
    with open(output_json, encoding="utf-8") as f:
        d = json.load(f)
    tl = d.get("scoring_time", [])
    if not tl:
        print("沒有 scoring_time")
        return
    layers = sorted({int(k) for x in tl for k in x["cum_s"]})
    n_img = [x for x in tl if x["n_cand"] > 0]
    tot_c = sum(x["n_cand"] for x in n_img)
    print(f"\n打分成本（{len(n_img)} 張有候選 tile 的圖，平均 {tot_c/len(n_img):.1f} 塊候選；"
          f"with_thumb={d.get('config', {}).get('with_thumb')}）")
    print(f"{'L':>4s} {'平均秒/張':>10s} {'每候選 ms':>10s}")
    for Lnum in layers:
        s = sum(x["cum_s"][str(Lnum)] for x in n_img)
        print(f"{Lnum:4d} {s/len(n_img):10.3f} {1000*s/tot_c:10.1f}")
    pk = [x["peak_gb"] for x in n_img]
    print(f"打分階段 peak 增量（相對打分前常駐量）：平均 {sum(pk)/len(pk):.2f} GB、最大 {max(pk):.2f} GB")
    print("（參考 InfoVQA b1024：SVMA TTFT ~0.75s、IF-Prune ~1.01s、uncompressed ~1.58s；"
          "SVMA peak 只比模型載入高 ~0.3GB）")


# ──────────────────────────────────────────────────────────────────────────
# 主流程
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--max_num", type=int, default=48)
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--layers", type=str, default=DEFAULT_LAYERS, help="probe 層（1-indexed，=只跑前 L 層）")
    ap.add_argument("--signals", type=str, default=",".join(SIGNALS), help=f"逗號分隔，可選 {SIGNALS}")
    ap.add_argument("--with_thumb", action="store_true",
                    help="打分 context 加回 thumbnail（跟 selfscore_tile_diagnostic.py 相同 context），預設不加")
    ap.add_argument("--score_batch_size", type=int, default=8)
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--use_ds", action="store_true")
    ap.add_argument("--dataset", type=str, default="infovqa", choices=["docvqa", "infovqa"])
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--reuse_from", type=str, default=None,
                    help="沿用既有 json 的 references / meta（要用有 references 的那份，例如 out_oracle_infovqa_b1024.json）")
    ap.add_argument("--skip_generate", action="store_true", help="只打分與量時間，不生成答案")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output_json", type=str, default="out_layersweep.json")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        report(args.output_json)
        return

    probe_layers = sorted({int(x) for x in args.layers.split(",") if x.strip()})
    signals = [x.strip() for x in args.signals.split(",") if x.strip()]
    if any(s not in SIGNALS for s in signals):
        raise SystemExit(f"未知 signal，可選 {SIGNALS}")

    from internvl_stream import (
        build_prompt, split_prompt_at_vision, encode_tile, build_model, load_image_tiles,
        DEVICE, DOCVQA_PROMPT, INFOVQA_PROMPT, _infer_tile_grid,
    )
    from sparsevlm_internvl import _get_eos_token_id
    from process_common import load_docvqa, load_infovqa

    dtype = torch.bfloat16
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    n_layers = len(model.language_model.model.layers)
    if max(probe_layers) > n_layers or min(probe_layers) < 1:
        raise SystemExit(f"--layers 要在 1..{n_layers}")
    num_image_token = model.num_image_token
    K = max(1, args.budget // num_image_token)
    eos_id = _get_eos_token_id(tokenizer, model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    embed_fn = model.language_model.get_input_embeddings()

    if args.dataset == "docvqa":
        datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
    else:
        datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
    print(f"Found {len(datasets)} image(s); K={K}（含 thumbnail）; layers={probe_layers}; "
          f"signals={signals}; with_thumb={args.with_thumb}")

    combos = [(s, Lnum) for s in signals for Lnum in probe_layers]
    tags = [tag_of(s, Lnum, args.budget) for s, Lnum in combos]

    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, encoding="utf-8") as f:
            out = json.load(f)
        for key in ("references", "scoring_time"):
            out.setdefault(key, [])
        for key in ("candidates", "tile_selection"):
            out.setdefault(key, {})
    else:
        out = {"references": [], "candidates": {}, "tile_selection": {}, "scoring_time": []}
        if args.reuse_from:
            with open(args.reuse_from, encoding="utf-8") as f:
                src = json.load(f)
            out["references"] = src.get("references", [])[: len(datasets)]
            if "meta" in src:
                out["meta"] = src["meta"][: len(datasets)]
    out["config"] = {"layers": probe_layers, "signals": signals, "with_thumb": args.with_thumb,
                     "budget": args.budget, "dataset": args.dataset}
    for t in tags:
        out["candidates"].setdefault(t, [])
        out["tile_selection"].setdefault(t, [])
    if "meta" not in out:
        out["meta"] = [{"answer": d.get("answer", ""), "answers": d.get("answers", []),
                        "question_type": d.get("question_type", "docvqa"), "id": d.get("id", str(k))}
                       for k, d in enumerate(datasets)]
    start = len(out["scoring_time"])
    if start:
        print(f"Resuming: {start}/{len(datasets)} images already done")

    t_all = time.time()
    for i, d in enumerate(datasets):
        if i < start:
            continue
        pixel_values = load_image_tiles(d["image"], input_size=448, max_num=args.max_num)
        num_tiles = pixel_values.shape[0]
        thumb = num_tiles - 1                            # 同 internvl_stream.py：縮圖是最後一個 tile
        cols, rows = _infer_tile_grid(max(num_tiles - 1, 1), d["image"].size[0], d["image"].size[1])
        print(f"\n[{i+1}/{len(datasets)}] tiles={num_tiles} (grid {cols}x{rows})")

        prompt = build_prompt(tokenizer, model, d["question"], num_tiles)
        text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
        eb = embed_fn(tokenizer(text_before, return_tensors="pt").input_ids.to(DEVICE))
        ea = embed_fn(tokenizer(text_after, return_tensors="pt").input_ids.to(DEVICE))

        with torch.no_grad():
            blocks = []
            for s in range(0, num_tiles, args.vit_batch):
                chunk = pixel_values[s:s + args.vit_batch].to(DEVICE, dtype=dtype)
                tt = encode_tile(model, chunk, dtype=dtype)
                blocks += [tt[j] for j in range(tt.shape[0])]
                del chunk, tt

            cand = [t for t in range(num_tiles) if t != thumb]
            scores = {c: {} for c in combos}
            cum = {Lnum: 0.0 for Lnum in probe_layers}
            torch.cuda.synchronize(DEVICE)
            mem_base = torch.cuda.memory_allocated(DEVICE)
            torch.cuda.reset_peak_memory_stats(DEVICE)

            tile_start = eb.shape[1]
            tile_end = tile_start + num_image_token
            for s in range(0, len(cand), args.score_batch_size):
                bt = cand[s:s + args.score_batch_size]
                B = len(bt)
                if args.with_thumb:
                    vis = torch.stack([torch.cat([blocks[t], blocks[thumb]], 0) for t in bt], 0)
                else:
                    vis = torch.stack([blocks[t] for t in bt], 0)
                inp = torch.cat([eb.expand(B, -1, -1), vis, ea.expand(B, -1, -1)], dim=1)
                q_start = eb.shape[1] + vis.shape[1]
                vals, times = score_candidates(model, inp, tile_start, tile_end, q_start,
                                               probe_layers, signals, DEVICE)
                for c in combos:
                    for t, v in zip(bt, vals[c].tolist()):
                        scores[c][t] = v
                for Lnum, sec in times.items():
                    cum[Lnum] += sec
                del vis, inp, vals

            peak_gb = (torch.cuda.max_memory_allocated(DEVICE) - mem_base) / 1e9 if cand else 0.0
            out["scoring_time"].append({"n_cand": len(cand), "peak_gb": round(peak_gb, 3),
                                        "cum_s": {str(k): round(v, 4) for k, v in cum.items()}})

            answer_cache = {}
            for (sig, Lnum), tag in zip(combos, tags):
                ranked = sorted(cand, key=lambda t: scores[(sig, Lnum)][t], reverse=True)
                chosen = sorted({thumb, *ranked[: max(0, K - 1)]})
                out["tile_selection"][tag].append({
                    "num_tiles": num_tiles, "cols": cols, "rows": rows, "chosen_tiles": chosen, "K": K,
                    "tile_scores": {int(t): v for t, v in scores[(sig, Lnum)].items()},
                })
                if args.skip_generate:
                    out["candidates"][tag].append("")
                    continue
                key = tuple(chosen)
                if key not in answer_cache:
                    vision = torch.stack([blocks[t] for t in chosen], 0).reshape(1, -1, eb.shape[-1])
                    emb = torch.cat([eb, vision, ea], dim=1)
                    ids = model.language_model.generate(
                        inputs_embeds=emb,
                        attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=DEVICE),
                        do_sample=False, max_new_tokens=args.max_new_tokens,
                        eos_token_id=eos_id, pad_token_id=pad_id, use_cache=True)
                    answer_cache[key] = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
                    del vision, emb, ids
                out["candidates"][tag].append(answer_cache[key])
            del blocks

        tline = "  ".join(f"L{k}={v:.2f}s" for k, v in cum.items())
        print(f"  scoring {len(cand)} tiles (peak +{peak_gb:.2f}GB): {tline}")
        print(f"  distinct tile sets across {len(combos)} configs: {len({tuple(out['tile_selection'][t][-1]['chosen_tiles']) for t in tags})}")

        del pixel_values
        gc.collect()
        torch.cuda.empty_cache()
        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nAll done. Total time: {time.time() - t_all:.1f}s")
    if args.save:
        print(f"Saved {args.output_json}")
        report(args.output_json)


if __name__ == "__main__":
    main()
