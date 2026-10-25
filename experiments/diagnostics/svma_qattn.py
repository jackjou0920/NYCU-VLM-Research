"""
SVMA-QAttn：SVMA 的 tile scorer 換成「中段層 question→tile attention」（q_attn），
其他維持 SVMA 的設計：streaming、tile 級選擇、有上限的候選池、看完圖後依空間順序寫 KV。

跟 internvl_stream.py（原 SVMA）的差異：
  - tile 分數：原本 = (1-w)·information_density + w·thumbnail_attention_prior（w=0.9）
               這裡 = q_attn：每塊 tile 到貨時跑一次
                     [text_before] + [這塊 tile 256 token] + [text_after(問題 + assistant 開頭)]
                     的前 L 層（預設 L=12，只算到第 L 層的 attention 就停），取 text_after
                     所有位置對 tile 256 token 的 attention 總和（head 平均）再平均。
                     分數只跟「這塊 tile + 問題」有關，打完分就丟，不寫 KV。
  - 移除 information_density / GlobalNormStats / rescore：純 q_attn 不需要
    （InfoVQA 第 200~799 張：純 q_attn 比原 SVMA 高 +7.6 ANLS，b1024 與 b2048 皆顯著）。
  - 移除 thumbnail attention prior（也就省掉那次 eager attention forward）。thumbnail 這塊
    tile 本身仍然一定保留在 KV 裡（全域脈絡）。
  - 移除 delay_tiles / COVERAGE_FLOOR：q_attn 分數不會隨後續 tile 改變，候選池只放 K−1 塊
    grid tile、邊串流邊淘汰最低分，最後留下的就是全域 top-(K−1)；多放 D 塊不會改變結果。
    （delay_tiles=0 時 COVERAGE_FLOOR 本來就不起作用。）

其餘沿用 internvl_stream.py 的元件（不修改它）：build_prompt / split_prompt_at_vision /
encode_tile / InternVLAdapter.lm_prefill / lm_decode_step、text_before 先 prefill、選中 tile
依 tile index 順序 chunked prefill（position 連續，同原 SVMA 預設 abs_pos=False）、
greedy decode、KV cache 估算公式。輸出 json 格式同 internvl_stream.py，evaluate_*.py 直接可用。

記憶體上限：同時常駐的視覺 token = 候選池 K−1 塊 + thumbnail + 當前 ViT micro-batch。

用法：
    python svma_qattn.py --use_ds --dataset infovqa --budget 1024 --run_stream --save \
        --output_json out_svmaqattn_infovqa_b1024.json \
        --ref_json internvl_svma_infovqa/out_block_short_InternVL3_5-8B_infovqa_b1024.json
    python evaluate_anls.py --input_json out_svmaqattn_infovqa_b1024.json --baseline_tag budget=1024_qattnL12
"""
from __future__ import annotations

import argparse
import gc
import heapq
import json
import os
import time

import torch
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import repeat_kv, rotate_half

from internvl_stream import (
    DEVICE, MAX_NEW_TOKENS, RANDOM_SEED,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
    InternVLAdapter, build_model, load_image_tiles, encode_tile,
    build_prompt, split_prompt_at_vision, generate_answer_standard, clean_answer,
    estimate_kv_cache_mb, estimate_prompt_token_lengths, dedup_hrbench_open,
)

QATTN_LAYER = 12      # 1-indexed：用第 12 層的 attention（只需跑前 11 層 + 第 12 層的 q/k）


# ──────────────────────────────────────────────────────────────────────────
# q_attn scorer
# ──────────────────────────────────────────────────────────────────────────
def _question_to_tile_attention(decoder_layer, hidden_states, position_embeddings,
                                q_start: int, tile_start: int, tile_end: int) -> torch.Tensor:
    """這一層的 attention 裡，text_after 每個位置（q_start..L-1）對 tile 區段的權重總和
    （head 平均），回傳 [B, nq]。只手算這幾列，不材料化 [H, L, L]；RoPE 分別套在 q 列與全部 k。"""
    sa = decoder_layer.self_attn
    normed = decoder_layer.input_layernorm(hidden_states)
    B, L, _ = normed.shape
    hd = sa.head_dim
    q = sa.q_norm(sa.q_proj(normed[:, q_start:]).view(B, L - q_start, -1, hd)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(normed).view(B, L, -1, hd)).transpose(1, 2)
    cos, sin = position_embeddings
    cq, sq = cos[:, q_start:].unsqueeze(1), sin[:, q_start:].unsqueeze(1)
    q = q * cq + rotate_half(q) * sq
    k = k * cos.unsqueeze(1) + rotate_half(k) * sin.unsqueeze(1)
    k = repeat_kv(k, sa.num_key_value_groups)
    logits = (q.float() @ k.float().transpose(-2, -1)) * sa.scaling          # [B, H, nq, L]
    q_pos = torch.arange(q_start, L, device=logits.device).view(1, 1, -1, 1)
    k_pos = torch.arange(L, device=logits.device).view(1, 1, 1, -1)
    attn = logits.masked_fill(k_pos > q_pos, float("-inf")).softmax(dim=-1)
    return attn[..., tile_start:tile_end].sum(dim=-1).mean(dim=1)


@torch.no_grad()
def qattn_scores(model, tile_tokens: torch.Tensor, emb_before: torch.Tensor, emb_after: torch.Tensor,
                 layer: int = QATTN_LAYER) -> torch.Tensor:
    """tile_tokens: [B, n_tok, D]（encode_tile 的輸出）。每塊 tile 獨立打分，回傳 [B]。"""
    lm_model = model.language_model.model
    B, n_tok, _ = tile_tokens.shape
    h = torch.cat([emb_before.expand(B, -1, -1), tile_tokens.to(emb_before.dtype),
                   emb_after.expand(B, -1, -1)], dim=1)
    L = h.shape[1]
    t0 = emb_before.shape[1]
    t1 = t0 + n_tok
    position_ids = torch.arange(L, device=h.device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(L, device=h.device)
    mask = create_causal_mask(config=lm_model.config, input_embeds=h, attention_mask=None,
                              cache_position=cache_position, past_key_values=None,
                              position_ids=position_ids)
    pos_emb = lm_model.rotary_emb(h, position_ids)
    for decoder_layer in lm_model.layers[: layer - 1]:
        h = decoder_layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=None,
                          use_cache=False, cache_position=cache_position, position_embeddings=pos_emb)
    mass = _question_to_tile_attention(lm_model.layers[layer - 1], h, pos_emb, t1, t0, t1)
    return mass.mean(dim=-1).float()


class QAttnTilePool:
    """有上限的候選池：保留分數最高的 K−1 塊 grid tile + thumbnail（protected）。
    分數是靜態的，所以串流淘汰的結果 = 全域 top-(K−1)。平手時淘汰 tile index 較大者（同原 SVMA）。"""

    def __init__(self, K: int, thumb_index: int):
        self.cap = max(0, K - 1)
        self.thumb_index = thumb_index
        self.heap = []          # (score, -tile_index, tile_index, tokens)：堆頂 = 最該淘汰的
        self.thumb = None
        self.scores = {}
        self.max_resident = 0

    def add(self, tokens, tile_index: int, score: float | None):
        if tile_index == self.thumb_index:
            self.thumb = tokens
        else:
            self.scores[tile_index] = score
            heapq.heappush(self.heap, (score, -tile_index, tile_index, tokens))
            if len(self.heap) > self.cap:
                heapq.heappop(self.heap)
        self.max_resident = max(self.max_resident, len(self.heap) + (self.thumb is not None))

    def committed(self):
        """回傳 [(tile_index, tokens)]，依 tile index（空間順序）排序。"""
        items = [(t, tok) for _, _, t, tok in self.heap]
        if self.thumb is not None:
            items.append((self.thumb_index, self.thumb))
        return sorted(items, key=lambda x: x[0])


# ──────────────────────────────────────────────────────────────────────────
# 單張圖：text_before prefill → streaming (ViT → q_attn → pool) → commit KV → decode
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def run_svma_qattn(model, adapter: InternVLAdapter, tokenizer, pixel_values, question: str,
                   budget: int, qattn_layer: int = QATTN_LAYER, vit_batch: int = 4,
                   chunk_size: int = 1024, max_new_tokens: int = MAX_NEW_TOKENS,
                   dtype=torch.bfloat16):
    n_img_tok = model.num_image_token
    if budget < n_img_tok:
        raise ValueError(f"budget={budget} < num_image_token={n_img_tok}")
    K = budget // n_img_tok
    num_tiles = pixel_values.shape[0]
    thumb = num_tiles - 1           # dynamic_preprocess 把縮圖放最後；num_tiles==1 時就是唯一那塊
    emb = adapter.get_input_embeddings(model)

    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    t_start = time.time()

    # Step A：text_before prefill（同原 SVMA）
    prompt = build_prompt(tokenizer, model, question, max(1, -(-budget // n_img_tok)))
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
    tok_b = tokenizer([text_before], return_tensors="pt")
    ids_b = tok_b.input_ids.to(DEVICE)
    running_mask = tok_b.attention_mask.to(DEVICE)
    out = adapter.lm_prefill(model, inputs_embeds=emb(ids_b), attention_mask=running_mask,
                             past_key_values=None, use_cache=True, return_dict=True)
    past_key_values = out.past_key_values
    del out

    # q_attn 打分用的 context（跟 prompt 同一套模板；只需算一次）
    emb_b = emb(ids_b)
    emb_a = emb(tokenizer([text_after], return_tensors="pt").input_ids.to(DEVICE))

    # Step B：streaming，每個 ViT micro-batch 編完就打分、進池、淘汰
    pool = QAttnTilePool(K, thumb)
    t_score = 0.0
    for s in range(0, num_tiles, vit_batch):
        e = min(s + vit_batch, num_tiles)
        chunk = pixel_values[s:e].to(DEVICE, dtype=dtype)
        toks = encode_tile(model, chunk, dtype=dtype)                    # [b, n_tok, D]
        del chunk
        grid_rows = [j for j in range(toks.shape[0]) if s + j != thumb]
        scores = {}
        if grid_rows:
            torch.cuda.synchronize(DEVICE)
            t0 = time.time()
            sc = qattn_scores(model, toks[grid_rows], emb_b, emb_a, qattn_layer).tolist()
            torch.cuda.synchronize(DEVICE)
            t_score += time.time() - t0
            scores = {s + j: v for j, v in zip(grid_rows, sc)}
        for j in range(toks.shape[0]):
            idx = s + j
            pool.add(toks[j].clone(), idx, scores.get(idx))
        del toks

    # Step C：commit，依空間順序 chunked prefill（position 連續）
    committed = pool.committed()
    sel_idx = [t for t, _ in committed]
    tiles_per_flush = max(1, chunk_size // n_img_tok)
    injected = 0
    for g in range(0, len(committed), tiles_per_flush):
        vchunk = torch.cat([tok for _, tok in committed[g:g + tiles_per_flush]], 0).unsqueeze(0)
        c = vchunk.shape[1]
        start_pos = past_key_values.get_seq_length()
        running_mask = torch.cat([running_mask, torch.ones((1, c), dtype=torch.long, device=DEVICE)], 1)
        pos = torch.arange(start_pos, start_pos + c, device=DEVICE).unsqueeze(0)
        out = adapter.lm_prefill(model, inputs_embeds=vchunk, attention_mask=running_mask,
                                 position_ids=pos, past_key_values=past_key_values,
                                 use_cache=True, return_dict=True)
        past_key_values = out.past_key_values
        injected += c
        del out, vchunk
    del committed, pool.heap, pool.thumb
    assert injected <= budget, f"KV visual tokens {injected} > budget {budget}"

    # Step D：text_after + greedy decode（同原 SVMA）
    ids_a = tokenizer([text_after], return_tensors="pt").input_ids.to(DEVICE)
    next_pos = past_key_values.get_seq_length()
    running_mask = torch.cat([running_mask, torch.ones_like(ids_a)], 1)
    pos = torch.arange(next_pos, next_pos + ids_a.shape[1], device=DEVICE).unsqueeze(0)
    next_pos += ids_a.shape[1]
    logits, past_key_values = adapter.lm_decode_step(model, input_ids=ids_a, attention_mask=running_mask,
                                                     position_ids=pos, past_key_values=past_key_values,
                                                     use_cache=True, return_dict=True)
    next_token = torch.argmax(logits, dim=-1, keepdim=True)
    torch.cuda.synchronize(DEVICE)
    ttft = time.time() - t_start

    eos = tokenizer.eos_token_id
    answer = tokenizer.decode(next_token[0], skip_special_tokens=False)
    finished = eos is not None and next_token[0, 0].item() == eos
    for _ in range(max_new_tokens):
        if finished:
            break
        running_mask = torch.cat([running_mask, torch.ones((1, 1), dtype=torch.long, device=DEVICE)], 1)
        pos = torch.full((1, 1), next_pos, dtype=torch.long, device=DEVICE)
        next_pos += 1
        logits, past_key_values = adapter.lm_decode_step(model, input_ids=next_token, attention_mask=running_mask,
                                                         position_ids=pos, past_key_values=past_key_values,
                                                         use_cache=True, return_dict=True)
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
        tid = next_token[0, 0].item()
        answer += tokenizer.decode([tid], skip_special_tokens=False)
        finished = eos is not None and tid == eos
    torch.cuda.synchronize(DEVICE)
    peak_gb = torch.cuda.max_memory_allocated(DEVICE) / 1e9

    tb_len, ta_len = estimate_prompt_token_lengths(tokenizer, text_before, text_after)
    n_layers = model.language_model.config.num_hidden_layers
    kv_mb = estimate_kv_cache_mb(model, [tb_len + injected + ta_len] * n_layers)
    del past_key_values
    stats = {
        "num_tiles": num_tiles, "K": K, "selected_tile_indices": sel_idx,
        "final_vis_tokens": injected, "max_resident_tiles": pool.max_resident,
        "ttft_s": round(ttft, 4), "qattn_score_s": round(t_score, 4),
        "peak_gb": round(peak_gb, 3), "kv_cache_mb": round(kv_mb, 2),
    }
    return clean_answer([answer])[0], stats


# ──────────────────────────────────────────────────────────────────────────
# CLI（同 internvl_stream.py 的慣例）
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--max_num", type=int, default=48, help="總 tile 上限（含縮圖），同 internvl_stream.py")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--qattn_layer", type=int, default=QATTN_LAYER)
    ap.add_argument("--run_stream", action="store_true", help="跑 SVMA-QAttn")
    ap.add_argument("--run_standard", action="store_true", help="跑 uncompressed baseline（references）")
    ap.add_argument("--ref_json", type=str, default=None,
                    help="沿用既有 json 的 references / meta（依 --start 切片、用 id 對齊檢查），不用跑 --run_standard")
    ap.add_argument("--use_ds", action="store_true")
    ap.add_argument("--dataset", type=str, default="infovqa", choices=["infovqa", "docvqa", "hrbench"])
    ap.add_argument("--start", type=int, default=0, help="跳過前 N 筆")
    ap.add_argument("--num_images", type=int, default=None, help="None/-1 = 全部")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output_json", type=str, default="output_results_svma_qattn.json")
    args = ap.parse_args()

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    dtype = torch.bfloat16
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    n_layers = len(model.language_model.model.layers)
    if not 1 <= args.qattn_layer <= n_layers:
        raise SystemExit(f"--qattn_layer 要在 1..{n_layers}")
    adapter = InternVLAdapter(tokenizer, model, budget=args.budget)

    if not args.use_ds:
        raise SystemExit("目前只支援 --use_ds")
    from process_common import load_docvqa, load_infovqa, load_hrbench
    n = None if args.num_images in (None, -1) else args.start + args.num_images
    if args.dataset == "infovqa":
        ds = load_infovqa(dataset="Ahren09/InfoVQA", split="val", num_image=n, prompt_mode=INFOVQA_PROMPT)
    elif args.dataset == "docvqa":
        ds = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                         num_image=n, prompt_mode=DOCVQA_PROMPT)
    else:
        ds = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT, num_image=n, prompt_mode=HRBENCH_PROMPT)
        if HRBENCH_PROMPT == "open":
            ds = dedup_hrbench_open(ds)
    datasets = ds[args.start:]
    print(f"{args.dataset}: images [{args.start}, {args.start + len(datasets)})  budget={args.budget}  "
          f"qattn_layer={args.qattn_layer}")

    tag = f"budget={args.budget}_qattnL{args.qattn_layer}"
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, encoding="utf-8") as f:
            out = json.load(f)
    else:
        out = {"references": [], "candidates": {}}
    out.setdefault("references", [])
    out.setdefault("candidates", {})
    out.setdefault("stats", {})
    out["candidates"].setdefault(tag, [])
    out["stats"].setdefault(tag, [])
    if args.ref_json and not out["references"]:
        with open(args.ref_json, encoding="utf-8") as f:
            src = json.load(f)
        out["references"] = src["references"][args.start: args.start + len(datasets)]
        out["meta"] = src["meta"][args.start: args.start + len(datasets)]
        bad = [i for i, (m, d) in enumerate(zip(out["meta"], datasets)) if str(m.get("id")) != str(d.get("id"))]
        if bad:
            raise SystemExit(f"--ref_json 跟資料集 id 對不上（第 {bad[0]} 筆），不能沿用 references")
    if "meta" not in out and datasets and "answer" in datasets[0]:
        out["meta"] = [{"answer": d.get("answer", ""), "options": d.get("options", []),
                        "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
                        "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
                        "answers": d.get("answers", [])}
                       for k, d in enumerate(datasets)]

    max_num_grid = max(1, args.max_num - 1)      # 同 internvl_stream.main：--max_num 是含縮圖的總上限
    t_all = time.time()
    for i, d in enumerate(datasets):
        need_ref = args.run_standard and len(out["references"]) <= i
        need_ours = args.run_stream and len(out["candidates"][tag]) <= i
        if not (need_ref or need_ours):
            continue
        pv = load_image_tiles(d["image"], input_size=448, max_num=max_num_grid)
        print(f"\n[{args.start + i}] ({i + 1}/{len(datasets)}) tiles={pv.shape[0]}")
        if need_ref:
            ref = generate_answer_standard(model, tokenizer, [pv], [d["question"]],
                                           generation_config=dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False))
            out["references"].append(clean_answer(ref)[0])
        if need_ours:
            ans, st = run_svma_qattn(model, adapter, tokenizer, pv, d["question"], args.budget,
                                     qattn_layer=args.qattn_layer, vit_batch=args.vit_batch,
                                     chunk_size=args.chunk_size, dtype=dtype)
            out["candidates"][tag].append(ans)
            out["stats"][tag].append(st)
            print(f"  selected={st['selected_tile_indices']}  TTFT={st['ttft_s']}s "
                  f"(q_attn {st['qattn_score_s']}s)  peak={st['peak_gb']}GB  KV={st['kv_cache_mb']}MB")
            print(f"  answer: {ans}")
        del pv
        gc.collect()
        torch.cuda.empty_cache()
        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    st = out["stats"][tag][1:]      # 略過第 1 張（CUDA warmup）
    if st:
        f = lambda k: sum(x[k] for x in st) / len(st)
        print(f"\n[{tag}] n={len(out['stats'][tag])}  TTFT={f('ttft_s'):.3f}s  q_attn={f('qattn_score_s'):.3f}s  "
              f"peak={f('peak_gb'):.2f}GB  KV={f('kv_cache_mb'):.1f}MB  （略過第 1 張 warmup）")
    print(f"All done. {time.time() - t_all:.1f}s")


if __name__ == "__main__":
    main()
