"""
P1 診斷：SVMA 跟 FastV / IF-Prune 的 accuracy 差距，是「打分訊號」還是「選擇單位」造成的？

做法：同一張圖、同一個 budget，把「打分器 (scorer)」跟「選擇單位 (unit)」拆成兩個獨立
的軸做 factorial，其餘（prompt、縮圖保護、剪枝位置、decode）全部固定成 SVMA 的樣子：

  scorer（每個視覺 token 一個分數，每張圖只算一次，所有組合共用）
    svma    : SVMA 目前的收斂設定 = (1-w)*info_density(token) + w*thumbnail prior(tile)，w=THUMB_ATTN_W
    fastv   : 大模型 layer agg_layer-1 的「最後一個 query 對視覺 token」attention（FastV 官方訊號，
              看過全圖 + 問題；這裡是『全圖』算的，非 streaming，所以是 streaming 近似的上界）
    ifprune : IF-Prune 小 VLM（InternVL2.5-1B + KL head）在全圖上的 importance（同上，全圖上界）
    random  : 對照組
  unit（分數怎麼聚合成「一起留或一起丟」的單位；聚合方式 --agg，預設 max 跟 SVMA 一致）
    tile    : 整個 tile（256 token）＝ SVMA 現況
    blockN  : tile 內 16x16 token 網格切成 NxN 的連續小塊（block8=64 token、block4=16 token）
    token   : 單一 token（＝ FastV / IF-Prune 的粒度）
  pos（保留 token 的 RoPE position）
    contig  : 連續重新編號（SVMA 舊行為）
    abs     : 保留未壓縮時的絕對位置（P0，internvl_stream.py --abs_pos）

固定的部分（刻意跟 SVMA 一致，才能回答「SVMA 換掉某一軸會怎樣」）：
  * 縮圖 tile（最後一塊）永遠整塊保留，剩下的 budget - 256 才分給其他單位
  * 剪枝發生在 LLM 輸入端（layer 0），只把保留的 token 餵進 LLM —— 跟 SVMA 一樣。
    注意這『不是』FastV / IF-Prune 原版（它們在 layer 2 才剪，前兩層看得到全部 token），
    所以 `ifprune_token_abs` 不會完全等於 IF-Prune 的表內數字；它回答的是「IF-Prune 的訊號
    如果搬到 SVMA 這種 input-side 剪枝，能拿到多少」。
  * tile 切法跟 SVMA 一樣：--max_num 是「含縮圖」的總 tile 上限（網格 = max_num-1）

怎麼讀結果（evaluate_anls.py / evaluate_hrbench_circular.py 直接吃輸出 JSON）：
  * ifprune_tile_abs ≈ ifprune_token_abs 且都遠高於 svma_tile_abs → 差距幾乎全在打分，
    tile 單位沒問題 → 投資 P2（把好訊號搬進 streaming）
  * ifprune_tile_abs 明顯 < ifprune_block*_abs / ifprune_token_abs → 單位也是瓶頸 → P3 一起做
  * *_abs vs *_contig → P0 的效果（每個 scorer/unit 下各看一次）
  JSON 另外存 diag_stats：每題的 num_tiles、每個 tag 實際碰到幾個 tile（覆蓋率），方便按圖大小拆。

用法：
    python diag_scorer_unit.py --use_ds --dataset docvqa --num_images 1000 \
        --budgets 1024,2048 --scorers svma,fastv,ifprune,random \
        --units tile,block8,token --pos abs,contig \
        --output_json diag_docvqa_n1000.json
    python evaluate_anls.py --input_json diag_docvqa_n1000.json --baseline_tag diag_b1024_svma_tile_contig
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from internvl_stream import (
    MAX_NEW_TOKENS, THUMB_ATTN_W, DEVICE, RANDOM_SEED,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
    build_model, load_image_tiles, build_prompt, split_prompt_at_vision, encode_tile,
    clean_answer, dedup_hrbench_open, strip_to_question, mean_pool_question_embed,
    score_information_density, GlobalNormStats, _infer_tile_grid, InternVLAdapter,
)
from sparsevlm_internvl import _segment_geometry
from fastv_internvl import _layer_forward_capture_last_row


# ──────────────────────────────────────────────────────────────────────────
# Scorers：全部回傳 [T, n_tok] 的 per-token 分數（T = tile 數，含最後的縮圖）
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def score_svma(model, adapter, vis, pixel_values, question, orig_size, dtype, thumb_attn_w):
    """SVMA 的 token 分數：info_density（全域 norm 統計量）混 thumbnail prior。
    對 tile 取 max 時 = (1-w)*max(info_density) + w*prior，跟 SVMA 的 tile 分數（tile_agg=max）
    一致；差別只在 SVMA 的 streaming eviction 用的是『到貨當下』的統計量，這裡用全圖統計量。"""
    T = vis.shape[0]
    tokenizer = adapter.get_tokenizer()
    qemb = mean_pool_question_embed(adapter.get_input_embeddings(model), tokenizer,
                                    [strip_to_question(question)], dtype)[0]
    norm = GlobalNormStats()
    for t in range(T):
        norm.update(vis[t])
    ns = norm.as_dict(DEVICE)
    info = torch.stack([score_information_density(vis[t], qemb, norm_stats=ns) for t in range(T)])

    prior = torch.zeros(T, device=DEVICE)
    if T > 1 and thumb_attn_w > 0:
        cols, rows = _infer_tile_grid(T - 1, *orig_size)
        prompt = build_prompt(tokenizer, model, question, 1)
        tb, ta = split_prompt_at_vision(prompt, tokenizer)
        p = adapter.thumbnail_saliency_prior(model, pixel_values, tb, ta, cols, rows, dtype)
        if p is not None:
            prior = torch.tensor([p.get(t, 0.0) for t in range(T)], device=DEVICE)
        else:
            thumb_attn_w = 0.0
    return (1 - thumb_attn_w) * info.float() + thumb_attn_w * prior[:, None]


@torch.no_grad()
def score_fastv(model, tokenizer, vis, question, agg_layer):
    """FastV 官方訊號（fastv_core.fastv_select_keep_indices 用的那一列）：全序列跑 layer
    0..agg_layer-1，取 layer agg_layer-1 最後一個 query 對視覺 token 的 attention（head 平均）。"""
    T, n_tok, D = vis.shape
    prompt = build_prompt(tokenizer, model, question, T)
    tb, ta = split_prompt_at_vision(prompt, tokenizer)
    emb = model.language_model.get_input_embeddings()
    e_b = emb(tokenizer(tb, return_tensors="pt").input_ids.to(DEVICE))
    e_a = emb(tokenizer(ta, return_tensors="pt").input_ids.to(DEVICE))
    h = torch.cat([e_b, vis.reshape(1, -1, D), e_a], dim=1)
    vis_start, vis_len = e_b.shape[1], T * n_tok

    lm_model = model.language_model.model
    cache = DynamicCache(config=lm_model.config)
    position_ids, cache_position = _segment_geometry(h, DEVICE)
    causal_mask = create_causal_mask(
        config=lm_model.config, input_embeds=h, attention_mask=None,
        cache_position=cache_position, past_key_values=cache, position_ids=position_ids,
    )
    pos_emb = lm_model.rotary_emb(h, position_ids)
    attn = None
    for layer_idx in range(agg_layer):
        layer = lm_model.layers[layer_idx]
        if layer_idx == agg_layer - 1:
            h, attn = _layer_forward_capture_last_row(
                layer, h, position_ids, pos_emb, cache, layer_idx, cache_position)
        else:
            h = layer(h, attention_mask=causal_mask, position_ids=position_ids,
                      past_key_values=cache, use_cache=True,
                      cache_position=cache_position, position_embeddings=pos_emb)
    s = attn.float().mean(dim=0)[vis_start: vis_start + vis_len]
    del cache, h, attn
    return s.reshape(T, n_tok)


# ──────────────────────────────────────────────────────────────────────────
# 選擇：scores [T, n_tok] → 保留的 flat token index（已排序 = 原始序列順序）
# ──────────────────────────────────────────────────────────────────────────
def unit_size(unit, n_tok):
    if unit == "tile":
        return n_tok
    if unit == "token":
        return 1
    if unit.startswith("block"):
        n = int(unit[len("block"):])
        return n * n
    raise ValueError(f"unknown unit {unit!r}")


def select_tokens(scores, unit, budget, agg="max"):
    """縮圖（最後一塊）整塊保護，剩下 budget-n_tok 依 unit 分數取 top。
    回傳 (kept_flat_idx [K] sorted, tiles_touched)。"""
    T, n_tok = scores.shape
    side = int(round(n_tok ** 0.5))
    if budget >= T * n_tok:
        return torch.arange(T * n_tok, device=scores.device), T
    keep = torch.zeros(T, n_tok, dtype=torch.bool, device=scores.device)
    G = T
    remaining = budget
    if T > 1:
        keep[T - 1] = True
        G = T - 1
        remaining -= n_tok
    if remaining > 0 and G > 0:
        s = scores[:G].float()
        red = (lambda x: x.amax(dim=-1)) if agg == "max" else (lambda x: x.mean(dim=-1))
        if unit == "tile":
            u = red(s)                                                     # [G]
            k = min(remaining // n_tok, G)
            keep[:G][u.topk(k).indices] = True
        elif unit == "token":
            k = min(remaining, G * n_tok)
            flat = keep[:G].reshape(-1)
            flat[s.reshape(-1).topk(k).indices] = True
            keep[:G] = flat.reshape(G, n_tok)
        else:
            n = int(unit[len("block"):])
            assert side % n == 0, f"{unit}: {side}x{side} 網格切不成 {n}x{n}"
            nb = side // n
            # [G, side, side] → [G, nb(row), n, nb(col), n] → [G, nb, nb, n*n]
            blk = s.reshape(G, nb, n, nb, n).permute(0, 1, 3, 2, 4).reshape(G, nb * nb, n * n)
            u = red(blk).reshape(-1)                                       # [G*nb*nb]
            k = min(remaining // (n * n), u.numel())
            m = torch.zeros(u.numel(), dtype=torch.bool, device=scores.device)
            m[u.topk(k).indices] = True
            m = m.reshape(G, nb, nb, 1, 1).expand(G, nb, nb, n, n)
            m = m.permute(0, 1, 3, 2, 4).reshape(G, n_tok)                # 回到 raster order
            keep[:G] |= m
    tiles_touched = int(keep.any(dim=-1).sum().item())
    return keep.reshape(-1).nonzero().view(-1), tiles_touched


# ──────────────────────────────────────────────────────────────────────────
# 生成：只把保留的 token 餵進 LLM（input-side 剪枝，同 SVMA），greedy decode
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_with_kept(model, tokenizer, vis_flat, kept, question, pos_mode, budget,
                       max_new_tokens):
    """一次 prefill [text_before, kept vision, text_after]，position 依 pos_mode：
      contig : 0..L-1
      abs    : vision token 用未壓縮序列裡的位置（vis_start + flat index），text_after 接在
               vis_start + 全部視覺 token 數之後 —— 跟 internvl_stream.py --abs_pos 同語意。
    單次 prefill、沒有 past cache，position 0 只出現在第一個 token，所以 flash_attention_2
    即使把空洞 position 當 packed sequence 也只切出一段（cu_seqlens=[0, L]），結果正確；
    attention_mask 明確給全 1，避免 sdpa/eager 路徑的 packed-sequence 遮罩偵測。"""
    n_eq = max(1, -(-budget // model.num_image_token))
    prompt = build_prompt(tokenizer, model, question, n_eq)     # 跟 SVMA 同一個 prompt
    tb, ta = split_prompt_at_vision(prompt, tokenizer)
    emb = model.language_model.get_input_embeddings()
    ids_b = tokenizer(tb, return_tensors="pt").input_ids.to(DEVICE)
    ids_a = tokenizer(ta, return_tensors="pt").input_ids.to(DEVICE)
    e_b, e_a = emb(ids_b), emb(ids_a)
    v = vis_flat[kept].unsqueeze(0)
    h = torch.cat([e_b, v, e_a], dim=1)
    Lb, Lv, La = e_b.shape[1], v.shape[1], e_a.shape[1]

    if pos_mode == "abs":
        after_start = Lb + vis_flat.shape[0]
        pos = torch.cat([
            torch.arange(Lb, device=DEVICE),
            Lb + kept.to(DEVICE),
            torch.arange(after_start, after_start + La, device=DEVICE),
        ])
    else:
        pos = torch.arange(Lb + Lv + La, device=DEVICE)
    next_pos = int(pos[-1].item()) + 1

    mask = torch.ones((1, h.shape[1]), dtype=torch.long, device=DEVICE)
    out = model.language_model(inputs_embeds=h, attention_mask=mask, position_ids=pos.unsqueeze(0),
                               use_cache=True, return_dict=True, logits_to_keep=1)
    cache = out.past_key_values
    nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
    del out, h

    eos = tokenizer.eos_token_id
    gen = [nxt[0, 0].item()]
    for _ in range(max_new_tokens - 1):
        if eos is not None and gen[-1] == eos:
            break
        mask = torch.cat([mask, torch.ones((1, 1), dtype=torch.long, device=DEVICE)], dim=1)
        out = model.language_model(input_ids=nxt, attention_mask=mask,
                                   position_ids=torch.full((1, 1), next_pos, device=DEVICE),
                                   past_key_values=cache, use_cache=True, return_dict=True)
        cache = out.past_key_values
        nxt = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        gen.append(nxt[0, 0].item())
        next_pos += 1
        del out
    del cache
    # 跟 SVMA 一樣 decode 時不跳 special token，再用 clean_answer 拿掉 <|im_end|>
    return clean_answer([tokenizer.decode(gen, skip_special_tokens=False)])[0]


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────
def load_datasets(args):
    from process_common import load_docvqa, load_infovqa, load_hrbench
    if args.dataset == "docvqa":
        return load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                           num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
    if args.dataset == "infovqa":
        return load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                            num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
    ds = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT,
                      num_image=args.num_images, prompt_mode=HRBENCH_PROMPT)
    return dedup_hrbench_open(ds) if HRBENCH_PROMPT == "open" else ds


def main():
    ap = argparse.ArgumentParser(description="P1 診斷：scorer x unit x pos factorial（input-side 剪枝）")
    ap.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--dataset", type=str, default="docvqa", choices=["docvqa", "infovqa", "hrbench"])
    ap.add_argument("--use_ds", action="store_true", help="相容其他腳本的旗標（本腳本一律讀 HF dataset）")
    ap.add_argument("--num_images", type=int, default=None)
    ap.add_argument("--max_num", type=int, default=48, help="含縮圖的總 tile 上限（同 internvl_stream.py）")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--budgets", type=str, default="1024,2048")
    ap.add_argument("--scorers", type=str, default="svma,fastv,ifprune,random")
    ap.add_argument("--units", type=str, default="tile,block8,token",
                    help="tile | blockN（N 整除 16，例如 block8 / block4）| token")
    ap.add_argument("--pos", type=str, default="abs,contig", help="abs | contig")
    ap.add_argument("--agg", type=str, default="max", choices=["max", "mean"],
                    help="unit 內 token 分數的聚合方式（SVMA 預設 max）")
    ap.add_argument("--thumb_attn_w", type=float, default=THUMB_ATTN_W)
    ap.add_argument("--fastv_agg_layer", type=int, default=2)
    ap.add_argument("--small_checkpoint", type=str, default=None, help="IF-Prune 小模型（預設同 ifprune_internvl.py）")
    ap.add_argument("--small_base", type=str, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--output_json", type=str, default="diag_scorer_unit.json")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    scorers = args.scorers.split(",")
    units = args.units.split(",")
    poses = args.pos.split(",")
    for p in poses:
        assert p in ("abs", "contig"), p
    tags = [f"diag_b{b}_{s}_{u}_{p}" for b in budgets for s in scorers for u in units for p in poses]
    if args.agg != "max":
        tags = [t + f"_{args.agg}" for t in tags]

    torch.manual_seed(RANDOM_SEED)
    dtype = torch.bfloat16
    t0 = time.time()
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    n_tok = model.num_image_token
    for u in units:
        unit_size(u, n_tok)   # 早點擋掉打錯的 unit
    adapter = InternVLAdapter(tokenizer, model, budget=max(budgets))

    guide = None
    if "ifprune" in scorers:
        from ifprune_internvl import load_ifprune_guide, DEFAULT_SMALL_CKPT, DEFAULT_SMALL_BASE
        guide = load_ifprune_guide(args.small_checkpoint or DEFAULT_SMALL_CKPT,
                                   args.small_base or DEFAULT_SMALL_BASE, dtype=dtype, device=DEVICE)
        assert guide.model.num_image_token == n_tok

    datasets = load_datasets(args)
    print(f"Found {len(datasets)} samples; {len(tags)} configs per sample")

    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            res = json.load(f)
        print(f"Resuming from {args.output_json}")
    else:
        res = {}
    res.setdefault("references", [])
    res.setdefault("candidates", {})
    res.setdefault("diag_stats", {"num_tiles": [], "tiles_touched": {}})
    res["meta"] = [
        {"answer": d.get("answer", ""), "options": d.get("options", []),
         "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
         "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
         "answers": d.get("answers", [])}
        for k, d in enumerate(datasets)
    ]
    for t in tags:
        res["candidates"].setdefault(t, [])
        res["diag_stats"]["tiles_touched"].setdefault(t, [])
    start = min(len(res["candidates"][t]) for t in tags)

    max_num_grid = max(1, args.max_num - 1)
    for i in range(start, len(datasets)):
        d = datasets[i]
        q = d["question"]
        pv = load_image_tiles(d["image"], input_size=448, max_num=max_num_grid)
        T = pv.shape[0]
        t_s = time.time()

        vis = torch.cat([encode_tile(model, pv[j:j + args.vit_batch].to(DEVICE, dtype=dtype), dtype)
                         for j in range(0, T, args.vit_batch)], dim=0)        # [T, n_tok, D]
        scores = {}
        for s in scorers:
            if s == "svma":
                scores[s] = score_svma(model, adapter, vis, pv, q, tuple(d["image"].size), dtype,
                                       args.thumb_attn_w)
            elif s == "fastv":
                scores[s] = score_fastv(model, tokenizer, vis, q, args.fastv_agg_layer)
            elif s == "ifprune":
                scores[s] = guide.importance(pv, q, vit_batch=args.vit_batch).float().reshape(T, n_tok)
            elif s == "random":
                scores[s] = torch.rand(T, n_tok, device=DEVICE)
            else:
                raise ValueError(f"unknown scorer {s!r}")
        t_score = time.time() - t_s

        vis_flat = vis.reshape(T * n_tok, -1)
        if len(res["diag_stats"]["num_tiles"]) <= i:
            res["diag_stats"]["num_tiles"].append(T)
        for b in budgets:
            for s in scorers:
                for u in units:
                    kept, touched = select_tokens(scores[s], u, b, args.agg)
                    for p in poses:
                        tag = f"diag_b{b}_{s}_{u}_{p}" + (f"_{args.agg}" if args.agg != "max" else "")
                        if len(res["candidates"][tag]) > i:
                            continue
                        ans = generate_with_kept(model, tokenizer, vis_flat, kept, q, p, b,
                                                 args.max_new_tokens)
                        res["candidates"][tag].append(ans)
                        res["diag_stats"]["tiles_touched"][tag].append(touched)
        print(f"[{i + 1}/{len(datasets)}] tiles={T} score={t_score:.2f}s "
              f"total={time.time() - t_s:.2f}s  gt={d.get('answers') or d.get('answer')}  "
              f"e.g. {tags[0]} -> {res['candidates'][tags[0]][i]!r}")

        del vis, vis_flat, scores, pv
        gc.collect()
        torch.cuda.empty_cache()
        if (i + 1) % 20 == 0 or i + 1 == len(datasets):
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(res, f, ensure_ascii=False)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False)

    # 快速總表（DocVQA / InfoVQA 的 ANLS）；正式數字請用 evaluate_anls.py（有 bootstrap CI）
    if args.dataset in ("docvqa", "infovqa"):
        from evaluate_anls import calculate_single_anls
        n = len(datasets)
        print(f"\n{'tag':<44} {'ANLS':>7} {'tiles_touched':>14}")
        for t in tags:
            preds = res["candidates"][t][:n]
            a = sum(calculate_single_anls(p, m["answers"]) for p, m in zip(preds, res["meta"])) / max(len(preds), 1)
            tt = res["diag_stats"]["tiles_touched"][t]
            print(f"{t:<44} {100 * a:7.2f} {sum(tt) / max(len(tt), 1):14.1f}")
    print(f"\nDone in {time.time() - t0:.1f}s -> {args.output_json}")


if __name__ == "__main__":
    main()
