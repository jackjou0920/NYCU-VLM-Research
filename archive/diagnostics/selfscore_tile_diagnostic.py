"""
第 1 步：「獨立打分」不看 GT，能逼近 per_tile oracle 到什麼程度？

per_tile oracle（oracle_tile_diagnostic.py --oracle_mode per_tile）對每個 grid tile 用
context = [問題] + [這塊 tile + thumbnail] 量「GT 答案的 log-prob」。這裡用『完全一樣的
context』，但把「偷看 GT」換成「只看模型自己的輸出」，每塊 tile 仍然獨立打分，所以跟 SVMA
逐 tile 串流的性質相容（每塊 tile 進來時 {tile, thumbnail, 問題} 跑一次短 forward，
打完分就丟，不寫進 KV、不看其他 tile）：

  neg_entropy     : 第一個答案 token 分佈的負熵（越大 = 模型越確定）
  top1_prob       : 第一個答案 token 的最大機率
  answer_logprob  : 模型自己 greedy 生成前 --n_gen 個 token 的平均 log-prob（到 EOS 為止）

三種分數共用同一次 generate（第 1 步 logits 算前兩個，整段序列算第三個），一次跑完。

評估兩層：
  (a) 打分品質（不需要生成答案，最快）：跟 oracle per_tile 的 tile 分數比
      Spearman 相關、oracle 最佳 tile 命中率、oracle top-(K-1) 召回、選中 tile 在 oracle
      分數下的 z-score；同一張表放 SVMA（從 run log 解析）與隨機當對照。
  (b) 真正的 ANLS：每個分數各自選 top-(K-1) + thumbnail，正常 generate，輸出
      跟 fastv/ifprune 同格式的 json，直接餵 evaluate_anls.py。

前提：--oracle_json 要是「新版」oracle_tile_diagnostic.py 跑出來、有 tile_selection 的檔案
（其 tile_scores 是 per_tile 分數：greedy 第 1 輪的分數等於 per_tile 分數，兩者都行）。

用法：
    # 跑打分 + 生成（一次 200 張，三種分數各自生成）
    python selfscore_tile_diagnostic.py --use_ds --dataset docvqa --num_images 200 --budget 1024 \
        --reuse_from out_oracle_sel_docvqa_b1024.json --save --output_json out_selfscore_docvqa_b1024.json
    # 只看打分品質、不生成（快很多）加 --skip_generate

    # 只重算報表（不載入模型）
    python selfscore_tile_diagnostic.py --report_only --output_json out_selfscore_docvqa_b1024.json \
        --oracle_json out_oracle_sel_docvqa_b1024.json \
        --svma_log results/svma_v1/internvl_svma_docvqa/output_short_docvqa.log --budget 1024

    python evaluate_anls.py --input_json out_selfscore_docvqa_b1024.json \
        --baseline_tag selfscore_answer_logprob_budget=1024
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time

import torch

SCORERS = ["neg_entropy", "top1_prob", "answer_logprob"]      # 需要跑模型的分數
ALL_SCORERS = SCORERS + ["random"]                             # random：對照組，不跑模型，同一條生成路徑


def tag_of(scorer: str, budget: int) -> str:
    return f"selfscore_{scorer}_budget={budget}"


# ──────────────────────────────────────────────────────────────────────────
# 純函式：從 generate 的 scores / sequences 算三種分數（可單獨測試）
# ──────────────────────────────────────────────────────────────────────────
def scorer_values(step_scores: torch.Tensor, seqs: torch.Tensor, eos_id: int) -> dict[str, torch.Tensor]:
    """step_scores: [B, T, V]（每步 logits）；seqs: [B, T]（生成的 token）。回傳每個 scorer 的 [B] 分數。"""
    lp = torch.log_softmax(step_scores.float(), dim=-1)                # [B, T, V]
    first = lp[:, 0, :]                                                # [B, V]
    p = first.exp()
    neg_entropy = (p * first).sum(dim=-1)                             # = -H
    top1_prob = p.max(dim=-1).values

    tok_lp = lp.gather(-1, seqs[:, : lp.shape[1]].unsqueeze(-1)).squeeze(-1)   # [B, T]
    is_eos = seqs[:, : lp.shape[1]] == eos_id
    before_eos = (torch.cumsum(is_eos.long(), dim=1) == 0)             # EOS 之前（不含 EOS）
    before_eos[:, 0] = True                                            # 第一個 token 一定算（避免空集合）
    answer_logprob = (tok_lp * before_eos).sum(dim=1) / before_eos.sum(dim=1).clamp(min=1)
    return {"neg_entropy": neg_entropy, "top1_prob": top1_prob, "answer_logprob": answer_logprob}


# ──────────────────────────────────────────────────────────────────────────
# 報表（不需要模型）
# ──────────────────────────────────────────────────────────────────────────
def _rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    for pos, i in enumerate(order):
        r[i] = float(pos)
    return r


def _spearman(a, b):
    ra, rb = _rank(a), _rank(b)
    n = len(a)
    ma, mb = sum(ra) / n, sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    return cov / (va * vb + 1e-12)


def report(output_json: str, oracle_json: str, svma_log: str | None, budget: int, min_grid: int = 4):
    from analyze_tile_selection import parse_svma_log
    with open(output_json, encoding="utf-8") as f:
        mine = json.load(f).get("tile_selection", {})
    with open(oracle_json, encoding="utf-8") as f:
        oracle = next(iter(json.load(f)["tile_selection"].values()))
    svma = parse_svma_log(svma_log, budget) if svma_log else None

    rows = {}   # name -> lists
    def add(name, key, val):
        rows.setdefault(name, {}).setdefault(key, []).append(val)

    n_used = 0
    rng = random.Random(0)
    for tag, sel_list in mine.items():
        name = tag.replace("selfscore_", "").replace(f"_budget={budget}", "")
        for i, x in enumerate(sel_list):
            if i >= len(oracle):
                break
            o = oracle[i]
            g = o["num_tiles"] - 1
            if g < min_grid or x["num_tiles"] != o["num_tiles"]:
                continue
            osc = {int(k): v for k, v in o["tile_scores"].items() if int(k) < g}
            msc = {int(k): v for k, v in x["tile_scores"].items() if int(k) < g}
            if len(osc) < min_grid or set(osc) != set(msc):
                continue
            k = max(1, min(g, o["K"] - 1))
            tiles = sorted(osc)
            ovals = [osc[t] for t in tiles]
            mu = sum(ovals) / len(ovals)
            sd = (sum((v - mu) ** 2 for v in ovals) / len(ovals)) ** 0.5 + 1e-9
            o_rank = sorted(tiles, key=lambda t: osc[t], reverse=True)
            o_top, o_topk = o_rank[0], set(o_rank[:k])
            m_rank = sorted(tiles, key=lambda t: msc[t], reverse=True)
            picks = m_rank[:k]
            add(name, "spearman", _spearman([msc[t] for t in tiles], ovals))
            add(name, "top1_hit", float(o_top in picks))
            add(name, "topk_recall", len(o_topk & set(picks)) / k)
            add(name, "z", sum((osc[t] - mu) / sd for t in picks) / k)
            if tag == next(iter(mine)):      # 對照列只算一次
                n_used += 1
                add("[oracle top-k]", "z", sum((osc[t] - mu) / sd for t in o_rank[:k]) / k)
                add("[oracle top-k]", "top1_hit", 1.0); add("[oracle top-k]", "topk_recall", 1.0)
                add("[oracle top-k]", "spearman", 1.0)
                rp = rng.sample(tiles, k)
                add("[random]", "z", sum((osc[t] - mu) / sd for t in rp) / k)
                add("[random]", "top1_hit", k / g); add("[random]", "topk_recall", k / g)
                add("[random]", "spearman", 0.0)
                if svma is not None and i < len(svma) and svma[i][0] == o["num_tiles"]:
                    sp = [t for t in svma[i][1] if t != g and t in osc]
                    if sp:
                        add("[SVMA picks]", "z", sum((osc[t] - mu) / sd for t in sp) / len(sp))
                        add("[SVMA picks]", "top1_hit", float(o_top in sp))
                        add("[SVMA picks]", "topk_recall", len(o_topk & set(sp)) / k)

    def f(a):
        return sum(a) / len(a) if a else float("nan")

    print(f"\n打分品質（相對 oracle per_tile 分數；評估圖數 = {n_used}，僅含 grid tile >= {min_grid} 的圖）")
    print(f"{'scorer':18s} {'Spearman':>9s} {'top1命中':>9s} {'topK召回':>9s} {'選中tile z':>11s}")
    order = [n for n in rows if not n.startswith("[")] + ["[SVMA picks]", "[random]", "[oracle top-k]"]
    for n in order:
        if n not in rows:
            continue
        r = rows[n]
        sp = f"{f(r['spearman']):+.3f}" if "spearman" in r and n not in ("[SVMA picks]",) else "   -  "
        print(f"{n:18s} {sp:>9s} {f(r['top1_hit']):9.1%} {f(r['topk_recall']):9.1%} {f(r['z']):+11.2f}")
    print("z：選中 tile 在 oracle per_tile 分數下的平均 z-score（隨機 0；[oracle top-k] 是不看 GT 之外的理論上限）")


# ──────────────────────────────────────────────────────────────────────────
# 主流程（需要模型）
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--max_num", type=int, default=48)
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--n_gen", type=int, default=8, help="answer_logprob 讓模型自己 greedy 生成的 token 數")
    ap.add_argument("--score_batch_size", type=int, default=8, help="一次疊幾個候選 tile 一起打分")
    ap.add_argument("--max_new_tokens", type=int, default=500)
    ap.add_argument("--scorers", type=str, default=",".join(SCORERS),
                    help=f"逗號分隔，可選 {ALL_SCORERS}。random = 隨機選 tile 的對照組（每張圖固定 seed）；"
                         "只給 random 時不會跑任何打分 forward，只做選 tile + 生成")
    ap.add_argument("--use_ds", action="store_true")
    ap.add_argument("--dataset", type=str, default="docvqa", choices=["docvqa", "infovqa"])
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--reuse_from", type=str, default=None,
                    help="沿用既有 oracle json 的 references / meta（不用重跑 batch_chat 的 uncompressed baseline）")
    ap.add_argument("--skip_generate", action="store_true", help="只打分、不生成答案（只看打分品質）")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output_json", type=str, default="out_selfscore.json")
    ap.add_argument("--report_only", action="store_true", help="不載入模型，只用 --output_json 重算報表")
    ap.add_argument("--oracle_json", type=str, default=None)
    ap.add_argument("--svma_log", type=str, default=None)
    args = ap.parse_args()

    if args.report_only:
        if not args.oracle_json:
            raise SystemExit("--report_only 需要 --oracle_json")
        report(args.output_json, args.oracle_json, args.svma_log, args.budget)
        return

    from internvl_stream import (
        build_prompt, split_prompt_at_vision, encode_tile, build_model, load_image_tiles,
        DEVICE, DOCVQA_PROMPT, INFOVQA_PROMPT, _infer_tile_grid,
    )
    from sparsevlm_internvl import _get_eos_token_id

    dtype = torch.bfloat16
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    num_image_token = model.num_image_token
    K = max(1, args.budget // num_image_token)
    eos_id = _get_eos_token_id(tokenizer, model)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else eos_id
    embed_fn = model.language_model.get_input_embeddings()

    from process_common import load_docvqa, load_infovqa
    if args.dataset == "docvqa":
        datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
    else:
        datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
    print(f"Found {len(datasets)} image(s); K={K} (含 thumbnail)")

    chosen_scorers = [x.strip() for x in args.scorers.split(",") if x.strip()]
    bad = [x for x in chosen_scorers if x not in ALL_SCORERS]
    if bad:
        raise SystemExit(f"未知 scorer {bad}，可選 {ALL_SCORERS}")
    need_model_scoring = any(x in SCORERS for x in chosen_scorers)
    tags = [tag_of(x, args.budget) for x in chosen_scorers]
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            out = json.load(f)
        out.setdefault("references", []); out.setdefault("candidates", {}); out.setdefault("tile_selection", {})
    else:
        out = {"references": [], "candidates": {}, "tile_selection": {}}
        if args.reuse_from:
            with open(args.reuse_from, "r", encoding="utf-8") as f:
                src = json.load(f)
            out["references"] = src.get("references", [])[: len(datasets)]
            if "meta" in src:
                out["meta"] = src["meta"][: len(datasets)]
    for t in tags:
        out["candidates"].setdefault(t, []); out["tile_selection"].setdefault(t, [])
    if "meta" not in out:
        out["meta"] = [{"answer": d.get("answer", ""), "answers": d.get("answers", []),
                        "question_type": d.get("question_type", "docvqa"), "id": d.get("id", str(k))}
                       for k, d in enumerate(datasets)]
    start = len(out["tile_selection"][tags[0]])
    if start:
        print(f"Resuming: {start}/{len(datasets)} images already done")

    t0 = time.time()
    for i, d in enumerate(datasets):
        if i < start:
            continue
        print(f"\n[{i+1}/{len(datasets)}] ", end="")
        pixel_values = load_image_tiles(d["image"], input_size=448, max_num=args.max_num)
        num_tiles = pixel_values.shape[0]
        thumb = num_tiles - 1                      # 同 internvl_stream.py：縮圖是最後一個 tile
        cols, rows = _infer_tile_grid(max(num_tiles - 1, 1), d["image"].size[0], d["image"].size[1])
        print(f"tiles={num_tiles} (grid {cols}x{rows})")

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
            per_scorer = {x: {} for x in chosen_scorers}
            torch.cuda.synchronize(DEVICE)
            mem_base = torch.cuda.memory_allocated(DEVICE)      # 打分前已常駐的量（模型 + 所有 tile embedding）
            torch.cuda.reset_peak_memory_stats(DEVICE)
            t_sc = time.time()
            for s in range(0, len(cand) if need_model_scoring else 0, args.score_batch_size):
                bt = cand[s:s + args.score_batch_size]
                B = len(bt)
                vis = torch.stack([torch.cat([blocks[min(t, thumb)], blocks[max(t, thumb)]], dim=0) for t in bt], 0)
                inp = torch.cat([eb.expand(B, -1, -1), vis, ea.expand(B, -1, -1)], dim=1)
                att = torch.ones(inp.shape[:2], dtype=torch.long, device=DEVICE)
                if args.n_gen == 1:
                    # 只需要第一個答案 token 的分佈：單次 forward，不建 KV、只算最後位置的 logits，
                    # 打分階段的暫存記憶體最小（peak memory 是主張指標，不要被打分拉高）
                    g = model.language_model(inputs_embeds=inp, attention_mask=att, use_cache=False,
                                             logits_to_keep=1, return_dict=True)
                    step = g.logits[:, -1:, :]
                    vals = scorer_values(step, step.argmax(dim=-1), eos_id)
                else:
                    g = model.language_model.generate(
                        inputs_embeds=inp, attention_mask=att,
                        do_sample=False, max_new_tokens=args.n_gen, eos_token_id=eos_id, pad_token_id=pad_id,
                        use_cache=True, output_scores=True, return_dict_in_generate=True)
                    vals = scorer_values(torch.stack(g.scores, dim=1), g.sequences, eos_id)
                for name in chosen_scorers:
                    if name in vals:
                        for t, v in zip(bt, vals[name].tolist()):
                            per_scorer[name][t] = v
                del vis, inp, g, vals
            torch.cuda.synchronize(DEVICE)
            score_s = time.time() - t_sc
            score_peak_gb = (torch.cuda.max_memory_allocated(DEVICE) - mem_base) / 1e9
            if "random" in chosen_scorers:
                rr = random.Random(1000 + i)
                per_scorer["random"] = {t: rr.random() for t in cand}

            answer_cache = {}
            for name, tag in zip(chosen_scorers, tags):
                ranked = sorted(cand, key=lambda t: per_scorer[name][t], reverse=True)
                chosen = sorted({thumb, *ranked[: max(0, K - 1)]}) if cand else [thumb]
                out["tile_selection"][tag].append({
                    "num_tiles": num_tiles, "cols": cols, "rows": rows, "chosen_tiles": chosen, "K": K,
                    "tile_scores": {int(t): v for t, v in per_scorer[name].items()},
                })
                if args.skip_generate:
                    out["candidates"][tag].append("")
                    continue
                key = tuple(chosen)
                if key not in answer_cache:
                    vision = torch.stack([blocks[t] for t in chosen], 0).reshape(1, -1, eb.shape[-1])
                    emb = torch.cat([eb, vision, ea], dim=1)
                    ids = model.language_model.generate(
                        inputs_embeds=emb, attention_mask=torch.ones(emb.shape[:2], dtype=torch.long, device=DEVICE),
                        do_sample=False, max_new_tokens=args.max_new_tokens, eos_token_id=eos_id,
                        pad_token_id=pad_id, use_cache=True)
                    answer_cache[key] = tokenizer.batch_decode(ids, skip_special_tokens=True)[0]
                    del vision, emb, ids
                out["candidates"][tag].append(answer_cache[key])
            print(f"  scoring {len(cand)} tiles: {score_s:.2f}s (peak +{score_peak_gb:.2f}GB)   picks: " +
                  " | ".join(f"{n[:6]}={out['tile_selection'][t][-1]['chosen_tiles']}" for n, t in zip(chosen_scorers, tags)))
            del blocks

        del pixel_values
        gc.collect(); torch.cuda.empty_cache()
        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nAll done. Total time: {time.time() - t0:.1f}s")
    if args.save:
        print(f"Saved {args.output_json}")
        if args.oracle_json:
            report(args.output_json, args.oracle_json, args.svma_log, args.budget)


if __name__ == "__main__":
    main()
