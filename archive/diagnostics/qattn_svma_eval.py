"""
q_attn（中段層問題→tile attention）接進 SVMA 真正的串流路徑，做「精準版」驗證
（不動 internvl_stream.py；只在這支腳本裡 monkeypatch 兩個模組層級的名字）。

跟 layersweep_tile_diagnostic.py 的差異（為什麼要重跑）：
  1) 換一批圖：L12 是在「前 200 張」上從 24 組設定挑出來的，有挑選偏差。這裡預設用
     --start 200 之後的圖（沒用來挑設定），而且可以跑更多張。
  2) 端到端：直接呼叫 internvl_stream.run_online_kv_with_memory_bank，tile 到貨時
     （每個 ViT micro-batch）就算 q_attn，進有上限的候選池、逐 tile 淘汰、看完圖再依空間
     順序 prefill、decode ＝ SVMA 完全同一條路；TTFT / peak memory / KV 是 pipeline 自己
     量的，含打分成本，跟 SVMA 在同一支程式、同一張卡上比。
  3) 同時跑多個 budget（b1024 有效之後也要看 b2048）。
  4) 同一條路徑順便測 MMR / 2D 多樣性（見下方 selector）。

config 格式 "scorer:selector"
  scorer  : svma      = 現在的 SVMA（info_density + thumbnail prior, w=THUMB_ATTN_W）
            qattn<L>  = 純 q_attn：[text_before]+[這塊 tile]+[text_after] 只跑前 L 層，
                        text_after 所有位置對 tile 256 token 的 attention 總和（head 平均）
                        再平均；不跑 thumbnail forward。每塊 tile 只算一次（分數跟其他
                        tile 無關，不需要 rescore）
  selector: topk      = SVMA 預設（delay_tiles=0）：候選池 = K，逐 tile 淘汰最低分
            mmr<λ>    = 候選池放大到 K + D（--pool_extra），看完圖後 MMR 選 K：
                        每次挑 λ·score − (1−λ)·max_{已選} cos(tile_i, tile_j)
                        （score 在池內 min-max 正規化；cos 用 tile 256 token 平均後的向量）
            grid2d    = 候選池 K + D，前 (K−1) 名照分數，最後 1 個名額給「跟已選 tile 的
                        2D (row, col) 距離最遠」的候選（SVMA 現在的 COVERAGE_FLOOR 用 raster
                        index 距離，而且 delay_tiles=0 時池子剛好 K 個，這個機制其實沒作用）

用法（InfoVQA 第 200~799 張，b1024 + b2048）：
    CUDA_VISIBLE_DEVICES=0 python qattn_svma_eval.py --dataset infovqa --start 200 --num_images 600 \
        --budgets 1024,2048 \
        --ref_json internvl_svma_infovqa/out_block_short_InternVL3_5-8B_infovqa_b1024.json \
        --save --output_json out_qattn_eval_infovqa.json

    python evaluate_anls.py --input_json out_qattn_eval_infovqa.json --baseline_tag svma:topk_budget=1024
    python evaluate_anls.py --input_json out_qattn_eval_infovqa.json --baseline_tag svma:topk_budget=2048
    python qattn_svma_eval.py --report_only --output_json out_qattn_eval_infovqa.json \
        --svma_log internvl_svma_infovqa/output_short_infovqa.log
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F

DEFAULT_CONFIGS = "svma:topk,svma:mmr0.7,svma:grid2d,qattn12:topk,qattn12:mmr0.7"


def parse_config(c: str):
    scorer, selector = c.split(":")
    if scorer == "svma":
        layer = None
    elif scorer.startswith("qattn"):
        layer = int(scorer[len("qattn"):])
    else:
        raise ValueError(f"bad scorer in {c!r}")
    if selector == "topk" or selector == "grid2d":
        lam = None
    elif selector.startswith("mmr"):
        lam = float(selector[3:])
    else:
        raise ValueError(f"bad selector in {c!r}")
    return scorer, layer, selector, lam


def tag_of(c: str, budget: int) -> str:
    return f"{c}_budget={budget}"


class _Ctx:
    """每張圖、每個 config 由主迴圈設定；patched 的 encode_tile / memory bank 讀這裡。"""
    scorer = "svma"
    layer = None
    selector = "topk"
    lam = None
    cols = 1
    model = None
    eb = None
    ea = None
    stash = {}


CTX = _Ctx()


def install_patches(S):
    """monkeypatch internvl_stream 的兩個模組層級名字（stream_image_to_kv 以 global 查找）：
      - encode_tile：q_attn 模式下，ViT micro-batch 一編完就批次算這幾塊 tile 的 q_attn，
                     依 tile tensor 的 data_ptr 暫存（add_tile 收到的正是 tile_tokens[j]）。
      - TileStreamingMemoryBank：q_attn 模式下分數取暫存值、不 rescore；非 topk 的 selector
                     覆寫 flush。topk + svma 完全走原本的程式碼。"""
    from layersweep_tile_diagnostic import score_candidates

    orig_encode = S.encode_tile

    def encode_tile_with_qattn(model, tile_pixel_values, dtype):
        out = orig_encode(model, tile_pixel_values, dtype)
        if CTX.scorer.startswith("qattn"):
            B = out.shape[0]
            eb, ea = CTX.eb, CTX.ea
            inp = torch.cat([eb.expand(B, -1, -1), out.to(eb.dtype), ea.expand(B, -1, -1)], dim=1)
            t0 = eb.shape[1]
            vals, _ = score_candidates(model, inp, t0, t0 + out.shape[1], t0 + out.shape[1],
                                       [CTX.layer], ["q_attn"], out.device)
            for j, v in enumerate(vals[("q_attn", CTX.layer)].tolist()):
                CTX.stash[out[j].data_ptr()] = v
            del inp, vals
        return out

    Base = S.TileStreamingMemoryBank

    class PatchedBank(Base):
        @torch.no_grad()
        def _score_tile(self, tile_tokens):
            if CTX.scorer.startswith("qattn"):
                return CTX.stash.pop(tile_tokens.data_ptr())
            return super()._score_tile(tile_tokens)

        @torch.no_grad()
        def _rescore_pool(self):
            if CTX.scorer.startswith("qattn"):
                return                      # q_attn 分數只跟這塊 tile + 問題有關，不需重算
            return super()._rescore_pool()

        @torch.no_grad()
        def flush(self):
            if CTX.selector == "topk":
                return super().flush()
            if self.rescore and not CTX.scorer.startswith("qattn"):
                super()._rescore_pool()
            protected = sorted((e for e in self.pool if self._protected(e.tile_index)),
                               key=lambda e: e.tile_index)
            rest = [e for e in self.pool if not self._protected(e.tile_index)]
            kept = protected[: self.K]
            keep_n = min(self.K - len(kept), len(rest))
            chosen = []
            if keep_n > 0:
                if CTX.selector == "grid2d":
                    rest.sort(key=lambda e: e.score, reverse=True)
                    chosen = rest[: keep_n - 1]
                    pending = rest[keep_n - 1:]
                    cols = max(1, CTX.cols)

                    def rc(i):
                        return divmod(i, cols)

                    def dist(e):
                        if not chosen:
                            return 0.0
                        r, c = rc(e.tile_index)
                        return min(((r - rc(x.tile_index)[0]) ** 2 + (c - rc(x.tile_index)[1]) ** 2) ** 0.5
                                   for x in chosen)
                    # 平手時取分數高的（pending 已依分數排序，max 取第一個最大值）
                    chosen.append(max(pending, key=dist))
                else:  # mmr
                    s = torch.tensor([e.score for e in rest], dtype=torch.float32)
                    s = (s - s.min()) / (s.max() - s.min() + 1e-9)
                    emb = torch.stack([F.normalize(e.tokens.float().mean(0), dim=-1) for e in rest], 0)
                    sim = (emb @ emb.T).cpu()
                    sel = []
                    for _ in range(keep_n):
                        best, best_v = None, -1e30
                        for i in range(len(rest)):
                            if i in sel:
                                continue
                            red = max((float(sim[i, j]) for j in sel), default=0.0)
                            v = CTX.lam * float(s[i]) - (1 - CTX.lam) * red
                            if v > best_v:
                                best, best_v = i, v
                        sel.append(best)
                    chosen = [rest[i] for i in sel]
            kept = kept + chosen
            kept.sort(key=lambda e: e.tile_index)
            self.total_committed = len(kept)
            self.committed_indices = [e.tile_index for e in kept]
            kept_ids = {id(e) for e in kept}
            for e in self.pool:
                if id(e) not in kept_ids:
                    del e.tokens
            out = [e.tokens for e in kept]
            self.pool = []
            return out

    S.encode_tile = encode_tile_with_qattn
    S.TileStreamingMemoryBank = PatchedBank


def report(output_json: str, svma_log: str | None):
    with open(output_json, encoding="utf-8") as f:
        d = json.load(f)
    cfg = d.get("config", {})
    print(f"\n{'tag':32s} {'n':>5s} {'TTFT':>7s} {'peak GB':>8s} {'KV MB':>7s}   （TTFT/peak 略過第 1 張：warmup）")
    for tag in d["candidates"]:
        t = [x for x in d["ttft"].get(tag, [])[1:] if x is not None]
        p = [x for x in d["peak_gb"].get(tag, [])[1:] if x is not None]
        k = [x for x in d["kv_mb"].get(tag, []) if x is not None]
        f = (lambda a: sum(a) / len(a) if a else float("nan"))
        print(f"{tag:32s} {len(d['candidates'][tag]):5d} {f(t):7.3f} {f(p):8.2f} {f(k):7.1f}")
    if svma_log:
        from analyze_tile_selection import parse_svma_log
        start = cfg.get("start", 0)
        for tag, sel in d["tile_selection"].items():
            if not tag.startswith("svma:topk_"):
                continue
            b = int(tag.split("budget=")[1])
            log = parse_svma_log(svma_log, b)
            n = min(len(sel), max(0, len(log) - start))
            same = sum(sorted(sel[i]) == sorted(log[start + i][1]) for i in range(n))
            print(f"[repro] {tag}: 跟 SVMA 原始 log 一致 {same}/{n} = {same / max(n, 1):.1%}（應接近 100%）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--dataset", type=str, default="infovqa", choices=["docvqa", "infovqa"])
    ap.add_argument("--start", type=int, default=200, help="跳過前 N 張（前 200 張是挑 L12 用的）")
    ap.add_argument("--num_images", type=int, default=600)
    ap.add_argument("--budgets", type=str, default="1024,2048")
    ap.add_argument("--configs", type=str, default=DEFAULT_CONFIGS)
    ap.add_argument("--pool_extra", type=int, default=4,
                    help="mmr / grid2d 的候選池額外名額 D（池 = K + D；多存 D 塊 tile 的 ViT 輸出，每塊約 2MB）")
    ap.add_argument("--max_num", type=int, default=48, help="同 internvl_stream.py：總 tile 上限（含縮圖）")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--chunk_size", type=int, default=1024)
    ap.add_argument("--ref_json", type=str, default=None,
                    help="SVMA 全量 json（同資料集），沿用其 references / meta（依 index 切片、用 id 對齊檢查）")
    ap.add_argument("--save", action="store_true")
    ap.add_argument("--output_json", type=str, default="out_qattn_eval.json")
    ap.add_argument("--report_only", action="store_true")
    ap.add_argument("--svma_log", type=str, default=None)
    args = ap.parse_args()

    if args.report_only:
        report(args.output_json, args.svma_log)
        return

    budgets = [int(x) for x in args.budgets.split(",") if x.strip()]
    configs = [c.strip() for c in args.configs.split(",") if c.strip()]
    for c in configs:
        parse_config(c)

    import internvl_stream as S
    from process_common import load_docvqa, load_infovqa

    install_patches(S)
    torch.manual_seed(S.RANDOM_SEED)
    torch.cuda.manual_seed_all(S.RANDOM_SEED)
    dtype = torch.bfloat16
    tokenizer, model = S.build_model(args.model_name, dtype=dtype, device=S.DEVICE)
    adapter = S.InternVLAdapter(tokenizer, model, budget=max(budgets))
    embed_fn = model.language_model.get_input_embeddings()

    total = args.start + args.num_images
    if args.dataset == "docvqa":
        ds = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                         num_image=total, prompt_mode=S.DOCVQA_PROMPT)
    else:
        ds = load_infovqa(dataset="Ahren09/InfoVQA", split="val", num_image=total, prompt_mode=S.INFOVQA_PROMPT)
    datasets = ds[args.start:]
    print(f"images [{args.start}, {args.start + len(datasets)}) ; budgets={budgets} ; configs={configs}")

    combos = [(b, c) for b in budgets for c in configs]
    tags = [tag_of(c, b) for b, c in combos]
    keys = ("candidates", "tile_selection", "ttft", "peak_gb", "kv_mb")
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, encoding="utf-8") as f:
            out = json.load(f)
    else:
        out = {"references": [], "meta": [], **{k: {} for k in keys}}
        if args.ref_json:
            with open(args.ref_json, encoding="utf-8") as f:
                src = json.load(f)
            out["references"] = src["references"][args.start: args.start + len(datasets)]
            out["meta"] = src["meta"][args.start: args.start + len(datasets)]
            bad = [i for i, (m, d) in enumerate(zip(out["meta"], datasets)) if str(m.get("id")) != str(d.get("id"))]
            if bad:
                raise SystemExit(f"--ref_json 的 meta 跟資料集 id 對不上（例如第 {bad[0]} 筆），不能沿用 references")
        else:
            print("[WARN] 沒給 --ref_json：沒有 uncompressed references，evaluate_anls 的 reference 列會缺")
            out["meta"] = [{"answer": d.get("answer", ""), "answers": d.get("answers", []),
                            "question_type": d.get("question_type", args.dataset), "id": d.get("id", str(k))}
                           for k, d in enumerate(datasets)]
    for k in keys:
        out.setdefault(k, {})
        for t in tags:
            out[k].setdefault(t, [])
    out["config"] = {"start": args.start, "budgets": budgets, "configs": configs, "pool_extra": args.pool_extra,
                     "dataset": args.dataset, "max_num": args.max_num, "thumb_attn_w": S.THUMB_ATTN_W}

    max_num_grid = max(1, args.max_num - 1)          # 同 internvl_stream.main
    t_all = time.time()
    for i, d in enumerate(datasets):
        todo = [((b, c), t) for (b, c), t in zip(combos, tags) if len(out["candidates"][t]) <= i]
        if not todo:
            continue
        pv = S.load_image_tiles(d["image"], input_size=448, max_num=max_num_grid)
        W, H = d["image"].size
        CTX.cols, _ = S._infer_tile_grid(max(pv.shape[0] - 1, 1), W, H)
        prompt = S.build_prompt(tokenizer, model, d["question"], 1)
        tb, ta = S.split_prompt_at_vision(prompt, tokenizer)
        with torch.no_grad():
            CTX.eb = embed_fn(tokenizer(tb, return_tensors="pt").input_ids.to(S.DEVICE))
            CTX.ea = embed_fn(tokenizer(ta, return_tensors="pt").input_ids.to(S.DEVICE))
        print(f"\n######## [{args.start + i}] ({i + 1}/{len(datasets)}) tiles={pv.shape[0]}  ({len(todo)} configs)")
        for (b, c), t in todo:
            scorer, layer, selector, lam = parse_config(c)
            CTX.scorer, CTX.layer, CTX.selector, CTX.lam = scorer, layer, selector, lam
            CTX.stash = {}
            delay = 0 if selector == "topk" else args.pool_extra
            answers, stats, timing = S.run_online_kv_with_memory_bank(
                model, adapter, [pv], [d["question"]],
                dtype=dtype, vit_batch=args.vit_batch, chunk_size=args.chunk_size, budget=b,
                score_fn="info_density", tile_agg="max", select="topk", max_new_tokens=S.MAX_NEW_TOKENS,
                delay_tiles=delay, thumb_attn=(scorer == "svma"), thumb_attn_w=S.THUMB_ATTN_W,
                orig_sizes=[(W, H)], abs_pos=False,
            )
            out["candidates"][t].append(answers[0])
            out["tile_selection"][t].append(stats[0].get("committed_tile_indices", []))
            out["ttft"][t].append(timing.get("TTFT"))
            out["peak_gb"][t].append(timing.get("_peak_mem_gb", {}).get("OVERALL"))
            out["kv_mb"][t].append(stats[0].get("kv_cache_mb"))
            print(f"  >> {t}: tiles={stats[0].get('committed_tile_indices')} TTFT={timing.get('TTFT')}s "
                  f"peak={timing.get('_peak_mem_gb', {}).get('OVERALL')}GB")
        CTX.stash = {}
        del pv
        gc.collect()
        torch.cuda.empty_cache()
        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nAll done. Total time: {time.time() - t_all:.1f}s")
    if args.save:
        print(f"Saved {args.output_json}")
        report(args.output_json, None)


if __name__ == "__main__":
    main()
