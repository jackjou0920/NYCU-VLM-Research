"""
cascade 的錯誤出在哪一階段？（診斷用，不動 internvl_stream.py）

cascade（舊 internvl_stream.py --cascade_m；現行 --qattn_tiles 的邊界打分在 a=K-1 時與之等價）：
  第一階段：thumbnail prior 排序 → 只留前 M = ceil(m*(K-1)) 塊 grid tile 當候選
  第二階段：候選 tile 跑 ViT + q_attn（L12）→ 取最高的 K-1 塊

以 oracle（oracle_tile_diagnostic.py 偷看 GT 答案算的 per-tile 分數，存在
out_oracle_sel_infovqa_b1024.json）當「真正有用的 tile」，對每張圖拆成：
  stage-1 漏掉 : oracle 最佳 tile 不在 prior 前 M 名
  stage-2 選錯 : 在候選裡，但 q_attn 沒把它排進候選中的前 K-1 名
  選中         : oracle 最佳 tile 進了最後的 K-1 塊

另外對每種打分訊號算「跟 oracle 分數的排名相關」（Spearman，逐張算再平均），並測兩個假設：
  (a) 換更深的層：q_attn L8 / L12 / L16 / L20 / L24
  (b) 打分時把 thumbnail 放進 context（[text_before][thumbnail][tile][text_after]），
      讓模型知道整張圖的版面：qattn_ctx L12 / L20
第一階段的 prior 也比兩種：新版 L21-28:q 與舊版 L29-36:last。

降低打分成本的變體（都走正式 pipeline 的 qattn_scores_with_prefix，前綴 KV 重用；另量每批耗時）：
  qattn_ctx_kv_L20   : 現行 cascade 的打分（基準；也用來確認跟 qattn_ctx_L20 整段 forward 一致）
  qattn_ctx_kv_L16   : 改用第 16 層
  qattn_ctx_pool_L20 : 打分時 tile 的 256 token 先 2x2 平均成 64 個（寫進 KV 的仍是 256 個），第 20 層
  qattn_ctx_pool_L16 : 同上，第 16 層

一致性檢查：用 prior L21-28:q + q_attn L12 重算的 cascade 選擇，跟 out_v2 實際跑出來的
tile_selection 比，一致率應接近 100%（bf16 雜訊造成的平手除外）。

只跑 forward、不生成答案。用法：
    python diag_cascade_stages.py --num_images 200 \
        --oracle_json out_oracle_sel_infovqa_b1024.json \
        --v2_json out_v2_InternVL3_5-8B_infovqa_b1024.json \
        --output_json out_diag_cascade_v2_infovqa_b1024.json

    python diag_cascade_stages.py --report_only --output_json out_diag_cascade_v2_infovqa_b1024.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

QATTN_LAYERS = [8, 12, 16, 20, 24]
CTX_LAYERS = [12, 20]
PRIOR_VARIANTS = {"prior_L21-28q": ((21, 28), "q"), "prior_L29-36last": ((29, 36), "last")}
CASC_MS = [1, 2, 3, 4, 5]
# 降低打分成本的變體：{名稱: (層, 打分前是否把 tile token 2x2 平均成 64 個)}
KV_SIGS = {
    "qattn_ctx_kv_L20": (20, False),
    "qattn_ctx_kv_L16": (16, False),
    "qattn_ctx_pool_L20": (20, True),
    "qattn_ctx_pool_L16": (16, True),
}
KEY_SIGS = ["qattn_L12", "qattn_ctx_L20"] + list(KV_SIGS)   # [2] 拆解只列這些（避免報表太長）


def pool_tile_tokens(toks):
    """[B, 256, D] → [B, 64, D]：tile token 是 16x16 依列排列（pixel_shuffle 的輸出），2x2 平均。"""
    B, n, D = toks.shape
    side = int(round(n ** 0.5))
    x = toks.view(B, side // 2, 2, side // 2, 2, D).float().mean(dim=(2, 4))
    return x.reshape(B, -1, D).to(toks.dtype)


# ──────────────────────────────────────────────────────────────────────────
# 收集分數
# ──────────────────────────────────────────────────────────────────────────
def collect(args):
    import torch
    import internvl_stream as S
    from process_common import load_infovqa

    dtype = torch.bfloat16
    tokenizer, model = S.build_model(args.model_name, dtype=dtype, device=S.DEVICE)
    adapter = S.InternVLAdapter(tokenizer, model, budget=args.budget)
    n_tok = model.num_image_token
    K = args.budget // n_tok
    emb = adapter.get_input_embeddings(model)

    oracle = json.load(open(args.oracle_json, encoding="utf-8"))
    otag = next(iter(oracle["tile_selection"]))
    osel = oracle["tile_selection"][otag]
    ometa = oracle["meta"]
    datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                            num_image=args.num_images, prompt_mode=S.INFOVQA_PROMPT)

    records = []
    for i, d in enumerate(datasets[: min(len(datasets), len(osel))]):
        rec = {"idx": i, "id": d.get("id")}
        o = osel[i]
        if str(ometa[i].get("id")) != str(d.get("id")):
            rec["skip"] = "id mismatch"
            records.append(rec)
            continue
        if not o.get("tile_scores"):
            rec["skip"] = "no oracle tile_scores"
            records.append(rec)
            continue
        # 同 internvl_stream.py main()：--max_num 48 扣掉縮圖 → 網格上限 47
        pv = S.load_image_tiles(d["image"], input_size=448, max_num=max(1, args.max_num - 1))
        nt = pv.shape[0]
        if nt != o["num_tiles"]:
            rec["skip"] = f"tile count mismatch (stream {nt} vs oracle {o['num_tiles']})"
            records.append(rec)
            continue
        thumb = nt - 1
        W, H = d["image"].size
        cols, rows = S._infer_tile_grid(nt - 1, W, H)

        prompt = S.build_prompt(tokenizer, model, d["question"], max(1, -(-args.budget // n_tok)))
        tb, ta = S.split_prompt_at_vision(prompt, tokenizer)
        emb_b = emb(tokenizer([tb], return_tensors="pt").input_ids.to(S.DEVICE))
        emb_a = emb(tokenizer([ta], return_tensors="pt").input_ids.to(S.DEVICE))

        t0 = time.time()
        priors = {}
        prefix_kv = None
        kv_upto = max(l for l, _ in KV_SIGS.values())
        for name, (layers, rows_mode) in PRIOR_VARIANTS.items():
            p = adapter.thumbnail_saliency_prior(model, pv, tb, ta, cols, rows, dtype,
                                                 layers=layers, attn_rows=rows_mode,
                                                 capture_kv_upto=kv_upto if name == "prior_L21-28q" else None)
            if name == "prior_L21-28q":
                prefix_kv = adapter._thumb_prefix_kv
            priors[name] = {int(k): float(v) for k, v in p.items() if int(k) != thumb}
        thumb_tok = adapter._thumb_tokens
        adapter._thumb_tokens = None
        adapter._thumb_prefix_kv = None

        scores = {f"qattn_L{l}": {} for l in QATTN_LAYERS}
        scores.update({f"qattn_ctx_L{l}": {} for l in CTX_LAYERS})
        scores.update({sig: {} for sig in KV_SIGS})
        sig_sec = {sig: 0.0 for sig in KV_SIGS}
        grid = list(range(nt - 1))
        lb = emb_b.shape[1]
        for s in range(0, len(grid), args.vit_batch):
            idxs = grid[s: s + args.vit_batch]
            toks = S.encode_tile(model, pv[idxs].to(S.DEVICE, dtype=dtype), dtype=dtype)
            B = toks.shape[0]
            # (a) 原本的 context：[text_before][tile][text_after]
            h = torch.cat([emb_b.expand(B, -1, -1), toks, emb_a.expand(B, -1, -1)], dim=1)
            r = S.prefix_attention_rows(model, h, lb + n_tok, QATTN_LAYERS)
            for l in QATTN_LAYERS:
                v = r[l][..., lb: lb + n_tok].sum(-1).mean(-1).float().tolist()
                for j, idx in enumerate(idxs):
                    scores[f"qattn_L{l}"][idx] = v[j]
            del h, r
            # (b) 加 thumbnail 當版面 context：[text_before][thumbnail][tile][text_after]
            h = torch.cat([emb_b.expand(B, -1, -1), thumb_tok.unsqueeze(0).expand(B, -1, -1),
                           toks, emb_a.expand(B, -1, -1)], dim=1)
            ts = lb + n_tok
            r = S.prefix_attention_rows(model, h, ts + n_tok, CTX_LAYERS)
            for l in CTX_LAYERS:
                v = r[l][..., ts: ts + n_tok].sum(-1).mean(-1).float().tolist()
                for j, idx in enumerate(idxs):
                    scores[f"qattn_ctx_L{l}"][idx] = v[j]
            del h, r
            # (c) 降低成本的變體：重用前綴 KV（同正式 pipeline），逐一計時
            for sig, (layer, pool) in KV_SIGS.items():
                torch.cuda.synchronize(S.DEVICE)
                tk = time.time()
                x = pool_tile_tokens(toks) if pool else toks
                v = S.qattn_scores_with_prefix(model, x, prefix_kv, emb_a, layer, compile_mode="none").tolist()
                torch.cuda.synchronize(S.DEVICE)
                sig_sec[sig] += time.time() - tk
                for j, idx in enumerate(idxs):
                    scores[sig][idx] = v[j]
            del toks
        torch.cuda.synchronize(S.DEVICE)

        rec.update({
            "num_tiles": nt, "K": K, "cols": cols, "rows": rows,
            "oracle": {int(k): float(v) for k, v in o["tile_scores"].items()},
            "oracle_greedy": [t for t in o["chosen_tiles"] if t != thumb],
            "priors": priors, "scores": scores, "sec": round(time.time() - t0, 3),
            "sig_ms_per_tile": {k: 1000 * v / (nt - 1) for k, v in sig_sec.items()},
        })
        records.append(rec)
        print(f"[{i + 1}/{len(datasets)}] tiles={nt} grid={cols}x{rows} {rec['sec']}s")
        if args.output_json and (i + 1) % 20 == 0:
            json.dump({"budget": args.budget, "records": records}, open(args.output_json, "w"))
    out = {"budget": args.budget, "records": records}
    if args.output_json:
        json.dump(out, open(args.output_json, "w"))
        print(f"[saved] {args.output_json}")
    return out


# ──────────────────────────────────────────────────────────────────────────
# 分析
# ──────────────────────────────────────────────────────────────────────────
def _rank_desc(d: dict) -> list:
    """依分數由高到低；平手時 tile index 小的在前（同 cascade 的淘汰規則）。"""
    return sorted(d, key=lambda t: (-d[t], t))


def _spearman(a: dict, b: dict) -> float | None:
    keys = sorted(set(a) & set(b))
    n = len(keys)
    if n < 3:
        return None

    def ranks(d):
        order = sorted(keys, key=lambda t: d[t])
        r = {}
        i = 0
        while i < n:                                   # 平手取平均名次
            j = i
            while j + 1 < n and d[order[j + 1]] == d[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2
            i = j + 1
        return r

    ra, rb = ranks(a), ranks(b)
    ma = sum(ra.values()) / n
    mb = sum(rb.values()) / n
    cov = sum((ra[k] - ma) * (rb[k] - mb) for k in keys)
    va = sum((ra[k] - ma) ** 2 for k in keys)
    vb = sum((rb[k] - mb) ** 2 for k in keys)
    return cov / math.sqrt(va * vb) if va > 0 and vb > 0 else None


def cascade_select(prior: dict, score: dict, slots: int, m: float | None) -> tuple[list, list]:
    """回傳 (候選, 最後選中的 grid tile)。m=None = 不篩（全部 tile 都是候選）。"""
    grid = sorted(prior)
    M = len(grid) if m is None else min(len(grid), max(slots, int(math.ceil(m * slots))))
    cand = _rank_desc(prior)[:M]
    if M <= slots:
        return cand, cand
    sel = _rank_desc({t: score[t] for t in cand})[:slots]
    return cand, sel


def _int_keys(d: dict) -> dict:
    return {int(k): v for k, v in d.items()}


def report(out: dict, v2_json: str | None):
    recs = [r for r in out["records"] if "skip" not in r]
    for r in recs:   # json 存回來 key 會變字串，統一成 int
        r["oracle"] = _int_keys(r["oracle"])
        r["priors"] = {k: _int_keys(v) for k, v in r["priors"].items()}
        r["scores"] = {k: _int_keys(v) for k, v in r["scores"].items()}
    skips = [r for r in out["records"] if "skip" in r]
    pct = lambda a: 100 * sum(a) / max(1, len(a))
    print(f"\n有效 {len(recs)} 張，跳過 {len(skips)} 張", end="")
    if skips:
        from collections import Counter
        print("：" + "；".join(f"{k.split(' (')[0]} x{v}" for k, v in Counter(r['skip'] for r in skips).items()))
    else:
        print()
    if not recs:
        return
    K = recs[0]["K"]
    slots = K - 1
    ng = sum(r["num_tiles"] - 1 for r in recs) / len(recs)
    print(f"budget={out['budget']} K={K}（選 {slots} 塊 grid tile），平均 grid tile 數 {ng:.1f}；"
          f"隨機挑 {slots} 塊選中 oracle 最佳 tile 的機率約 {100 * slots / ng:.1f}%")

    # ---- 1. 各打分訊號跟 oracle 的一致程度 ----
    print("\n[1] 打分訊號 vs oracle（per-tile 分數）")
    print(f"  {'訊號':22s} {'Spearman':>9s} {'oracle最佳tile的名次(中位數)':>26s} {'最佳tile在前K-1':>15s} {'ms/tile':>8s}")
    sigs = list(recs[0]["priors"]) + list(recs[0]["scores"])
    for sig in sigs:
        rho, rank_pct, top = [], [], []
        for r in recs:
            sc = r["priors"].get(sig) or r["scores"].get(sig)
            o = r["oracle"]
            best = max(o, key=lambda t: o[t])
            x = _spearman(sc, o)
            if x is not None:
                rho.append(x)
            rk = _rank_desc(sc).index(best) + 1
            rank_pct.append(rk / len(sc))
            top.append(rk <= slots)
        rank_pct.sort()
        med = rank_pct[len(rank_pct) // 2]
        ms = [r.get("sig_ms_per_tile", {}).get(sig) for r in recs]
        ms = [x for x in ms[1:] if x is not None]                      # 略過第 1 張（warmup）
        ms_s = f"{sum(ms) / len(ms):8.1f}" if ms else f"{'—':>8s}"
        print(f"  {sig:22s} {sum(rho) / len(rho):9.3f} {100 * med:21.0f}% 處 {pct(top):14.1f}% {ms_s}")

    # ---- 2. cascade 拆解：stage-1 漏掉 / stage-2 選錯 / 選中 ----
    print("\n[2] cascade 拆解（以 oracle 最佳 tile 為準；三欄加總 = 100%）")
    combos = [("prior_L21-28q", s) for s in KEY_SIGS if s in recs[0]["scores"]]
    for prior_name, sig in combos:
        print(f"\n  stage-1 = {prior_name} , stage-2 = {sig}")
        print(f"    {'m':>4s} {'M(平均)':>8s} {'stage-1漏掉':>11s} {'stage-2選錯':>11s} {'選中':>7s}"
              f" {'選中greedy-oracle tile比例':>24s}")
        for m in CASC_MS + [None]:
            miss1, miss2, hit, gov, Ms = [], [], [], [], []
            for r in recs:
                o = r["oracle"]
                best = max(o, key=lambda t: o[t])
                prior = {int(k): v for k, v in r["priors"][prior_name].items()}
                score = {int(k): v for k, v in r["scores"][sig].items()}
                cand, sel = cascade_select(prior, score, slots, m)
                Ms.append(len(cand))
                miss1.append(best not in cand)
                miss2.append(best in cand and best not in sel)
                hit.append(best in sel)
                g = set(r["oracle_greedy"])
                if g:
                    gov.append(len(g & set(sel)) / len(g))
            lab = "all" if m is None else f"{m:g}"
            print(f"    {lab:>4s} {sum(Ms) / len(Ms):8.1f} {pct(miss1):10.1f}% {pct(miss2):10.1f}% "
                  f"{pct(hit):6.1f}% {pct(gov):23.1f}%")

    # ---- 4. 總表：每塊 tile 打分成本 vs 選中率（stage-1 = prior_L21-28q）----
    print("\n[4] 打分成本 vs 選中率（oracle 最佳 tile 被選中的比例；stage-1 = prior_L21-28q）")
    hdr = "".join(f"{('m=' + format(m, 'g')) if m else 'all':>8s}" for m in CASC_MS[1:] + [None])
    print(f"  {'訊號':22s} {'ms/tile':>8s} {'vs ctx_kv_L20':>13s}{hdr}")
    ms_of = {}
    for sig in KEY_SIGS:
        ms = [r.get("sig_ms_per_tile", {}).get(sig) for r in recs][1:]
        ms = [x for x in ms if x is not None]
        ms_of[sig] = sum(ms) / len(ms) if ms else None
    base_ms = ms_of.get("qattn_ctx_kv_L20")
    for sig in KEY_SIGS:
        if sig not in recs[0]["scores"]:
            continue
        cells = ""
        for m in CASC_MS[1:] + [None]:
            hit = []
            for r in recs:
                o = r["oracle"]
                best = max(o, key=lambda t: o[t])
                _, sel = cascade_select(r["priors"]["prior_L21-28q"], r["scores"][sig], slots, m)
                hit.append(best in sel)
            cells += f"{pct(hit):7.1f}%"
        ms = ms_of[sig]
        rel = f"{ms / base_ms:12.2f}x" if ms and base_ms else f"{'—':>13s}"
        print(f"  {sig:22s} {ms if ms else float('nan'):8.1f} {rel}{cells}")
    print("  （ms/tile 只有走前綴 KV 路徑的訊號有量；qattn_L12 / qattn_ctx_L20 是多層一次算的診斷路徑）")

    # ---- 5. 一致性：qattn_ctx_L20（整段 forward）vs qattn_ctx_kv_L20（前綴 KV 重用）----
    if "qattn_ctx_kv_L20" in recs[0]["scores"]:
        same = []
        for r in recs:
            a = _rank_desc(r["scores"]["qattn_ctx_L20"])[:slots]
            b = _rank_desc(r["scores"]["qattn_ctx_kv_L20"])[:slots]
            same.append(set(a) == set(b))
        print(f"\n[5] 一致性：整段 forward 與前綴 KV 重用（L20）的前 {slots} 名完全相同 {pct(same):.1f}%")

    # ---- 3. 一致性檢查：重算的 cascade vs out_v2 實際選擇 ----
    if v2_json and os.path.exists(v2_json):
        v2 = json.load(open(v2_json, encoding="utf-8"))
        print("\n[3] 一致性檢查：重算的 cascade（prior L21-28q + q_attn L12）vs out_v2 實際選的 tile")
        for m in [2, 3, 4, 5]:
            tag = f"budget={out['budget']}_info_density_delay0_tp21-28q_casc{m}_qa12"
            ts = v2.get("tile_selection", {}).get(tag)
            if not ts:
                continue
            same = []
            for r in recs:
                thumb = r["num_tiles"] - 1
                prior = {int(k): v for k, v in r["priors"]["prior_L21-28q"].items()}
                score = {int(k): v for k, v in r["scores"]["qattn_L12"].items()}
                _, sel = cascade_select(prior, score, slots, m)
                same.append(set(sel) == set(ts[r["idx"]]) - {thumb})
            print(f"  casc_m={m}: 完全一致 {pct(same):.1f}%")

    sec = [r["sec"] for r in recs]
    print(f"\n（每張圖收集分數平均 {sum(sec) / len(sec):.2f}s；含兩種 prior + 全部 tile 的兩種 q_attn，"
          f"非 pipeline TTFT）")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--num_images", type=int, default=200)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--max_num", type=int, default=48, help="同 internvl_stream.py --max_num（含縮圖）")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--oracle_json", default="out_oracle_sel_infovqa_b1024.json")
    ap.add_argument("--v2_json", default="out_v2_InternVL3_5-8B_infovqa_b1024.json")
    ap.add_argument("--output_json", default="out_diag_cascade_v2_infovqa_b1024.json")
    ap.add_argument("--report_only", action="store_true")
    args = ap.parse_args()

    if args.report_only:
        out = json.load(open(args.output_json, encoding="utf-8"))
    else:
        out = collect(args)
    report(out, args.v2_json)


if __name__ == "__main__":
    main()
