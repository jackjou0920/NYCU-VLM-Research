"""
正式跑實驗前的檢查：q_attn「重用前綴 KV」實作是否正確、每塊 tile 的打分成本多少。

對前 N 張 InfoVQA 圖，每張的所有 grid tile 用三種方式打分：
  old      : qattn_scores（L12，context = [text_before][tile][text_after]）← 舊 cascade
  ctx_full : 整段 [text_before][thumbnail][tile][text_after] 直接 forward 到 L20
             （= diag_cascade_stages.py 的 qattn_ctx_L20，參考答案）
  ctx_kv   : qattn_scores_with_prefix（前綴 KV 取自 thumbnail prior 那次 forward）← 新 cascade

正確性：ctx_kv vs ctx_full 的相對誤差應在 bf16 雜訊內（~1e-2 以下），前 K-1 名應幾乎全同。
成本：每個 ViT micro-batch（4 塊 tile）各方式的平均秒數（不含 ViT），另列 prior forward
      多存 KV 的額外時間。

用法：
    python check_qattn_prefix.py --num_images 5
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--num_images", type=int, default=5)
    ap.add_argument("--budget", type=int, default=1024)
    ap.add_argument("--max_num", type=int, default=48)
    ap.add_argument("--vit_batch", type=int, default=4)
    args = ap.parse_args()

    import torch
    import internvl_stream as S
    from process_common import load_infovqa

    dtype = torch.bfloat16
    dev = S.DEVICE
    tokenizer, model = S.build_model(args.model_name, dtype=dtype, device=dev)
    adapter = S.InternVLAdapter(tokenizer, model, budget=args.budget)
    n_tok = model.num_image_token
    K = args.budget // n_tok
    L = S.QATTN_LAYER
    emb = adapter.get_input_embeddings(model)
    datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                            num_image=args.num_images, prompt_mode=S.INFOVQA_PROMPT)

    def timed(fn):
        torch.cuda.synchronize(dev)
        t0 = time.time()
        out = fn()
        torch.cuda.synchronize(dev)
        return out, time.time() - t0

    t_old, t_full, t_kv, t_prior0, t_prior1 = [], [], [], [], []
    max_rel, top_same, n_img = 0.0, 0, 0
    for i, d in enumerate(datasets):
        pv = S.load_image_tiles(d["image"], input_size=448, max_num=max(1, args.max_num - 1))
        nt = pv.shape[0]
        if nt < 3:
            continue
        W, H = d["image"].size
        cols, rows = S._infer_tile_grid(nt - 1, W, H)
        prompt = S.build_prompt(tokenizer, model, d["question"], max(1, -(-args.budget // n_tok)))
        tb, ta = S.split_prompt_at_vision(prompt, tokenizer)
        emb_b = emb(tokenizer([tb], return_tensors="pt").input_ids.to(dev))
        emb_a = emb(tokenizer([ta], return_tensors="pt").input_ids.to(dev))

        # prior forward：不存 KV vs 存 KV（量多存 KV 的額外成本）
        _, t = timed(lambda: adapter.thumbnail_saliency_prior(model, pv, tb, ta, cols, rows, dtype))
        t_prior0.append(t)
        _, t = timed(lambda: adapter.thumbnail_saliency_prior(model, pv, tb, ta, cols, rows, dtype,
                                                              capture_kv_upto=L))
        t_prior1.append(t)
        thumb, prefix_kv = adapter._thumb_tokens, adapter._thumb_prefix_kv
        P = prefix_kv["len"]
        assert P == emb_b.shape[1] + n_tok, (P, emb_b.shape[1])

        s_old, s_full, s_kv = {}, {}, {}
        grid = list(range(nt - 1))
        for s in range(0, len(grid), args.vit_batch):
            idxs = grid[s: s + args.vit_batch]
            toks = S.encode_tile(model, pv[idxs].to(dev, dtype=dtype), dtype=dtype)
            B = toks.shape[0]
            v, t = timed(lambda: S.qattn_scores(model, toks, emb_b, emb_a, 12).tolist())
            t_old.append(t)
            s_old.update(zip(idxs, v))

            def full():
                h = torch.cat([emb_b.expand(B, -1, -1), thumb.unsqueeze(0).expand(B, -1, -1),
                               toks, emb_a.expand(B, -1, -1)], dim=1)
                r = S.prefix_attention_rows(model, h, P + n_tok, {L})[L]
                return r[..., P: P + n_tok].sum(-1).mean(-1).float().tolist()
            v, t = timed(full)
            t_full.append(t)
            s_full.update(zip(idxs, v))
            v, t = timed(lambda: S.qattn_scores_with_prefix(model, toks, prefix_kv, emb_a, L).tolist())
            t_kv.append(t)
            s_kv.update(zip(idxs, v))
            del toks

        rel = max(abs(s_kv[k] - s_full[k]) / (abs(s_full[k]) + 1e-6) for k in grid)
        top = lambda sc: set(sorted(sc, key=lambda k: (-sc[k], k))[: K - 1])
        same = top(s_kv) == top(s_full)
        max_rel = max(max_rel, rel)
        top_same += same
        n_img += 1
        print(f"[{i + 1}] tiles={nt}  max 相對誤差(ctx_kv vs ctx_full)={rel:.2e}  "
              f"前{K - 1}名相同={same}  old前{K - 1}={sorted(top(s_old))}  ctx前{K - 1}={sorted(top(s_kv))}")
        adapter._thumb_tokens = adapter._thumb_prefix_kv = None

    avg = lambda a: sum(a[1:]) / max(1, len(a) - 1)   # 略過第 1 筆 warmup
    print(f"\n正確性：{n_img} 張圖，最大相對誤差 {max_rel:.2e}，前 K-1 名完全相同 {top_same}/{n_img}")
    print(f"成本（每個 micro-batch = {args.vit_batch} 塊 tile，不含 ViT）：")
    print(f"  old      L12、無 context         {avg(t_old) * 1000:7.1f} ms")
    print(f"  ctx_full L{L}、整段重算          {avg(t_full) * 1000:7.1f} ms")
    print(f"  ctx_kv   L{L}、重用前綴 KV       {avg(t_kv) * 1000:7.1f} ms   ← 新 cascade 用這個")
    print(f"  prior forward：不存 KV {avg(t_prior0) * 1000:.1f} ms → 存 KV {avg(t_prior1) * 1000:.1f} ms")

    # bf16 下兩條路徑的數值不會完全相同；相對誤差超過 5% 或前 K-1 名有 2 張以上不同就當成實作有問題
    ok = n_img > 0 and max_rel < 0.05 and (n_img - top_same) <= 1
    print("\n[PASS] 前綴 KV 重用結果與整段 forward 一致" if ok else
          "\n[FAIL] 前綴 KV 重用結果與整段 forward 不一致，先別跑正式實驗")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
