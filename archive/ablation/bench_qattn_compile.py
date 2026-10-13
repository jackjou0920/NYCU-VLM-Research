"""
q_attn 打分 torch.compile 三種模式的比較：速度、記憶體、結果是否一致。

模式（internvl_stream.QATTN_COMPILE）：
  none            : 不編譯（現行）
  default         : torch.compile，只做 kernel fusion
  reduce-overhead : torch.compile + CUDA Graph

對前 N 張 InfoVQA 圖，每張圖的 ViT 只跑一次，同一批 tile token 輪流給三種模式打分，量：
  打分時間      : 每個 micro-batch（VIT_BATCH 塊 tile）的平均毫秒數（略過第 1 張圖）
  暫時記憶體    : 打分期間 max_memory_allocated 比打分前多出多少（每張圖取最大）
  常駐記憶體    : 預編譯（warmup）前後 memory_allocated / memory_reserved 的差，
                  也就是這個模式之後會一直佔著的量（CUDA Graph 的記憶體池在這裡）
  一致性        : 跟 none 的分數比，最大相對誤差、前 K-1 名完全相同的張數

用法：
    python bench_qattn_compile.py --num_images 20
    python bench_qattn_compile.py --num_images 20 --modes none,default
"""
from __future__ import annotations

import argparse
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--num_images", type=int, default=20)
    ap.add_argument("--budget", type=int, default=1024, help="只用來決定 K（比較前 K-1 名）")
    ap.add_argument("--max_num", type=int, default=48)
    ap.add_argument("--modes", default="none,default,reduce-overhead")
    args = ap.parse_args()
    modes = [m for m in args.modes.split(",") if m]
    if modes[0] != "none":
        modes = ["none"] + [m for m in modes if m != "none"]      # none 當一致性的基準

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
    MB = 1024 ** 2

    # ---- 預編譯，量每個模式的常駐記憶體（依序做，none 不需要）----
    resident = {"none": (0.0, 0.0)}
    for mode in modes:
        if mode == "none":
            continue
        torch.cuda.synchronize(dev)
        a0, r0 = torch.cuda.memory_allocated(dev), torch.cuda.memory_reserved(dev)
        S.warmup_qattn_compile(model, adapter, dtype=dtype, budget=args.budget, mode=mode)
        torch.cuda.synchronize(dev)
        resident[mode] = ((torch.cuda.memory_allocated(dev) - a0) / MB,
                          (torch.cuda.memory_reserved(dev) - r0) / MB)

    datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                            num_image=args.num_images, prompt_mode=S.INFOVQA_PROMPT)
    times = {m: [] for m in modes}
    transient = {m: 0.0 for m in modes}
    max_rel = {m: 0.0 for m in modes}
    top_same = {m: 0 for m in modes}
    n_img = 0
    for i, d in enumerate(datasets):
        pv = S.load_image_tiles(d["image"], input_size=448, max_num=max(1, args.max_num - 1))
        nt = pv.shape[0]
        if nt < 3:
            continue
        W, H = d["image"].size
        cols, rows = S._infer_tile_grid(nt - 1, W, H)
        prompt = S.build_prompt(tokenizer, model, d["question"], max(1, -(-args.budget // n_tok)))
        tb, ta = S.split_prompt_at_vision(prompt, tokenizer)
        emb_a = emb(tokenizer([ta], return_tensors="pt").input_ids.to(dev))
        adapter.thumbnail_saliency_prior(model, pv, tb, ta, cols, rows, dtype, capture_kv_upto=L)
        prefix_kv = adapter._thumb_prefix_kv

        grid = list(range(nt - 1))
        scores = {m: {} for m in modes}
        for s in range(0, len(grid), S.VIT_BATCH):
            idxs = grid[s: s + S.VIT_BATCH]
            toks = S.encode_tile(model, pv[idxs].to(dev, dtype=dtype), dtype=dtype)
            for mode in modes:
                torch.cuda.synchronize(dev)
                base = torch.cuda.memory_allocated(dev)
                torch.cuda.reset_peak_memory_stats(dev)
                t0 = time.time()
                v = S.qattn_scores_with_prefix(model, toks, prefix_kv, emb_a, L, compile_mode=mode).tolist()
                torch.cuda.synchronize(dev)
                if i > 0:
                    times[mode].append(time.time() - t0)
                transient[mode] = max(transient[mode], (torch.cuda.max_memory_allocated(dev) - base) / MB)
                scores[mode].update(zip(idxs, v))
            del toks

        top = lambda sc: set(sorted(sc, key=lambda k: (-sc[k], k))[: K - 1])
        ref = scores["none"]
        line = []
        for mode in modes[1:]:
            rel = max(abs(scores[mode][k] - ref[k]) / (abs(ref[k]) + 1e-6) for k in grid)
            same = top(scores[mode]) == top(ref)
            max_rel[mode] = max(max_rel[mode], rel)
            top_same[mode] += same
            line.append(f"{mode}: rel={rel:.1e} 前{K - 1}同={same}")
        n_img += 1
        print(f"[{i + 1}] tiles={nt}  " + "  ".join(line))
        adapter._thumb_tokens = adapter._thumb_prefix_kv = None

    print(f"\n{'mode':16s} {'打分 ms/批':>10s} {'加速':>6s} {'暫時記憶體 MB':>13s} "
          f"{'常駐 alloc MB':>13s} {'常駐 reserved MB':>16s} {'最大相對誤差':>12s} {'前K-1名相同':>11s}")
    t_none = sum(times["none"]) / max(1, len(times["none"]))
    for mode in modes:
        t = sum(times[mode]) / max(1, len(times[mode]))
        same = "—" if mode == "none" else f"{top_same[mode]}/{n_img}"
        rel = "—" if mode == "none" else f"{max_rel[mode]:.1e}"
        print(f"{mode:16s} {t * 1000:10.1f} {t_none / t:5.2f}x {transient[mode]:13.1f} "
              f"{resident[mode][0]:13.1f} {resident[mode][1]:16.1f} {rel:>12s} {same:>11s}")
    print(f"\n（每批 {S.VIT_BATCH} 塊 tile、L{L}、略過第 1 張圖的時間；"
          f"參考：b1024 KV cache 約 154 MB、整個 pipeline peak 約 17.4 GB）")


if __name__ == "__main__":
    main()
