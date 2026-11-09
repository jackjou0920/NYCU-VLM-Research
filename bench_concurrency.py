"""
實測一張 GPU 能同時常駐幾個請求（取代 build_paper_results.scaling_points() 的估計式）。

情境跟估計式一樣：請求一個接一個 prefill（max_new_tokens=1），每個請求做完後把它的 KV cache
留在 GPU 上（代表它還在 decode），一直送到 OOM 為止；成功的請求數 = 同時請求數。
最後一個請求 prefill 時的峰值也算在內，所以跟估計式 floor((free − peak) / kv) + 1 是同一個定義。

做法：不改任何方法的程式，只攔截 transformers 的 DynamicCache.__init__，記下每個請求建立的所有
cache，請求結束後只留「層數等於大模型層數、而且最大」的那個（= 拿去 decode 的那個；IF-Prune 小 VLM 的
cache 層數不同、FoveaTile 的 prior 不用 DynamicCache），其他的放掉。
  - 請求輪流用 HR-Bench 8K 前 --num_images 張圖（同 run_scaling.sh；第 0 張只 warmup，不留 cache）
  - IF-Prune 的小 VLM 整段常駐 GPU（實際部署也是這樣）
  - SparseVLM 的 λ 每張圖先 auto_tune_lam 一次（同 run_scaling.sh），校準時的 cache 不留
  - 量測前檢查 GPU 上沒有別人的程序佔記憶體（--max_foreign_gb），否則不跑
結果：一列 CSV（method, max_num, raw_tiles, n_requests, kv_gb_per_req, free_gb, ...）
用法：python bench_concurrency.py --method foveatile --max_num 201 --out_csv results/concurrency/n201_foveatile.csv
"""
import gc
import os
import csv
import time
import argparse
import statistics as st

import torch
from transformers.cache_utils import DynamicCache

from internvl_preprocess import build_model, load_image_tiles
import foveatile_internvl as S
import sparsevlm_internvl as SV
import fastv_internvl as FV
import visionzip_internvl as VZ

_created = []                       # 這個請求期間建立的 DynamicCache
_capture = [True]                   # SparseVLM 校準 λ 時關掉，校準的 cache 才不會被留住
_orig_init = DynamicCache.__init__


def _hooked_init(self, *a, **kw):
    _orig_init(self, *a, **kw)
    if _capture[0]:
        _created.append(self)


DynamicCache.__init__ = _hooked_init


def cache_bytes(c):
    n = 0
    for layer in getattr(c, "layers", []):
        for t in (getattr(layer, "keys", None), getattr(layer, "values", None)):
            if isinstance(t, torch.Tensor):
                n += t.numel() * t.element_size()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--dataset", default="hrbench_8k")
    ap.add_argument("--method", required=True, choices=["uncompressed", "foveatile", "fastv", "sparsevlm", "visionzip", "ifprune", "hired"])
    ap.add_argument("--max_num", type=int, default=48)
    ap.add_argument("--budget", type=int, default=1536)
    ap.add_argument("--num_images", type=int, default=11)
    ap.add_argument("--max_requests", type=int, default=400)
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--max_foreign_gb", type=float, default=1.0, help="GPU 上別人的程序最多可佔多少，超過就不跑")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()
    dtype, b = torch.bfloat16, args.budget

    free0, total = torch.cuda.mem_get_info(S.DEVICE)
    foreign = (total - free0) / 1e9
    if foreign > args.max_foreign_gb:
        raise SystemExit(f"GPU 上已有 {foreign:.2f} GB 被佔用（> {args.max_foreign_gb}），不量")

    tokenizer, model = build_model(args.model_name, dtype=dtype, device=S.DEVICE)
    n_layers = model.language_model.config.num_hidden_layers
    guide = None
    if args.method == "ifprune":
        import ifprune_internvl as IP
        guide = IP.load_ifprune_guide("ZachSun/internvl2_5_1b_pruning_kl_lora32", "OpenGVLab/InternVL2_5-1B",
                                      dtype=dtype, device=S.DEVICE, head_dtype=torch.bfloat16)
        guide.onload()
    torch.cuda.synchronize(S.DEVICE)
    weights_gb = torch.cuda.memory_allocated(S.DEVICE) / 1e9
    free_gb = torch.cuda.mem_get_info(S.DEVICE)[0] / 1e9      # 權重（含 IF-Prune 小 VLM）載完後剩下的
    print(f"model={args.model_name} method={args.method} max_num={args.max_num} B={b}  "
          f"weights={weights_gb:.2f} GB  free={free_gb:.2f} GB")

    ds = S.load_eval_dataset(args.dataset, args.num_images)
    reqs = []
    for d in ds:
        img = S.get_image(d)
        req = {"q": d["question"], "img": img,
               "pv": load_image_tiles(img, input_size=448, max_num=max(1, args.max_num - 1))}
        req["tiles"] = req["pv"].shape[0]
        if args.method == "hired":
            from internvl_preprocess import dynamic_grid
            req["grid"] = dynamic_grid(img, 1, max(1, args.max_num - 1), 448)
        reqs.append(req)
    if args.method == "sparsevlm":
        _capture[0] = False
        for r in reqs:
            lam, _ = SV.auto_tune_lam(model, tokenizer, r["pv"], r["q"], budget=b, vit_batch=args.vit_batch)
            r["lam"] = SV.budget_lam(r["tiles"] * model.num_image_token, b, lam)
            gc.collect()
            torch.cuda.empty_cache()
        _capture[0] = True
    m_cov = S.default_coverage_tiles(b)
    s_q = S.default_qattn_tiles(b, m_cov)

    def run(r):
        if args.method == "uncompressed":
            S.generate_answer_standard(model, tokenizer, [r["pv"]], [r["q"]],
                                       generation_config=dict(max_new_tokens=1, do_sample=False))
        elif args.method == "foveatile":
            ms = S.prepare_inputs([r["img"]], "foveatile", b, args.max_num, coverage_tiles=m_cov)[0]
            adapter = S.InternVLAdapter(tokenizer, model, budget=b)
            S.run_online_kv_with_memory_bank(model, adapter, [ms], [r["q"]], budget=b, qattn_tiles=s_q,
                                             max_new_tokens=1, dtype=dtype)
        elif args.method == "fastv":
            FV.generate_fastv(model, tokenizer, r["pv"], r["q"], budget=b, agg_layer=2,
                              vit_batch=args.vit_batch, max_new_tokens=1)
        elif args.method == "sparsevlm":
            SV.generate_sparsevlm(model, tokenizer, r["pv"], r["q"], budget=b, lam=r["lam"],
                                  vit_batch=args.vit_batch, max_new_tokens=1)
        elif args.method == "visionzip":
            VZ.generate_visionzip(model, tokenizer, r["pv"], r["q"], budget=b, contextual_ratio=VZ.CONTEXTUAL_RATIO,
                                  vit_batch=args.vit_batch, max_new_tokens=1)
        elif args.method == "hired":
            import hired_internvl as HD
            HD.generate_hired(model, tokenizer, r["pv"], r["grid"], r["q"], b, vit_batch=args.vit_batch, max_new_tokens=1)
        else:
            IP.generate_ifprune(model, tokenizer, r["pv"], r["q"], guide, budget=b, prune_layer=2,
                                vit_batch=args.vit_batch, max_new_tokens=1)

    def take_cache():
        """這個請求 decode 用的 cache（大模型層數、最大的那個）；其他的放掉。"""
        cands = [c for c in _created if len(getattr(c, "layers", [])) == n_layers]
        _created.clear()
        return max(cands, key=cache_bytes) if cands else None

    run(reqs[0])                      # warmup，不留
    take_cache()
    gc.collect()
    torch.cuda.empty_cache()

    resident, kv_gb, tiles, peak_gb, oom = [], [], [], [], False
    t0 = time.time()
    while len(resident) < args.max_requests:
        r = reqs[1 + len(resident) % (len(reqs) - 1)]
        torch.cuda.reset_peak_memory_stats(S.DEVICE)
        try:
            run(r)
        except torch.OutOfMemoryError:
            oom = True
            break
        c = take_cache()
        if c is None:
            raise RuntimeError("找不到這個請求的 KV cache")
        resident.append(c)
        kv_gb.append(cache_bytes(c) / 1e9)
        tiles.append(r["tiles"])
        peak_gb.append(torch.cuda.max_memory_allocated(S.DEVICE) / 1e9)
        gc.collect()
        if len(resident) % 10 == 0:
            print(f"  {len(resident)} requests resident, allocated={torch.cuda.memory_allocated(S.DEVICE) / 1e9:.2f} GB")
    n = len(resident)
    _created.clear()
    print(f"[{args.method} max_num={args.max_num}] n_requests={n} ({'OOM' if oom else 'hit max_requests'}), "
          f"kv/req={st.mean(kv_gb) if kv_gb else 0:.3f} GB, {time.time() - t0:.0f}s")

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    row = {"model": args.model_name, "dataset": args.dataset, "method": args.method, "max_num": args.max_num,
           "budget": b, "raw_tiles_mean": st.mean(tiles) if tiles else None, "n_requests": n, "stopped_by_oom": oom,
           "kv_gb_per_req_mean": st.mean(kv_gb) if kv_gb else None, "weights_gb": weights_gb, "free_gb": free_gb,
           "first_req_peak_gb": peak_gb[0] if peak_gb else None, "foreign_gb_at_start": foreign}
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row))
        w.writeheader()
        w.writerow(row)
    print(f"[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
