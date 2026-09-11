# ══════════════════════════════════════════════════════════════════════════════
# Streaming Memory Bank  (固定 budget)
#   + Importance-aware Eviction
#   + Progressive Merge
#   + Online KV Construction
#
# 【重構說明】原本這支檔案裡的 run_online_kv_with_memory_bank() 已經拆成兩塊：
#   - 跟模型無關的骨架（text_before prefill / chunked vision injection /
#     text_after + decode）搬到 streaming_common.py，跟 LLaVA-OV 共用。
#   - 模型專屬的「tile encode → memory bank」搬到 internvl_adapter.py 的
#     InternVLAdapter.encode_and_bank()，內部邏輯（逐 tile encode_tile →
#     add_tile，tile 之間彼此獨立、不需等其他 tile）完全沒有改變。
# build_prompt / split_prompt_at_vision / encode_tile / generate_answer_standard
# 這四個函式維持原樣留在這裡，因為 internvl_adapter.py 直接 import 它們，
# 沒有重寫任何內部數學或順序。
# ══════════════════════════════════════════════════════════════════════════════
import glob
import os
import gc
import time
import argparse
import json
import torch
import numpy as np
from internvl_preprocess import build_model, load_image_tiles
from internvl_core import MAX_NEW_TOKENS, measure_baseline_ttft
from internvl_adapter import InternVLAdapter
from stream_adapters import DEVICE, measure_peak_memory
from stream_common import run_online_kv_with_memory_bank


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def save_results_incremental(output_path: str, results: dict):
    """每處理完一張圖就整份重寫一次 JSON。圖片數量通常不多（十幾二十張），
    整份重寫的成本可忽略，但換來的是「跑到一半 OOM 也不會流失前面的結果」。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num",     type=int, default=48,   help="max dynamic tiles")
    parser.add_argument("--vit_batch",   type=int, default=4,    help="ViT micro-batch size")
    parser.add_argument("--chunk_size",  type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--batch_size",  type=int, default=1,    help="Image batch size")
    parser.add_argument("--budget",      type=int, default=1024, help="The maximum number of vision tokens per image")
    parser.add_argument("--score_fn",    type=str, default="info_density", choices=["l2_norm", "info_density", "random"])
    parser.add_argument(
        "--merge_mode", type=str, default="evict",
        choices=["fifo", "evict", "merge", "merge_spatial"],
        help="fifo/evict = 硬淘汰；merge / merge_spatial = 用 evict 的同一組 Top-K 選"
             "存活 tile，再把被淘汰 tile 的 token 折進存活 token。"
             "merge = 全域 cosine 最近（會跨 tile 散開）；"
             "merge_spatial = 只折進 raster 距離最近的那一個存活 tile（空間約束）。",
    )
    parser.add_argument(
        "--merge_weight", type=str, default="score", choices=["score", "uniform"],
        help="merge_mode=merge 專用：折合被淘汰 token 時的加權方式。"
             "score=用 token importance 加權；uniform=等權（score_fn=random 時用這個）。",
    )
    parser.add_argument(
        "--select", type=str, default="topk", choices=["topk", "spatial_bin"],
        help="存活 tile 的選擇策略（跟 merge_mode 正交）。topk=純看分數；"
             "spatial_bin=依空間順序分箱、每箱取最高分，存活 tile 保證鋪滿全圖。"
             "配 merge_spatial 時每個被淘汰 tile 才真的有近鄰可折。",
    )
    parser.add_argument(
        "--tile_agg", type=str, default="mean",
        choices=["mean", "max", "topk_mean", "mean_std", "quantile"],
        help="how a tile's per-token scores are aggregated into one tile score "
             "before eviction ranking. 'mean' = current baseline; the others are "
             "high-tail-sensitive variants for the paper's ablation figure.",
    )
    parser.add_argument("--run_stream", action="store_true", help="run online KV pipeline")
    parser.add_argument("--run_standard", action="store_true",
                            help="run the official (uncompressed) HF generate() path to produce reference answers")
     
    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_internvl.json")
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")
     
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()
 
    # ── 1. 載入模型 ──
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"downsample_ratio         : {model.downsample_ratio}")
    print(f"select_layer             : {model.select_layer}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
 
    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    adapter = InternVLAdapter(tokenizer, model, budget=args.budget)

    # ── 2. 載入圖像與問題 ──
    if args.use_ds:
        from process_common import load_hf_dataset, load_mmmu
        # MMMU：選擇題，prompt 會帶選項；每筆會多存 answer/options/question_type 進 meta
        datasets = load_mmmu(split="validation", num_image=args.num_images)
        # datasets = load_hf_dataset(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation", num_image=args.num_images)
        # datasets = load_hf_dataset(dataset="lmms-lab-encoder/MMVet", subject=None, num_image=args.num_images)
    else:
        from process_common import load_local
        datasets = load_local(args.image, args.batch_size, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process (batch_size={args.batch_size})")

    # ── 3. 讀取舊的 output_results.json（如果存在），支援中斷後繼續跑 ──
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        if "references" not in output_results: output_results["references"] = []
        if "candidates" not in output_results: output_results["candidates"] = {}
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', {}))} images already done)")
    else:
        output_results = {"references": [], "candidates": {}}

    # ── 2b. MMMU：把每題的 answer/options/question_type 存進 JSON 的 meta，
    #        對齊 candidates[tag][i] / references[i]，讓 evaluate_mmmu.py 直接讀、
    #        不用重載 dataset（免對齊風險）。非 MMMU（load_local / DocVQA）沒這些欄位就跳過。
    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k))}
            for k, d in enumerate(datasets)
        ]

    tag = f"budget={args.budget}_{args.merge_mode}_{args.score_fn}"
    if args.tile_agg != "mean":
        # 只有非預設聚合方式才進 tag，維持舊的 baseline 結果檔（tile_agg=mean）不用重跑就能被 resume 認得。
        tag += f"_{args.tile_agg}"
    if args.merge_mode in ("merge", "merge_spatial"):
        # merge_weight 進 tag，避免不同 merge 設定的結果互相覆蓋。
        tag += f"_w{args.merge_weight}"
    if args.select != "topk":
        # 非預設選擇策略才進 tag，維持舊結果檔（select=topk）能被 resume 認得。
        tag += f"_{args.select}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        print(f"\nAll done. Total time: {elapsed:.2f} s")
        return

    # ── 4. 逐張（或依 batch_size 分小批）處理，避免一次把全部圖片攤在同個 batch ──
    for i in range(0, len(datasets), args.batch_size):
        batch_datasets = datasets[i:i + args.batch_size]
 
        print(f"\n{'='*70}")
        print(f"[{i+1}~{i+len(batch_datasets)}/{len(datasets)}] Processing Batch...")
        print(f"{'='*70}")

        # 只在真正要處理這一小批時才做影像前處理（tile 化），處理完就可以釋放
        pixel_values_list = [
            load_image_tiles(d["image"], input_size=448, max_num=args.max_num) for d in batch_datasets
        ]
        questions = [d["question"] for d in batch_datasets]
        tile_counts = [pv.shape[0] for pv in pixel_values_list]  # 每張圖的 raw tile 數
        print(f"  tiles per image = {tile_counts}")

        # print(f"DocVQA tile 數分佈: mean={np.mean(tile_counts):.1f}, "
        #     f"median={np.median(tile_counts):.0f}, max={max(tile_counts)}")
        # print(f"budget={args.budget} 能保留的比例: {3/np.array(tile_counts)}")  # 3 = capacity_tiles - protected
 
        torch.cuda.reset_peak_memory_stats(DEVICE)

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("internvl_standard_generate"):
                    ref_answers = adapter.generate_baseline(model, pixel_values_list=pixel_values_list, questions=questions)
                    output_results["references"] += ref_answers

                # baseline_ttft = measure_baseline_ttft(model, tokenizer, pixel_values_list, questions, device=DEVICE)
                
                # print("\n[Baseline Answer]")
                # # for i, answer in enumerate(ref_answers):
                # #     print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")
                # print(f"  Baseline TTFT = {baseline_ttft:.3f} sec")

            if args.run_stream:
                if tag not in output_results["candidates"]: output_results["candidates"][tag] = []

                print(f"\n[Online Stream] Running online kv stream with memory bank ...")
                with measure_peak_memory("internvl_online_kv_memory_bank"):
                    answers, all_stats, timing = run_online_kv_with_memory_bank(
                        model, adapter, pixel_values_list, questions,
                        image_sizes_list=None,   # InternVL 不需要（沒有 pack_image_features 那種全圖依賴）
                        dtype=dtype,
                        vit_batch=args.vit_batch,
                        chunk_size=args.chunk_size,
                        budget=args.budget,
                        merge_mode=args.merge_mode,
                        score_fn=args.score_fn,
                        tile_agg=args.tile_agg,
                        merge_weight=args.merge_weight,
                        select=args.select,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    output_results["candidates"][tag] += answers
                    # print(all_stats)

                print("\n[Online KV Answer]")
                # for i, answer in enumerate(answers):
                #     print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")

                for item, tt in timing.items():
                    if item.startswith("_"):   # e.g. _peak_mem_gb（記憶體 dict，不是秒數）
                        continue
                    print(f"  {item}: {tt} sec")
                     
            del pixel_values_list, questions

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on batch: {e}")
            raise

        gc.collect()
        torch.cuda.empty_cache()
         
        if args.save:
            save_results_incremental(args.output_json, output_results)
            print(f"\n[Saved] {args.output_json} updated "
                  f"({len(output_results['references'])}/{len(datasets)} images done)")

    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"\nAll done. Total time: {elapsed:.2f} s")

    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
