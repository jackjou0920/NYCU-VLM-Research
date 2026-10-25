# ══════════════════════════════════════════════════════════════════════════════
# Streaming Memory Bank on LLaVA-OneVision-Qwen2  (grid-aware 版)
#   port 自 internvl_svm.py 的 run_online_kv_with_memory_bank()
#
# ── 跟 InternVL 版本的關鍵架構差異 ──────────────────────────────────────────
#
# 1. Tile 的定義：
#      InternVL : 每張圖動態切成 N 個 448x448 tile + 1 個 thumbnail（放最後），
#                 每個 tile 固定輸出 256 個 token。
#      LLaVA-OV : anyres 前處理把每張圖切成 N 個 "patch"：patch[0] 是整圖縮圖
#                 （base image，放最前面），patch[1:] 是 anyres 網格裁出的
#                 crop，依 row-major 排列。每個 patch 固定輸出
#                 (image_size // patch_size)**2 個 token（7b-ov-hf 預設是
#                 SigLIP 384px / patch14 → 27*27 = 729 token/patch）。
#      → 把每個 "patch" 當作 TileStreamingMemoryBank 的一個 "tile"，
#        num_image_token 從 256 換成 729，protect_position 用 "first"
#        （base image 永遠保留，對應 InternVL 保護 thumbnail 的精神）。
#
# 2. 官方 `vision_aspect_ratio="anyres_max_9"` 這個設定，代表模型從來沒有
#    在訓練時看過「超過 9 個 crop 直接攤平串接」這種輸入型態——crop 數只要超過
#    9，官方一定會用雙線性內插把整個網格強制壓回 ~6561 token 的固定上限，這個
#   「連續內插、每個角落都保留一點模糊資訊」的表示，跟「離散選幾個完整 crop、
#    其他角落完全空白」在分佈上是兩回事，即使 token 數接近，模型也認不得。
#
# 所以這版改弦易轍：**不再自己重刻 unpad/interpolate，直接呼叫官方
# `model.model.pack_image_features()`**，拿到跟官方 generate() 一模一樣、
# 保證 in-distribution 的 packed 序列（本來就會處理 unpad + 內插 + newline，
# 不管 crop 數多寡都在模型訓練時看過的分佈內）。
#
# Memory bank 的角色因此往後退一步：不再負責「決定哪些 crop 存活」，而是在
# 官方 pack 完、已經是正確表示之後，**以「攤平後的一列（含結尾的 newline，
# 長度 = curr_width+1）」為 tile 單位**，用 TileStreamingMemoryBank 做進一步
# 壓縮（只有 budget < 官方 pack 出來的長度時才需要）。這個「一列 = 一個
# tile」的抽象，跟 InternVL 的「一個 448x448 tile = 一個 tile」是同一個類別
# （TileStreamingMemoryBank）的兩種不同實例化，不需要為 LLaVA-OV 另外寫一個
# eviction 演算法——streaming eviction 這件事跟「tile 的內容從哪裡來」是解耦的。
#
# 跟官方 100% 對齊的部分：unpad、interpolate、newline 位置——全部直接呼叫官方
# 函式，不是照抄數學重寫。
# 跟官方不一樣、屬於我們自己壓縮策略的部分：budget < pack 出來的長度時，用
# score-based eviction 再砍掉一些「列」，這步官方沒有，是我們疊加上去、拿來
# 驗證 memory bank 本身有沒有用的部分。
#
# 【重構說明】原本這支檔案裡的 run_online_kv_with_memory_bank() 已經拆成兩塊：
#   - 跟模型無關的骨架（text_before prefill / chunked vision injection /
#     text_after + decode）搬到 streaming_common.py，跟 InternVL 共用。
#   - 模型專屬的「per-patch SigLIP → 官方 pack_image_features → (可選) 列淘汰」
#     搬到 llava_ov_adapter.py 的 LlavaOVAdapter.encode_and_bank()，內部邏輯
#     逐行照搬，完全沒有改變。
# clean_answer / generate_answer_standard / build_prompt / split_prompt_at_vision /
# encode_patch / num_image_tokens_per_patch / compute_packed_row_layout 這幾個
# 函式維持原樣搬到 llava_ov_core.py，因為 llava_ov_adapter.py 直接 import
# 它們（放在這裡會跟 llava_ov_adapter.py 形成循環 import，原因見
# llava_ov_core.py 開頭說明）。
# ══════════════════════════════════════════════════════════════════════════════

import os
import gc
import time
import argparse
import json
import torch
from exp.llava_ov_preprocess import build_model, load_image_patches
from llava_ov_core import MAX_NEW_TOKENS, num_image_tokens_per_patch
from llava_ov_adapter import LlavaOVAdapter
from stream_adapters import DEVICE, measure_peak_memory
from stream_common import run_online_kv_with_memory_bank


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def save_results_incremental(output_path: str, results: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--vit_batch", type=int, default=4, help="ViT micro-batch size")
    parser.add_argument("--chunk_size", type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--batch_size", type=int, default=1, help="Image batch size")
    parser.add_argument("--budget", type=int, default=1024, help="The maximum number of vision tokens per image")
    parser.add_argument(
        "--score_fn", type=str, default="info_density",
        choices=["l2_norm", "info_density", "random"],
    )
    parser.add_argument(
        "--merge_mode", type=str, default="evict",
        choices=["fifo", "evict"],
    )
    parser.add_argument("--run_stream", action="store_true", help="run online KV pipeline")
    parser.add_argument("--run_standard", action="store_true",
                            help="run the official (uncompressed) HF generate() path to produce reference answers")
 
    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_llava_ov.json")
    args = parser.parse_args()
 
    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")
 
    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    # ── 1. 載入模型 ──
    processor, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    print(f"num_image_token per patch : {num_image_tokens_per_patch(model)}")
    print(f"vision_feature_layer      : {model.config.vision_feature_layer}")
    print(f"vision_feature_select     : {model.config.vision_feature_select_strategy}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")
 
    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    adapter = LlavaOVAdapter(processor)

    # ── 2. 載入圖像與問題 ──
    if args.use_ds:
        from process_common import load_hf_dataset
        # datasets = load_hf_dataset(dataset="MMMU/MMMU", subject="Agriculture", split="test", num_image=args.num_images)
        # datasets = load_hf_dataset(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", num_image=args.num_images)
        datasets = load_hf_dataset(dataset="lmms-lab-encoder/MMVet", subject=None, num_image=args.num_images)
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

    tag = f"budget={args.budget}_{args.merge_mode}_{args.score_fn}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        print(f"\nAll done. Total time: {elapsed:.2f} s")
        return

    # ── 4. 逐批處理 ──
    for i in range(0, len(datasets), args.batch_size):
        batch_datasets = datasets[i:i + args.batch_size]
 
        print(f"\n{'='*70}")
        print(f"[{i+1}~{i+len(batch_datasets)}/{len(datasets)}] Processing Batch ...")
        print(f"{'='*70}")
 
        torch.cuda.reset_peak_memory_stats(DEVICE)

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                print(f"\n[Standard] Running official (uncompressed) generate() path ...")
                with measure_peak_memory("llava_ov_standard_generate"):
                    ref_answers = adapter.generate_baseline(model, batch_datasets)
                    output_results["references"] += ref_answers

                # print("\n[Baseline Answer]")
                # for i, answer in enumerate(ref_answers):
                #     print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")

            if args.run_stream:
                if tag not in output_results["candidates"]: output_results["candidates"][tag] = []

                print(f"\n[Online Stream] Running online kv stream with memory bank ...")
                questions = [dataset["question"] for dataset in batch_datasets]

                pixel_values_list, image_sizes_list = load_image_patches(processor, batch_datasets)
                print(f"  patches per image = {[pv.shape[0] for pv in pixel_values_list]}")

                with measure_peak_memory("llava_ov_online_kv_memory_bank"):
                    answers, all_stats = run_online_kv_with_memory_bank(
                        model, adapter, pixel_values_list, questions,
                        image_sizes_list=image_sizes_list,
                        dtype=dtype,
                        vit_batch=args.vit_batch,
                        chunk_size=args.chunk_size,
                        budget=args.budget,
                        merge_mode=args.merge_mode,
                        score_fn=args.score_fn,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    output_results["candidates"][tag] += answers

                # print("\n[Online KV Answer]")
                # for i, answer in enumerate(answers):
                #     print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")
 
                del pixel_values_list, image_sizes_list
    
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
