# ══════════════════════════════════════════════════════════════════════════════
# Qwen3-VL-8B-Instruct 版 Streaming Memory Bank
#   ViT+merger 整張圖跑一次 → 依 raster 順序切 pseudo-tile → Importance-aware 選擇
#   → 壓縮到固定 budget → 帶 DeepStack 特徵一起灌進 LLM(單次 prefill,budget 已封頂長度)
#
# 跟 internvl_svm.py 的關鍵差異(務必先讀 qwen_memory_bank.py 開頭的說明):
#   1. 沒有 encode_tile() 的逐 tile 串流:Qwen3-VL ViT 需要完整 patch 網格做
#      2D-RoPE + attention,無法切開,所以 ViT 對每張圖只跑一次。
#      這裡壓縮的是「LLM prefill 長度 + KV cache」,不是 ViT 計算量。
#   2. 多了 DeepStack 特徵(3 層),要跟 main token 套用同一個 keep_idx。
#   3. MRoPE 位置:借用官方 model.model.get_rope_index() 對「未壓縮」序列算一次,
#      再用保留 token 的原始座標去索引取值,不用自己重寫 3D 位置編碼公式。
#   4. 因為 budget 把總長度封頂在合理範圍(~千級 token),不需要像 InternVL 版本
#      那樣做 chunked prefill,單次 forward 即可,程式碼因此簡化不少。
#   5. 為求正確性,batch_size 建議固定為 1(right-padding 下的批次生成位置對齊
#      是已知的簡化點,跟 internvl_svm.py 目前的處理方式一致)。
# ══════════════════════════════════════════════════════════════════════════════
import glob
import os
import gc
import time
import argparse
import json
import torch
from exp.qwen_preprocess import measure_peak_memory, build_model


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150


# ──────────────────────────────────────────────────────────────────────────────
# 標準路徑(用於正確性比較):完全依賴官方 processor + generate()
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer_standard(model, processor, images, questions):
    assert len(images) == len(questions)
    answers = []
    for image, question in zip(images, questions):
        messages = [{
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": question}],
        }]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        trimmed = generated_ids[0][inputs["input_ids"].shape[-1]:]
        answers.append(processor.decode(trimmed, skip_special_tokens=True))
    return answers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--min_pixels", type=int, default=None, help="native resolution 下限")
    parser.add_argument("--max_pixels", type=int, default=None, help="native resolution 上限(對應 InternVL 的 max_num)")
    parser.add_argument("--tokens_per_tile", type=int, default=64, help="pseudo-tile 大小")
    parser.add_argument("--batch_size", type=int, default=1, help="建議固定為 1(見檔頭說明)")
    parser.add_argument("--budget", type=int, default=1024, help="每張圖的 vision token 上限")
    parser.add_argument(
        "--score_fn", type=str, default="information_density",
        choices=["l2_norm", "info_density", "random"],
    )
    parser.add_argument("--merge_mode", type=str, default="evict", choices=["fifo", "evict"])
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg")
    parser.add_argument("--save", action="store_true")
    parser.add_argument("--output_json", type=str, default="qwen_output_results.json")
    parser.add_argument("--run_stream", action="store_true")
    args = parser.parse_args()
 
    question = "What is shown in this image in extreme detail?"
    dtype = torch.bfloat16
 
    torch.cuda.synchronize()
    t0 = time.time()
 
    tokenizer, processor, model = build_model(args.model_name, dtype=dtype)
    print(f"image_token_id            : {model.config.image_token_id}")
    print(f"spatial_merge_size        : {model.config.vision_config.spatial_merge_size}")
    print(f"deepstack_visual_indexes  : {model.config.vision_config.deepstack_visual_indexes}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    torch.cuda.synchronize()
    print(f"Load model time: {time.time() - t0:.2f} s")

    if os.path.isdir(args.image):
        image_paths = sorted(sum(
            [glob.glob(os.path.join(args.image, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []
        ))
        if not image_paths:
            raise FileNotFoundError(f"No images in {args.image}")
    else:
        image_paths = [args.image] * args.batch_size
    print(f"Found {len(image_paths)} image(s) to process (batch_size={args.batch_size})")

    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]
        print(f"\n{'='*70}\n[{i+1}~{i+len(batch_paths)}/{len(image_paths)}] Processing: {batch_paths}\n{'='*70}")

        # images = []
        # for p in batch_paths:
        #     _, _, img = load_image_patches(processor, p, args.min_pixels, args.max_pixels)
        #     images.append(img)
        # questions = [question] * len(batch_paths)

        # 每張圖（或每個小批次）開始前重置記憶體統計，量測才不會被前一批污染
        torch.cuda.reset_peak_memory_stats()

        try:
            pass
        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on batch {batch_paths}: {e}")
            raise


if __name__ == "__main__":
    main()
