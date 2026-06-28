"""
InternVL3.5-8B Online KV Construction Pipeline
================================================
概念與原始 LlavaOnevision 版本完全相同：
  Step 1 : 影像前處理 (Dynamic High-Res tiles)
  Step 2 : ViT 微批次串流 (分批 encode，避免 OOM)
  Step 3 : pixel_shuffle + MLP Projector (extract_feature)
  Step 4 : LLM Chunked KV 增量建構 (真正零峰值 Prefill)
  Step 5 : 文字後半 Prefill + 自迴歸解碼

InternVL3.5 架構差異摘要（對比 LlavaOnevision）：
  - ViT : InternViT-300M，tile 大小 448×448，每 tile → 1024 個 patch token
  - Projector : pixel_shuffle (downsample_ratio=0.5) + MLP (mlp1)
                每 tile 最終壓縮為 256 個 LLM token
  - LLM : Qwen2ForCausalLM (InternVL3.5-8B 使用 Qwen3-8B-Base)
  - 模型透過 trust_remote_code=True 從 HF 載入
  - 圖像 token 佔位符為 <IMG_CONTEXT>，以 <img>...</img> 包圍
  - 沒有 LlavaOnevision 的 pack_image_features / unpad；
    各 tile 的 vision token 直接串接後注入 LLM
"""

import time
import argparse
import warnings
import torch
import numpy as np
import torch.cuda.nvtx as nvtx
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, PreTrainedModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150

# ──────────────────────────────────────────────────────────────────────────────
# 影像前處理（動態高解析度，複製自官方 HF model card 範例）
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    best_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width  = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image_tiles(image_path, input_size=448, max_num=12):
    """回傳 pixel_values [N_tiles, 3, H, W] 以及 tile 數量 num_patches"""
    image = Image.open(image_path).convert("RGB")
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values, len(tiles)


# ──────────────────────────────────────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    dtype: torch.dtype = torch.bfloat16,
):
    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )

    print(f"Loading model from {model_name} ...")
    device_map = Accelerator().device
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=True,  # 關閉 Flash Attention 以降低記憶體碎片（可改 True 加速）
        low_cpu_mem_usage=True,
        device_map=device_map
    ).to(DEVICE).eval()

    return tokenizer, model


# ──────────────────────────────────────────────────────────────────────────────
# 工具函式
# ──────────────────────────────────────────────────────────────────────────────

IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

# ──────────────────────────────────────────────────────────────────────────────
# 標準 generate 參考路徑（用於正確性比較）
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer_standard(model, tokenizer, pixel_values: torch.Tensor, question: str):
    """走官方 model.chat() 路徑，作為 Online KV 的比對 baseline。"""
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    with torch.no_grad():
        response, history = model.chat(
            tokenizer,
            pixel_values.to(DEVICE, dtype=model.dtype),
            question,
            generation_config,
            return_history=True,
        )
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num",     type=int, default=12,  help="max dynamic tiles")
    parser.add_argument("--vit_batch",   type=int, default=4,   help="ViT micro-batch size")
    parser.add_argument("--chunk_size",  type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--compare",     action="store_true",   help="also run standard generate for comparison")
    args = parser.parse_args()

    image = "4000x6000.jpg"
    question = "What is shown in this image in extreme detail?"

    torch.cuda.synchronize()
    t0 = time.time()

    nvtx.range_push("Load_Model")

    # ── 1. 載入模型 ──
    dtype = torch.bfloat16
    tokenizer, model = build_model(args.model_name, dtype=dtype)
    nvtx.range_pop()
    print(f"Model loaded, CUDA alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"downsample_ratio         : {model.downsample_ratio}")
    print(f"select_layer             : {model.select_layer}")

    # ── 2. 影像前處理 ──
    nvtx.range_push("Load_Image")
    pixel_values, num_patches = load_image_tiles(image, input_size=448, max_num=args.max_num)
    nvtx.range_pop()
    print(f"\nImage loaded: {num_patches} tiles, pixel_values={pixel_values.shape}")

    nvtx.range_push(f"INFERENCE")
    # ── 3. (可選) 標準 generate baseline ──
    if args.compare:
        print("\n[Baseline] Running standard model.chat() ...")
        ref_answer = generate_answer_standard(model, tokenizer, pixel_values, question)
        print("\n[Baseline Answer]")
        print(ref_answer)
        print(f"CUDA alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # # ── 4. Online KV Pipeline ──
    # answer = run_internvl_online_kv_pipeline(
    #     model=model,
    #     tokenizer=tokenizer,
    #     pixel_values=pixel_values,
    #     num_patches=num_patches,
    #     question=args.question,
    #     dtype=dtype,
    #     vit_batch=args.vit_batch,
    #     chunk_size=args.chunk_size,
    # )

    # print("\n[Online KV Answer]")
    # print(answer)

    nvtx.range_pop()

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
