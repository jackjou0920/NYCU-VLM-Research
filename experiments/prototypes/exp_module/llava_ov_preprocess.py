import torch
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration


# ──────────────────────────────────────────────────────────────────────────────
# 影像前處理
#
# 跟 InternVL 的 dynamic_preprocess 不同，LLaVA-OneVision 的 anyres 切圖邏輯已經
# 內建在 LlavaOnevisionImageProcessor 裡（select_best_resolution + pad + crop），
# 所以這裡不重造輪子，直接呼叫 processor.image_processor。
#
# 回傳的 pixel_values 形狀是 [num_patches, 3, H, W]，其中：
#   patch 0        = "base image"，整張圖 resize 成正方形的縮圖（相當於 InternVL 的 thumbnail，
#                     只是位置在最前面而不是最後面）
#   patch 1..N-1   = anyres 網格裁切出來的高解析度 crop，順序是 row-major（跟 InternVL tile 的
#                     raster-scan 順序一致）
# ──────────────────────────────────────────────────────────────────────────────
def load_image_patches(processor, datasets):
    """回傳 pixel_values [N_patches, 3, H, W] 以及該圖的原始尺寸 image_size (H, W)"""
    pixel_values_list, image_sizes_list = [], []
    for dataset in datasets:
        out = processor.image_processor(images=dataset["image"], return_tensors="pt")
        pixel_values_list.append(out["pixel_values"][0])  # [N_patches, 3, H, W]
        image_sizes_list.append(out["image_sizes"][0])    # tensor([H, W])，原圖尺寸（本專案的串流路徑用不到，但保留供除錯用）

    return pixel_values_list, image_sizes_list


# ──────────────────────────────────────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────────────────────────────────────
def build_model(
    model_name: str = "llava-hf/llava-onevision-qwen2-7b-ov-hf",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0"
):
    print(f"Loading processor from {model_name} ...")
    processor = LlavaOnevisionProcessor.from_pretrained(model_name)

    print(f"Loading model from {model_name} ...")
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).to(device).eval()

    return processor, model
