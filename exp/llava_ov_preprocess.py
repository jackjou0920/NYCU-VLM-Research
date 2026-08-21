import re
import os
import glob
import time
import torch
import contextlib
from PIL import Image
from datasets import load_dataset
from accelerate import Accelerator
from transformers import LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@contextlib.contextmanager
def measure_peak_memory(tag: str, record_timeline: bool = False):
    """跟 internvl_preprocess.py 完全相同的量測邏輯，直接複製過來避免額外的跨檔相依。"""
    if record_timeline:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        print(f"\n[{tag}] time={elapsed:.2f} sec  "
              f"peak_allocated={peak_alloc:.3f} GB  "
              f"peak_reserved={peak_reserved:.3f} GB")

        if record_timeline:
            torch.cuda.memory._dump_snapshot(f"{tag}_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)


def load_local(image, batch_size, num_image=None):
    question = "What is shown in this image in extreme detail?"

    if os.path.isdir(image):
        image_paths = sorted(sum(
            [glob.glob(os.path.join(image, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []
        ))
        if not image_paths:
            raise FileNotFoundError(f"No images in {image}")
    else:
        image_paths = [image] * batch_size

    if num_image is not None:
        image_paths = image_paths[:num_image]

    samples = []
    for path in image_paths:
        samples.append({
            "question": question, "image": Image.open(path).convert("RGB")
        })
    # print(f"Loaded {len(samples)} samples")

    return samples


def extract_question_image(sample):
    """
    從 MMMU sample 中：
    1. 找 question 中第一個 <image N>
    2. 取得對應的 image_N
    3. 移除 question 中所有 <image N>
    
    Returns:
        {
            "question": str,
            "image": PIL.Image,
            "image_index": int,
        }
    """
    question = sample["question"]

    # --------------------------------------------------------
    # 找第一個 image placeholder
    # --------------------------------------------------------
    match = re.search(r"<image\s+(\d+)>", question)
    if match is None:
        return None

    image_index = int(match.group(1))

    # --------------------------------------------------------
    # 根據 placeholder 找真正對應的 image
    # --------------------------------------------------------
    image = sample.get(f"image_{image_index}")
    if image is None:
        return None

    # --------------------------------------------------------
    # 移除所有 image placeholder
    # --------------------------------------------------------
    question = re.sub(r"<image\s+\d+>", "", question).strip()

    return {
        "question": question,
        "image": image,
        "image_index": image_index,
    }


def load_mmmu(subject="Agriculture", split="test", num_image=None):
    ds = load_dataset("MMMU/MMMU", subject, split=split)

    if num_image is None:
        num_image = float("inf")

    samples = []
    for i in range(min(num_image, len(ds))):
        result = extract_question_image(ds[i])
        if result is not None:
            samples.append(result)
    # print(f"Loaded {len(samples)} samples")

    # for i, sample in enumerate(samples):
    #     print("=" * 80)
    #     print(f"Sample {i+1}")
    #     print(f"Image index: {sample['image_index']}")
    #     print(f"Image size: {sample['image'].size}")
    #     print(f"Question: {sample['question']}")
    return samples


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
):
    print(f"Loading processor from {model_name} ...")
    processor = LlavaOnevisionProcessor.from_pretrained(model_name)

    print(f"Loading model from {model_name} ...")
    device_map = Accelerator().device
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
        device_map=device_map,
    ).to(DEVICE).eval()

    return processor, model
