import re
import os
import glob
import torch
from datasets import load_dataset


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

def generate_answer(model, processor, inputs, messages):
    """
    Returns generated answer from LLaVA-OneVision.
    """

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=False,
        )

    answer = processor.batch_decode(
        output_ids,
        skip_special_tokens=True,
    )[0]
    return answer

def load_mmmu(subject="Agriculture", split="test", num_image=None):
    ds = load_dataset("MMMU/MMMU", subject, split=split)

    if num_image is None:
        num_image = float("inf")

    samples = []
    for i in range(min(num_image, len(ds))):
        result = extract_question_image(ds[i])
        if result is not None:
            samples.append(result)
    print(f"Loaded {len(samples)} samples")

    for i, sample in enumerate(samples):
        print("=" * 80)
        print(f"Sample {i+1}")
        print(f"Image index: {sample['image_index']}")
        print(f"Image size: {sample['image'].size}")
        print(f"Question: {sample['question']}")


load_mmmu(num_image=3)
