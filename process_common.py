import re
import os
import glob
from PIL import Image
from datasets import load_dataset


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
        return {
            "question": question,
            "image": sample.get("image"),
            "image_index": -1,
        }

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


def load_hf_dataset(dataset="MMMU/MMMU", subject="Agriculture", split="test", num_image=None):
    ds = load_dataset(dataset, subject, split=split)

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
