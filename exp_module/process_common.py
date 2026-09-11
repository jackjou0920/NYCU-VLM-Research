import re
import os
import ast
import glob
import base64
import string
from io import BytesIO
from PIL import Image
from datasets import load_dataset


def _to_pil(img):
    """把 HF dataset 各種可能的影像表示統一轉成 RGB 的 PIL.Image。

    HR-Bench（DreamMr/HR-Bench）的 image 欄位存的是 base64 字串，不是 PIL 物件，
    直接 .convert() 會炸 AttributeError，所以這裡集中處理：
      - 已是 PIL.Image           → 直接 convert
      - dict（HF Image 未 decode）→ 讀 bytes / path
      - bytes / bytearray        → 當成圖檔位元組
      - str                      → 先試 base64（HR-Bench），失敗再當本機路徑
    無法解讀時回傳 None，讓呼叫端當「沒有影像」跳過。
    """
    if img is None:
        return None
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, dict):
        if img.get("bytes"):
            return Image.open(BytesIO(img["bytes"])).convert("RGB")
        if img.get("path") and os.path.exists(img["path"]):
            return Image.open(img["path"]).convert("RGB")
        return None
    if isinstance(img, (bytes, bytearray)):
        return Image.open(BytesIO(bytes(img))).convert("RGB")
    if isinstance(img, str):
        s = img.split(",", 1)[-1] if img.startswith("data:image") else img
        try:
            return Image.open(BytesIO(base64.b64decode(s))).convert("RGB")
        except Exception:  # noqa: BLE001  不是 base64 就往下試路徑
            pass
        if os.path.exists(img):
            return Image.open(img).convert("RGB")
    return None


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
        # 沒有 <image N> placeholder：MMMU 沒有純 "image" 欄位，退回 image_1。
        img = sample.get("image") or sample.get("image_1")
        return {
            "question": question,
            "image": img.convert("RGB") if img is not None else None,
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
        "image": image.convert("RGB"),   # MMMU 有非 RGB 圖，統一轉 RGB（跟 load_local 一致）
        "image_index": image_index,
    }


def load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation", num_image=None):
    ds = load_dataset(dataset, subject, split=split)

    if num_image is None:
        num_image = float("inf")

    samples = []
    skipped = 0
    for i in range(min(num_image, len(ds))):
        result = extract_question_image(ds[i])
        if result is not None and result.get("image") is not None:
            samples.append(result)
        else:
            skipped += 1
    if skipped:
        print(f"[load_hf_dataset] skipped {skipped} sample(s) with no usable image")
    # print(f"Loaded {len(samples)} samples")
    return samples


def load_mmmu(split="validation", num_image=None, dataset="lmms-lab-encoder/MMMU"):
    """MMMU 專用 loader（給 internvl_svm.py 用）。跟 load_hf_dataset 的差別：

    1. prompt 會把選項一起帶上（MMMU 是選擇題，模型看不到選項根本沒法答），
       格式比照官方 MMMU：question + "A. .. / B. .." + "Answer with the option's
       letter from the given choices directly."；open 題則加簡答指示。
    2. 每筆多回傳 answer / options / question_type / id，讓 internvl_svm.py 存進
       輸出 JSON 的 meta，evaluate_mmmu.py 直接讀、不用重載 dataset（免對齊風險）。
    3. 影像取 question 裡第一個 <image N> 對應的 image_N（多圖題只用第一張，
       單圖 pipeline 的限制，會印出多圖題數量）。沒有可用影像的題目跳過。
    """
    ds = load_dataset(dataset, None, split=split)
    limit = len(ds) if num_image is None else min(int(num_image), len(ds))
    letters = string.ascii_uppercase

    samples, skipped, multi = [], 0, 0
    for i in range(limit):
        s = ds[i]
        q = str(s.get("question", "") or "")
        idxs = re.findall(r"<image\s+(\d+)>", q)
        if idxs:
            if len(set(idxs)) > 1:
                multi += 1
            img = s.get(f"image_{int(idxs[0])}")
        else:
            img = s.get("image") or s.get("image_1")
        if img is None:
            skipped += 1
            continue

        q_clean = re.sub(r"<image\s+\d+>", "", q).strip()

        raw_opts = s.get("options")
        if isinstance(raw_opts, str):
            try:
                raw_opts = ast.literal_eval(raw_opts)
            except (ValueError, SyntaxError):
                raw_opts = []
        opts = [str(o) for o in (raw_opts or [])]

        qtype = str(s.get("question_type") or ("multiple-choice" if opts else "open"))
        answer = s.get("answer")
        if isinstance(answer, list):
            answer = answer[0] if answer else ""

        if qtype == "multiple-choice" and opts:
            choice_block = "\n".join(f"{letters[j]}. {o}" for j, o in enumerate(opts))
            prompt_q = (f"{q_clean}\n{choice_block}\n"
                        "Answer with the option's letter from the given choices directly.")
        else:
            prompt_q = f"{q_clean}\nAnswer the question using a single word or phrase."

        samples.append({
            "question": prompt_q,
            "image": img.convert("RGB"),
            "answer": "" if answer is None else str(answer),
            "options": opts,
            "question_type": qtype,
            "id": str(s.get("id", i)),
        })

    print(f"[load_mmmu] {dataset}:{split} -> kept {len(samples)}, "
          f"skipped {skipped} (no image), {multi} multi-image question(s) (first image only)")
    return samples


def load_hrbench(dataset="DreamMr/HR-Bench", split="hrbench_4k", num_image=None, prompt_mode="letter"):
    """HR-Bench loader（給 internvl_stream_v2.py 用）。

    HR-Bench 每筆長這樣（欄位名固定）：
        question : "What is the number displayed above the entrance ..."
        A / B / C / D : 四個選項文字，例如 "27B" / "37B" / "27D" / "27E"
        answer   : 正解的選項字母，例如 "A"
        category : "single" / "cross"（fine-grained single/cross-instance perception）
        image    : PIL.Image（4K/8K 高解析度原圖）

    prompt_mode 決定丟給模型的 question 長怎樣，對應你要比的兩種設定：

      "letter"（預設）：把四個選項組進 prompt，並明確要求模型「只輸出一個選項字母」。
          {question}
          A. 27B
          B. 37B
          C. 27D
          D. 27E
          Answer with the option's letter from the given choices directly.
        格式比照 load_mmmu 的選擇題 prompt，evaluate 時直接比對輸出字母 vs answer。

      "open"：只給題目本身，不帶任何選項也不加作答指示，讓模型自由生成完整答案。
        evaluate 時改用 answer_text（正解選項的文字）去比對模型的自由輸出。

    每筆額外回傳 answer / answer_text / options / question_type / category / id，
    讓 internvl_stream_v2.py 存進輸出 JSON 的 meta（比照 load_mmmu），
    evaluate 直接讀、不用重載 dataset。
    """
    print("prompt_mode:", prompt_mode)
    if prompt_mode not in ("letter", "open"):
        raise ValueError(f"prompt_mode must be 'letter' or 'open', got {prompt_mode!r}")

    ds = load_dataset(dataset, split=split)
    limit = len(ds) if (num_image is None or num_image == -1) else min(int(num_image), len(ds))
    letters = ["A", "B", "C", "D"]

    samples, skipped = [], 0
    for i in range(limit):
        s = ds[i]
        image = _to_pil(s.get("image") or s.get("image_1"))
        if image is None:
            skipped += 1
            continue

        q = str(s.get("question", "") or "").strip()
        opts = [str(s.get(L, "") or "") for L in letters]

        answer = str(s.get("answer", "") or "").strip()
        answer_text = ""
        if answer in letters:
            answer_text = opts[letters.index(answer)]

        if prompt_mode == "letter":
            choice_block = "\n".join(f"{L}. {o}" for L, o in zip(letters, opts) if o)
            prompt_q = (f"{q}\n{choice_block}\n"
                        "Answer with the option's letter from the given choices directly.")
        else:
            prompt_q = q

        samples.append({
            "question": prompt_q,
            "image": image,
            "answer": answer,
            "answer_text": answer_text,
            "options": opts,
            "question_type": "multiple-choice",
            "category": str(s.get("category", "") or ""),
            "id": str(s.get("id", i)),
        })

    print(f"[load_hrbench] {dataset}:{split} (prompt_mode={prompt_mode}) -> "
          f"kept {len(samples)}, skipped {skipped} (no image)")
    return samples
