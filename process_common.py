import os
import json
import glob
import base64
from io import BytesIO
from PIL import Image
import random

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


def load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                 num_image=None, prompt_mode="short"):
    """DocVQA loader（給 internvl_stream.py 用）。跟 load_hrbench 同一套設計：

    prompt_mode 對應兩種要比的設定（跟 HRBENCH_PROMPT 的 letter/open 是同一個精神）：

      "short"（預設，官方 ANLS 協定）：question 後面加一句作答指示，逼模型輸出
          短答案（DocVQA 官方就是這樣問的）。evaluate_anls.py 拿它跟 GT 算 ANLS。
      "open"：只給題目本身，不加任何作答指示，讓模型自由生成完整說明。
          沒有「正確答案」可比對，改用 semantic similarity 比「每個 budget 的輸出
          vs uncompressed reference」（跟 evaluate_anls.py --similarity / HR-Bench
          open 模式同一套）。

    每筆額外回傳 answer / answers / id，讓 internvl_stream.py 存進輸出 JSON 的
    meta（比照 load_hrbench）：
      - answers：DocVQA 官方允許的多個可接受答案（list），ANLS 要取 max。
      - id：HF dataset 原始的 questionId，不是攤平後的位置索引。

    這兩點是刻意補的：舊版 loader 完全不存 answer/id，導致 evaluate_anls.py
    只能用 --use_hf_gt 重新載入 HF dataset、
    靠「第 i 筆對第 i 筆」的假設對齊 GT——一旦這裡跳過任何一張圖（no image /
    壞檔），後面每一筆都會位移、ANLS 全部算錯且不會有任何警告。存進 meta 之後
    evaluate_anls.py 直接讀，不用重載、沒有對齊風險（跟 HR-Bench 修過的坑一樣）。
    """
    if prompt_mode not in ("short", "open"):
        raise ValueError(f"prompt_mode must be 'short' or 'open', got {prompt_mode!r}")

    ds = load_dataset(dataset, subject, split=split)
    limit = len(ds) if (num_image is None or num_image == -1) else min(int(num_image), len(ds))

    samples, skipped = [], 0
    for i in range(limit):
        s = ds[i]
        image = _to_pil(s.get("image") or s.get("image_1"))
        if image is None:
            skipped += 1
            continue

        q = str(s.get("question", "") or "").strip()

        raw_ans = s.get("answers")
        if raw_ans is None:
            raw_ans = s.get("answer")
        if isinstance(raw_ans, str):
            raw_ans = [raw_ans]
        answers = [str(a) for a in (raw_ans or [])]

        if prompt_mode == "short":
            prompt_q = f"{q}\nAnswer the question using a single word or phrase."
        else:
            prompt_q = q

        samples.append({
            "question": prompt_q,
            "image": image,
            "answer": answers[0] if answers else "",
            "answers": answers,
            "question_type": "docvqa",
            "id": str(s.get("questionId", s.get("id", i))),
        })

    print(f"[load_docvqa] {dataset}:{split} (prompt_mode={prompt_mode}) -> "
          f"kept {len(samples)}, skipped {skipped} (no image)")
    return samples


def load_infovqa(dataset="Ahren09/InfoVQA", split="val", num_image=None, prompt_mode="short",
                 sample_seed=None):
    """InfoVQA loader（給 internvl_stream.py 用）。設計比照 load_docvqa：

    prompt_mode：
      "short"（預設，官方 ANLS 協定）：question 後面加一句作答指示，輸出短答案。
      "open"：只給題目本身，自由生成。

    Ahren09/InfoVQA 的欄位：id (int) / question / answer (list[str]) / image / deck_name /
    image_url。注意 validation split 在這個 dataset 裡叫 "val"，不是 "validation"。
    sample_seed：None = 依原始順序取前 num_image 題；給整數時改成用這個 seed 隨機抽 num_image 題
    （held-out 校準集用，例如 train split 抽 600 題；抽到的順序固定，可重現）。
    每筆額外回傳 answer / answers / id，讓 internvl_stream.py 存進輸出 JSON 的 meta，
    ANLS 直接讀 meta，不用重載 HF dataset（避免跳圖造成的對齊位移，理由同 load_docvqa）。
    """
    if prompt_mode not in ("short", "open"):
        raise ValueError(f"prompt_mode must be 'short' or 'open', got {prompt_mode!r}")

    ds = load_dataset(dataset, split=split)
    limit = len(ds) if (num_image is None or num_image == -1) else min(int(num_image), len(ds))
    order = list(range(len(ds)))
    if sample_seed is not None:
        random.Random(sample_seed).shuffle(order)

    samples, skipped = [], 0
    for i in order[:limit]:
        s = ds[i]
        image = _to_pil(s.get("image"))
        if image is None:
            skipped += 1
            continue

        q = str(s.get("question", "") or "").strip()

        raw_ans = s.get("answer")
        if isinstance(raw_ans, str):
            raw_ans = [raw_ans]
        answers = [str(a) for a in (raw_ans or [])]

        if prompt_mode == "short":
            prompt_q = f"{q}\nAnswer the question using a single word or phrase."
        else:
            prompt_q = q

        samples.append({
            "question": prompt_q,
            "image": image,
            "answer": answers[0] if answers else "",
            "answers": answers,
            "question_type": "infovqa",
            "id": str(s.get("id", i)),
        })

    print(f"[load_infovqa] {dataset}:{split} (prompt_mode={prompt_mode}) -> "
          f"kept {len(samples)}, skipped {skipped} (no image)")
    return samples


def load_hrbench(dataset="DreamMr/HR-Bench", split="hrbench_4k", num_image=None, prompt_mode="letter"):
    """HR-Bench loader（給 internvl_stream.py 用）。

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
        格式跟 DocVQA 的作答指示同一個精神，evaluate 時直接比對輸出字母 vs answer。

      "open"：只給題目本身，不帶任何選項也不加作答指示，讓模型自由生成完整答案。
        evaluate 時改用 answer_text（正解選項的文字）去比對模型的自由輸出。

    每筆額外回傳 answer / answer_text / options / question_type / category / id，
    讓 internvl_stream.py 存進輸出 JSON 的 meta，evaluate 直接讀、不用重載 dataset。
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


MME_RW_ROOT = os.path.expanduser(os.environ.get("MME_RW_ROOT", "~/datasets/MME-RealWorld"))
# 官方 MME-RealWorld / VLMEvalKit 的作答指示
MME_RW_INSTRUCTION = ("Select the best answer to the above multiple-choice question based on the image. "
                      "Respond with only the letter (A, B, C, D, or E) of the correct option.\n"
                      "The best answer is:")


def load_mme_realworld(split="calib", num_image=None, root=MME_RW_ROOT):
    """MME-RealWorld loader（英文版，選擇題）。題目清單由 prepare_mme_realworld.py 產生：
      split="calib"：held-out 校準集（完整版扣掉 Lite 與 Lite 的圖，5 個 Subtask 各 120 題，seed 0）
      split="lite" ：測試集 MME-RealWorld-Lite（1,919 題）

    圖片很大（remote sensing 常超過 5000x5000），所以不預先讀進記憶體：每筆只放 image_path，
    用 get_image(d) 在用到時才開檔（internvl_stream.py 已改用 get_image）。
    meta：answer = 正解字母，options = 去掉 "(A) " 前綴的選項文字，category = Subtask/Category。
    """
    path = os.path.join(root, {"calib": "calib600.json", "lite": "lite.json"}[split])
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}，請先跑 python prepare_mme_realworld.py")
    items = json.load(open(path))
    if num_image is not None and num_image != -1:
        items = items[:int(num_image)]
    samples = []
    for d in items:
        choices = [str(c).strip() for c in d["Answer choices"]]
        options = [c[3:].strip() if len(c) > 3 and c[0] == "(" and c[2] == ")" else c for c in choices]
        prompt_q = (f"{d['Text'].strip()}\nThe choices are listed below:\n" + "\n".join(choices)
                    + f"\n{MME_RW_INSTRUCTION}")
        samples.append({
            "question": prompt_q,
            "image_path": os.path.join(root, d["Image"]),
            "answer": str(d["Ground truth"]).strip(),
            "answer_text": "",
            "options": options,
            "question_type": "multiple_choice",
            "category": f"{d['Subtask']}/{d['Category']}",
            "id": d["Question_id"],
        })
    print(f"[load_mme_realworld] {split}: {len(samples)} 題（{path}）")
    return samples


def get_image(d):
    """回傳 sample 的 PIL 影像：有 image 直接用，否則（MME-RealWorld 等大圖）從 image_path 開檔。"""
    if d.get("image") is not None:
        return d["image"]
    Image.MAX_IMAGE_PIXELS = None          # remote sensing 圖超過 PIL 預設的 decompression-bomb 上限
    with Image.open(d["image_path"]) as im:
        return im.convert("RGB")


VSTAR_ROOT = os.path.expanduser(os.environ.get("VSTAR_ROOT", "~/datasets/vstar_bench"))


def load_vstar(num_image=None, root=VSTAR_ROOT):
    """V* Bench（craigwu/vstar_bench）：direct_attributes 115 題 + relative_position 76 題 = 191 題，選擇題。
    題目直接用官方 test_questions.jsonl 的 text（已含選項與 "Answer with the option's letter ..." 作答指示）。
    下載：huggingface_hub.snapshot_download('craigwu/vstar_bench', repo_type='dataset', local_dir=VSTAR_ROOT,
          allow_patterns=['test_questions.jsonl', 'direct_attributes/*', 'relative_position/*'])
    圖片延遲開檔（get_image），meta：answer = 正解字母、options = 選項文字、category = 子類別。"""
    path = os.path.join(root, "test_questions.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"找不到 {path}，請先下載 V* Bench（見 load_vstar docstring）")
    items = [json.loads(l) for l in open(path)]
    if num_image is not None and num_image != -1:
        items = items[:int(num_image)]
    samples = []
    for d in items:
        options = [ln.strip()[3:].strip() for ln in d["text"].splitlines()
                   if len(ln.strip()) > 3 and ln.strip()[0] == "(" and ln.strip()[2] == ")"]
        samples.append({
            "question": d["text"],
            "image_path": os.path.join(root, d["image"]),
            "answer": str(d["label"]).strip(),
            "answer_text": "",
            "options": options,
            "question_type": "multiple_choice",
            "category": d["category"],
            "id": str(d["question_id"]),
        })
    print(f"[load_vstar] {len(samples)} 題（{path}）")
    return samples


RWQA_ROOT = os.path.expanduser(os.environ.get("RWQA_ROOT", "~/datasets/RealWorldQA"))


def load_realworldqa(num_image=None, root=RWQA_ROOT):
    """RealWorldQA（xai-org/RealWorldQA，test 765 題）：選擇題 438 題 + 開放短答 327 題（Yes/No/數字）。
    題目直接用官方 question（已含選項與作答指示："...only the letter of the correct option..." 或
    "...a single word or number."）。第一次呼叫時把圖存到 root/imgs/<idx>.png、題目存成 questions.json，
    之後圖片延遲開檔（get_image）。
    meta：選擇題 answer = 正解字母、options = 選項文字、question_type = "multiple_choice"；
          開放題 answer = 正解文字、options = []、question_type = "open"（summarize_results.py 用正規化後完全比對）。"""
    path = os.path.join(root, "questions.json")
    if not os.path.exists(path):
        os.makedirs(os.path.join(root, "imgs"), exist_ok=True)
        ds = load_dataset("xai-org/RealWorldQA", split="test")
        items = []
        for i, d in enumerate(ds):
            rel = f"imgs/{i:04d}.png"
            d["image"].convert("RGB").save(os.path.join(root, rel))
            items.append({"id": str(i), "question": d["question"], "answer": str(d["answer"]).strip(), "image": rel})
        json.dump(items, open(path, "w"), ensure_ascii=False, indent=1)
    items = json.load(open(path))
    if num_image is not None and num_image != -1:
        items = items[:int(num_image)]
    samples = []
    for d in items:
        options = [ln.strip()[2:].strip() for ln in d["question"].splitlines()
                   if len(ln.strip()) > 2 and ln.strip()[0] in "ABCDEF" and ln.strip()[1] in ".:"]
        samples.append({
            "question": d["question"],
            "image_path": os.path.join(root, d["image"]),
            "answer": d["answer"],
            "answer_text": "",
            "options": options,
            "question_type": "multiple_choice" if options else "open",
            "category": "multiple_choice" if options else "open",
            "id": d["id"],
        })
    print(f"[load_realworldqa] {len(samples)} 題（{path}）")
    return samples
