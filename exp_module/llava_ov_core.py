# ══════════════════════════════════════════════════════════════════════════════
# LLaVA-OneVision 專屬的低階函式（prompt 構造 / patch 編碼 / row layout / baseline）。
#
# 獨立成這支「葉節點」模組，原因跟 internvl_core.py 一樣：避免
# llava_ov_svm.py（entry point，需要 LlavaOVAdapter）與 llava_ov_adapter.py
# （需要這裡的函式）互相 import 形成循環。內容跟原本 llava_ov_svm.py 裡的
# 版本逐行相同，純粹搬家，沒有改任何邏輯或數學。
# ══════════════════════════════════════════════════════════════════════════════
import math
import torch
from transformers.models.llava_onevision.modeling_llava_onevision import unpad_image
from stream_adapters import DEVICE
 
MAX_NEW_TOKENS = 300
IMAGE_TOKEN = "<image>"
 
 
def clean_answer(datasets, answers):
    for i, (dataset, answer) in enumerate(zip(datasets, answers)):
        answer = answer.replace("user", "").strip()
        answer = answer.replace(dataset["question"], "").strip()
        answer = answer.replace("assistant", "").strip()
 
        answer = answer.replace("<|im_end|>", "").strip()
        answers[i] = answer
 
    return answers
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 標準 generate 參考路徑（用於正確性比較）
#
# 跟 InternVL 那邊用 model.batch_chat() 不一樣：LLaVA-OneVision 是標準 HF
# 模型，沒有 trust_remote_code 自帶的 chat helper，所以這裡直接走
# processor(...) + model.generate(...) 這條「完全沒有做任何壓縮」的官方路徑
# —— 也就是會跑滿完整的 pack_image_features()（unpad + 必要時的雙線性內插 +
# 逐 token-row image_newline），是跟 streaming memory bank 版本比較答案品質
# 時真正該拿來當 ground truth 的基準。
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer_standard(model, processor, datasets):
    texts = []
    for dataset in datasets:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": dataset["question"]},
                ],
            }
        ]
        prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        texts.append(prompt)
 
    images = [dataset["image"] for dataset in datasets]
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
 
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )
 
    answers = processor.batch_decode(output_ids, skip_special_tokens=True)
 
    del inputs, images, output_ids
    torch.cuda.empty_cache()
    return clean_answer(datasets, answers)
 
 
# ──────────────────────────────────────────────────────────────────────────────
# Prompt 工具
# ──────────────────────────────────────────────────────────────────────────────
def build_prompt(processor, question: str) -> str:
    messages = [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question},
        ],
    }]
    return processor.apply_chat_template(messages, add_generation_prompt=True)
 
 
def split_prompt_at_vision(prompt: str):
    idx = prompt.find(IMAGE_TOKEN)
    if idx == -1:
        raise ValueError(f"Prompt 裡找不到 {IMAGE_TOKEN}，chat template 可能跟預期不一樣: {prompt!r}")
    text_before = prompt[:idx]
    text_after = prompt[idx + len(IMAGE_TOKEN):]
    return text_before, text_after
 
 
# ──────────────────────────────────────────────────────────────────────────────
# 串流核心：單一 patch 的 SigLIP → multi_modal_projector
# （這一段純粹是為了 ViT forward 的 peak activation memory 不要一次爆掉，
#   跟「要不要做 eviction」是兩件事，budget 開多大都需要跑這段）
# ──────────────────────────────────────────────────────────────────────────────
def encode_patch(model, patch_pixel_values: torch.Tensor, dtype) -> torch.Tensor:
    """
    輸入  : patch_pixel_values [B_patch, 3, H, W]
    輸出  : [B_patch, num_image_token, D_llm]（已經過 multi_modal_projector）
    """
    core = model.model
    vision_feature_layer = model.config.vision_feature_layer
    vision_feature_select_strategy = model.config.vision_feature_select_strategy
 
    with torch.no_grad():
        vis_out = core.vision_tower(
            pixel_values=patch_pixel_values,
            output_hidden_states=True,
            return_dict=True,
        )
        if isinstance(vision_feature_layer, int):
            feats = vis_out.hidden_states[vision_feature_layer]
        else:
            feats = torch.cat([vis_out.hidden_states[i] for i in vision_feature_layer], dim=-1)
 
        if vision_feature_select_strategy == "default":
            feats = feats[:, 1:, :]
        del vis_out
 
        tile_tokens = core.multi_modal_projector(feats)
        del feats
 
    return tile_tokens.to(dtype)
 
 
def num_image_tokens_per_patch(model) -> int:
    """anyres 每個 patch（base image 或 grid crop）固定輸出幾個 vision token。"""
    h = w = model.config.vision_config.image_size // model.config.vision_config.patch_size
    return h * w
 
 
def compute_packed_row_layout(model, image_size, grid_shape):
    """
    複製官方 pack_image_features() 裡「攤平後每一列（含 image_newline）有幾個
    token、總共幾列」的那段數學（reshape → unpad → 視情況 bilinear
    interpolate），直接呼叫官方的 unpad_image，只用一個內容全 0、形狀正確的
    dummy tensor 跑一次拿 shape，不涉及任何真正的視覺特徵、也不會跟真正的
    pack_image_features 算出不一樣的結果（用的是同一個函式）。
 
    回傳 (num_rows, row_width_with_newline)。
    row_width_with_newline = 官方 interpolate 後的 curr_width + 1
                              （+1 是官方會在每一列最後補的 image_newline）。
    """
    num_patch_height, num_patch_width = grid_shape
    h = w = model.config.vision_config.image_size // model.config.vision_config.patch_size
 
    dummy = torch.zeros(1, num_patch_height * h, num_patch_width * w)
    dummy = unpad_image(dummy, image_size)
    _, curr_height, curr_width = dummy.shape
 
    max_num_patches = 9   # 官方預設 vision_aspect_ratio="anyres_max_9"
    ratio = math.sqrt(curr_height * curr_width / (max_num_patches * h * h))
    if ratio > 1.1:
        curr_height, curr_width = int(curr_height // ratio), int(curr_width // ratio)
 
    return curr_height, curr_width + 1
