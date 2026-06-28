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


def build_prompt(tokenizer, model, question: str, num_patches: int) -> str:
    """
    使用 InternVL 的 conv_template 建構完整 prompt，
    並將 <image> 佔位符展開為真正的 <img><IMG_CONTEXT>×N</img> 格式。
    """
    from internvl.model.internvl_chat.conversation import get_conv_template  # noqa: F401

    # 若無法 import，使用 model 內建的 conv_template
    template = model.conv_template
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    prompt = template.get_prompt()

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
        + IMG_END_TOKEN
    )
    prompt = prompt.replace("<image>", image_tokens, 1)
    return prompt


def _build_prompt_simple(tokenizer, model, question: str, num_patches: int) -> str:
    """
    不依賴 internvl 套件的簡化版本：直接用 Qwen3 chat template。
    InternVL3.5-8B (Qwen3 backbone) 的 system message 格式：
        <|im_start|>system\n{sys}<|im_end|>\n
        <|im_start|>user\n{content}<|im_end|>\n
        <|im_start|>assistant\n
    """
    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * model.num_image_token * num_patches
        + IMG_END_TOKEN
    )
    content = image_tokens + "\n" + question

    # apply_chat_template（Qwen3 格式）
    messages = [{"role": "user", "content": content}]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return prompt


def split_prompt_at_vision(prompt: str, tokenizer):
    """
    把 prompt 在 <img>…</img> 區段切成三份：
        text_before : <img> 之前的文字
        vision_span : 整個 <img>...</img> 字串（僅用於計算，不 tokenize）
        text_after  : </img> 之後的文字
    """
    img_start = prompt.find(IMG_START_TOKEN)
    img_end   = prompt.find(IMG_END_TOKEN) + len(IMG_END_TOKEN)

    text_before = prompt[:img_start]
    text_after  = prompt[img_end:]
    return text_before, text_after


# ──────────────────────────────────────────────────────────────────────────────
# Online KV 主函式
# ──────────────────────────────────────────────────────────────────────────────

def run_internvl_online_kv_pipeline(
    model,
    tokenizer,
    pixel_values: torch.Tensor,     # [N_tiles, 3, 448, 448]
    num_patches: int,
    question: str,
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,             # 每次送入 ViT 的 tile 數
    chunk_size: int = 512,          # LLM Chunked Prefill 的 token 數
):
    """
    InternVL3.5 Online KV Construction Pipeline

    架構對應關係（InternVL vs LlavaOnevision）：
      model.vision_model         ↔  model.model.vision_tower
      model.mlp1                 ↔  model.model.multi_modal_projector
      pixel_shuffle (內嵌在 extract_feature) ↔ pack_image_features
      model.language_model       ↔  model.model.language_model
    """

    # ──────────────────────────────────────────────────────────────────
    # Step 1 : 建構 Prompt，設定 img_context_token_id
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 1] Build prompt ...")

    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    model.img_context_token_id = img_context_token_id

    prompt = _build_prompt_simple(tokenizer, model, question, num_patches)
    print(f"  ├─> num_image_token per tile : {model.num_image_token}")
    print(f"  ├─> num_patches              : {num_patches}")
    print(f"  └─> total vision tokens      : {model.num_image_token * num_patches}")

    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)

    # ──────────────────────────────────────────────────────────────────
    # Step 2 : ViT 微批次串流
    #   InternVL : vision_model 輸出 hidden_states[select_layer]
    #              → 去掉 CLS token ([:, 1:, :])
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 2] ViT Micro-batching ...")
    pixel_values = pixel_values.to(DEVICE, dtype=dtype)
    N_tiles = pixel_values.shape[0]     # e.g. 13

    vit_feature_list = []
    select_layer = model.select_layer   # 通常 = -1 或某個負數

    with torch.no_grad():
        for i in range(0, N_tiles, vit_batch):
            chunk = pixel_values[i : i + vit_batch]

            if select_layer == -1:
                vit_out = model.vision_model(
                    pixel_values=chunk,
                    output_hidden_states=False,
                    return_dict=True,
                )
                feats = vit_out.last_hidden_state     # [B, 1+HW, D]
            else:
                vit_out = model.vision_model(
                    pixel_values=chunk,
                    output_hidden_states=True,
                    return_dict=True,
                )
                feats = vit_out.hidden_states[select_layer]  # [B, 1+HW, D]

            feats = feats[:, 1:, :]   # 去掉 CLS token → [B, HW, D]
            vit_feature_list.append(feats)

            del vit_out, chunk
            torch.cuda.empty_cache()

            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  ├─> ViT chunk {i}~{min(i+vit_batch, N_tiles)} done, alloc={mem:.2f} GB")

    vit_embeds = torch.cat(vit_feature_list, dim=0)  # [N_tiles, HW, D_vit]
    print(f"  └─> vit_embeds shape = {vit_embeds.shape}")

    # ──────────────────────────────────────────────────────────────────
    # Step 3 : pixel_shuffle + MLP Projector
    #   InternVL 的 extract_feature 邏輯：
    #     vit_embeds → reshape → pixel_shuffle → reshape → mlp1
    #   等價於把每個 tile 的 1024 個 patch token 壓縮到 256 個
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 3] pixel_shuffle + MLP Projector ...")

    with torch.no_grad():
        h = w = int(vit_embeds.shape[1] ** 0.5)   # 32 for 448px tile
        vit_embeds_hw = vit_embeds.reshape(N_tiles, h, w, -1)
        vit_shuffled  = model.pixel_shuffle(vit_embeds_hw, scale_factor=model.downsample_ratio)
        # pixel_shuffle 後 shape: [N_tiles, h/2, w/2, D_vit*4]
        vit_shuffled  = vit_shuffled.reshape(N_tiles, -1, vit_shuffled.shape[-1])
        image_features = model.mlp1(vit_shuffled)  # [N_tiles, 256, D_llm]

    # 攤平成一長串 vision token
    total_vision_tokens = N_tiles * image_features.shape[1]  # N_tiles × 256
    packed_image_features = image_features.reshape(1, total_vision_tokens, -1)
    print(f"  └─> packed_image_features = {packed_image_features.shape}")
    print(f"      (total_vision_tokens = {total_vision_tokens})")

    mem = torch.cuda.memory_allocated() / 1e9
    print(f"CUDA memory allocated: {mem:.2f} GB")

    # ──────────────────────────────────────────────────────────────────
    # Step 4 : LLM Chunked KV 增量建構
    # ──────────────────────────────────────────────────────────────────
    print(f"\n[Step 4] Chunked KV Prefill (chunk_size={chunk_size}) ...")

    past_key_values = None

    # 4a. 注入 <img> 之前的文字
    if text_before:
        ids_before = tokenizer(text_before, return_tensors="pt").to(DEVICE)
        embeds_before = model.language_model.get_input_embeddings()(ids_before.input_ids)
        attn_mask = torch.ones((1, embeds_before.shape[1]), dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            out = model.language_model(
                inputs_embeds=embeds_before,
                attention_mask=attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = out.past_key_values
        del out, embeds_before, attn_mask
        torch.cuda.empty_cache()

        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  ├─> text_before injected, alloc={mem:.2f} GB")

    # 4b. 分批注入 vision token
    for i in range(0, total_vision_tokens, chunk_size):
        chunk_end   = min(i + chunk_size, total_vision_tokens)
        vision_chunk = packed_image_features[:, i:chunk_end, :]
        chunk_len    = chunk_end - i

        past_seq_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        attn_mask    = torch.ones((1, past_seq_len + chunk_len), dtype=torch.long, device=DEVICE)
        position_ids = torch.arange(
            past_seq_len, past_seq_len + chunk_len, dtype=torch.long, device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            out = model.language_model(
                inputs_embeds=vision_chunk,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = out.past_key_values
        del out, vision_chunk, attn_mask, position_ids
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  ├─> vision chunk {i:05d}~{chunk_end:05d}/{total_vision_tokens}, alloc={mem:.2f} GB")

    print("  └─> Vision KV cache built")
    mem = torch.cuda.memory_allocated() / 1e9
    print(f"CUDA memory allocated: {mem:.2f} GB")

    # ──────────────────────────────────────────────────────────────────
    # Step 5 : 文字後半 Prefill + 自迴歸解碼
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 5] Text Prefill + Autoregressive Decode ...")

    ids_after = tokenizer(text_after, return_tensors="pt").to(DEVICE)
    ids_after_ids = ids_after.input_ids
    text_len = ids_after_ids.shape[1]

    past_seq_len = past_key_values.get_seq_length()
    attn_mask    = torch.ones((1, past_seq_len + text_len), dtype=torch.long, device=DEVICE)
    position_ids = torch.arange(
        past_seq_len, past_seq_len + text_len, dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    with torch.no_grad():
        out = model.language_model(
            input_ids=ids_after_ids,
            attention_mask=attn_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )

    past_key_values    = out.past_key_values
    next_token_logits  = out.logits[:, -1, :]
    next_token         = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    del out, attn_mask, position_ids

    eos_token_id = tokenizer.eos_token_id

    print("=" * 60)
    answer = tokenizer.decode(next_token[0])

    # 自迴歸解碼迴圈
    for step in range(MAX_NEW_TOKENS):
        past_seq_len = past_key_values.get_seq_length()
        attn_mask    = torch.ones((1, past_seq_len + 1), dtype=torch.long, device=DEVICE)
        position_ids = torch.tensor([[past_seq_len]], dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            out = model.language_model(
                input_ids=next_token,
                attention_mask=attn_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        past_key_values   = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_token        = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

        token_id = next_token.item()
        if token_id == eos_token_id:
            del out, next_token_logits, attn_mask, position_ids
            break

        token   = tokenizer.decode([token_id])
        answer += token

        del out, next_token_logits, attn_mask, position_ids

        if step % 50 == 0:
            torch.cuda.empty_cache()

    print("=" * 60)
    return answer


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

    dtype = torch.bfloat16

    torch.cuda.synchronize()
    t0 = time.time()

    # ── 1. 載入模型 ──
    tokenizer, model = build_model(args.model_name, dtype=dtype)
    print(f"Model loaded, CUDA alloc: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"downsample_ratio         : {model.downsample_ratio}")
    print(f"select_layer             : {model.select_layer}")

    # ── 2. 影像前處理 ──
    pixel_values, num_patches = load_image_tiles(image, input_size=448, max_num=args.max_num)
    print(f"\nImage loaded: {num_patches} tiles, pixel_values={pixel_values.shape}")

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

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
