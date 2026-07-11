
"""
InternVL3.5-8B Online KV Construction Pipeline
================================================================
與「分階段版」(Step2 全部做完 → Step3 全部做完 → Step4 才開始) 的差異：
 
  分階段版：
      [ViT tile1..N] → [Projector tile1..N] → [LLM chunked prefill]
      三個階段的 activation 峰值是「疊加」的（Step3 開始時 Step2 的中間產物還留著）
 
  本版（per-tile 串流）：
      for tile in tiles:
          ViT(tile) → Projector(tile) → 立刻丟進 LLM 做 chunked prefill
      三個階段的峰值是「交錯」的：同一時間只有「當前這個 tile」的 ViT/Projector
      activation + LLM 的 KV cache，不會有「全部 tile 的 image_features 同時存在」
      這個量級的記憶體佔用。
 
可行性前提（已在前一輪確認）：
  InternVL 沒有 LLaVA-OneVision 的 pack_image_features（跨 tile 全局空間重組），
  每個 tile 經 pixel_shuffle + mlp1 後就是獨立的 256 個 token，tile 間互不依賴，
  因此可以「算完一個 tile 就丟進 LLM」，不需要等全部 tile 的 ViT 都跑完。
 
LLM chunk 的對齊規則：
  每個 tile 固定產生 num_image_token（通常 256）個 vision token。
  為了讓「LLM chunk 邊界」盡量對齊「tile 邊界」(避免每個 tile 都觸發一次 LLM forward
  造成 launch overhead 過高)，本版採用「tile 緩衝區」設計：
      把連續幾個 tile 的 projector 輸出先暫存，湊滿 chunk_size 再丟進 LLM。
  這樣依然是串流（不需等全部 tile 跑完 ViT），但 LLM forward 次數可控。
"""

import time
import argparse
import torch
from PIL import Image
from accelerate import Accelerator
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, StaticCache

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150

# ──────────────────────────────────────────────────────────────────────────────
# 影像前處理（動態高解析度，複製自官方 HF model card 範例）
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


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
    return pixel_values


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
# 標準 generate 參考路徑（用於正確性比較）
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer_standard(model, tokenizer, pixel_values: torch.Tensor, question: str, batch_size: int):
    """走官方 model.chat() / model.batch_chat() 路徑，把同一張圖的 pixel_values 複製 batch_size 次一起餵進去。"""

    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    pixel_values = pixel_values.to(DEVICE, dtype=model.dtype)

    # 同一張圖複製 batch_size 份，沿 batch 維度接在一起
    num_patches_list = [pixel_values.shape[0]] * batch_size
    pixel_values_batch = pixel_values.repeat(batch_size, 1, 1, 1)

    questions = [f'<image>\n{question}'] * batch_size
    responses = model.batch_chat(
        tokenizer,
        pixel_values_batch,
        num_patches_list=num_patches_list,
        questions=questions,
        generation_config=generation_config,
    )
    
    # # single-image multi-round conversation (单图多轮对话)
    # question = f'<image>\n{question}'
    # response, history = model.chat(
    #     tokenizer, 
    #     pixel_values.to(DEVICE, dtype=model.dtype), 
    #     question, 
    #     generation_config, 
    #     history=None, 
    #     return_history=True
    # )

    return responses


# ──────────────────────────────────────────────────────────────────────────────
# Prompt 工具
# ──────────────────────────────────────────────────────────────────────────────

IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"

def build_prompt(tokenizer, model, question: str, num_image_tiles: int) -> str:
    """
    不依賴 internvl 套件的簡化版本：直接用 Qwen3 chat template。
    InternVL3.5-8B (Qwen3 backbone) 的 system message 格式：
        <|im_start|>system\n{sys}<|im_end|>\n
        <|im_start|>user\n{content}<|im_end|>\n
        <|im_start|>assistant\n
    """
    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * model.num_image_token * num_image_tiles
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

    e.g.
    <|im_start|>user
    <img><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT><IMG_CONTEXT> ..... <IMG_CONTEXT></img>
    What is shown in this image in extreme detail?<|im_end|>
    <|im_start|>assistant
    """
    # print(f"Imput Prompt:\n{prompt}")
    img_start = prompt.find(IMG_START_TOKEN)
    img_end   = prompt.find(IMG_END_TOKEN) + len(IMG_END_TOKEN)

    text_before = prompt[:img_start]
    text_after  = prompt[img_end:]
    return text_before, text_after


# ──────────────────────────────────────────────────────────────────────────────
# 串流核心：單一 tile 的 ViT → pixel_shuffle → mlp1
# ──────────────────────────────────────────────────────────────────────────────
def encode_tile(model, tile_pixel_values: torch.Tensor, dtype) -> torch.Tensor:
    """
    輸入  : tile_pixel_values [B_tile, 3, 448, 448]  (B_tile 通常 = vit_micro_batch)
    輸出  : [B_tile, num_image_token, D_llm]  (已經過 pixel_shuffle + mlp1，可直接餵 LLM)
 
    這個函式刻意保持「無狀態、單一 tile-batch 進、單一 tile-batch 出」，
    是讓 Step2/3 能跟 Step4 交錯執行的關鍵：呼叫方可以一次只處理一小批 tile，
    處理完立刻丟去做 LLM prefill，不需要等其他 tile。
    """

    select_layer = model.select_layer  # 通常 = -1 或某個負數

    with torch.no_grad():
        if select_layer == -1:
            vit_out = model.vision_model(
                pixel_values=tile_pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
            feats = vit_out.last_hidden_state
        else:
            vit_out = model.vision_model(
                pixel_values=tile_pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            feats = vit_out.hidden_states[select_layer]

        feats = feats[:, 1:, :]   # 去 CLS token → [B_tile, HW, D_vit]
        del vit_out

        B_tile = feats.shape[0]
        h = w = int(feats.shape[1] ** 0.5)   # 32 for 448px tile
        feats_hw  = feats.reshape(B_tile, h, w, -1)
        shuffled  = model.pixel_shuffle(feats_hw, scale_factor=model.downsample_ratio)
        shuffled  = shuffled.reshape(B_tile, -1, shuffled.shape[-1])
        tile_tokens = model.mlp1(shuffled)   # [B_tile, num_image_token, D_llm]

        del feats, feats_hw, shuffled

    return tile_tokens.to(dtype)


# ──────────────────────────────────────────────────────────────────────────────
# Online KV 主函式
# ──────────────────────────────────────────────────────────────────────────────

def run_internvl_online_kv_stream(
    model,
    tokenizer,
    pixel_values: torch.Tensor,     # [N_tiles, 3, 448, 448]
    question: str,
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,             # 每次送入 ViT 的 tile 數
    chunk_size: int = 1024,          # LLM Chunked Prefill 的 token 數
):
    # ──────────────────────────────────────────────────────────────────
    # Step 1 : 建構 Prompt，設定 img_context_token_id
    # ──────────────────────────────────────────────────────────────────
    num_image_tiles = pixel_values.shape[0]
    total_vision_tokens = model.num_image_token * num_image_tiles
    
    print("\n[Step 1] Build prompt ...")
    prompt = build_prompt(tokenizer, model, question, num_image_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
    # print(f"text_before: {text_before}")
    # print(f"text_after: {text_after}")

    print(f"  ├─> num_image_token per tile : {model.num_image_token}")
    print(f"  ├─> num_image_tiles          : {num_image_tiles}")
    print(f"  └─> total vision tokens      : {total_vision_tokens}")

    past_key_values = None

    # ──────────────────────────────────────────────────────────────
    # Step 2~4 合併前的準備：先把 <img> 之前的文字 prefill 進去
    # （這段量很小，不影響峰值，保留獨立階段即可）
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 2+3+4 fused] text_before prefill ...")
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
        del out, embeds_before, attn_mask, ids_before
        torch.cuda.empty_cache()
        
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ├─> text_before injected, peak alloc={mem:.2f} GB")

    # ──────────────────────────────────────────────────────────────
    # 核心串流迴圈：per-tile ViT → projector → 累積 buffer → 滿了就 LLM prefill
    # ──────────────────────────────────────────────────────────────
    print(f"\n[Streaming Loop] vit_batch={vit_batch}, chunk_size={chunk_size} ...")

    token_buffer = []          # list of [B_tile, num_image_token, D_llm]
    buffered_token_count = 0
    tokens_done = 0             # 已經真正 prefill 進 LLM 的 vision token 累計數

    def flush_buffer():
        """把 buffer 裡累積的 vision token 一次性丟進 LLM 做 chunked prefill，並清空 buffer。"""
        nonlocal past_key_values, token_buffer, buffered_token_count, tokens_done

        if not token_buffer:
            return
        
        vision_chunk = torch.cat(token_buffer, dim=1)  # [1, buffered_token_count, D_llm]
        chunk_len = vision_chunk.shape[1]

        past_seq_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        attn_mask = torch.ones((1, past_seq_len + chunk_len), dtype=torch.long, device=DEVICE)
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
        tokens_done += chunk_len

        del out, vision_chunk, attn_mask, position_ids
        token_buffer = []
        buffered_token_count = 0
 
        torch.cuda.empty_cache()
        mem = torch.cuda.memory_allocated() / 1e9
        print(f"  ├─> [LLM flush] {tokens_done:05d}/{total_vision_tokens} vision tokens "
              f"injected, alloc={mem:.2f} GB")

    for i in range(0, num_image_tiles, vit_batch):
        tile_chunk = pixel_values[i : i + vit_batch].to(DEVICE, dtype=dtype)

        # ── ViT + projector（單一小批 tile）──
        tile_tokens = encode_tile(model, tile_chunk, dtype=dtype)   # [B_tile, num_image_token, D_llm]
        del tile_chunk
        torch.cuda.empty_cache()

        # 攤平 batch 維度成序列維度： [B_tile, num_image_token, D] -> [1, B_tile*num_image_token, D]
        B_tile = tile_tokens.shape[0]
        tile_tokens = tile_tokens.reshape(1, B_tile * model.num_image_token, -1)

        token_buffer.append(tile_tokens)
        buffered_token_count += tile_tokens.shape[1]

        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ├─> [ViT+Proj] tile {i}~{min(i+vit_batch, num_image_tiles)}/{num_image_tiles} done, "
              f"buffered={buffered_token_count} tokens, peak alloc={mem:.2f} GB")
        
        # ── buffer 湊滿 chunk_size，立刻觸發 LLM chunked prefill ──
        if buffered_token_count >= chunk_size:
            flush_buffer()
    
    # 迴圈跑完，buffer 裡若還有剩餘（不足一個 chunk_size），補做最後一次 prefill
    flush_buffer()
 
    print("  └─> Vision KV cache built (fully streamed)")
    print(f"Peak CUDA memory allocated: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    past_seq_len = past_key_values.get_seq_length()
    
    # ──────────────────────────────────────────────────────────────
    # Step 5 : 文字後半 Prefill + 自迴歸解碼（與分階段版相同，無需串流化）
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 5] Text Prefill + Autoregressive Decode ...")
    ids_after = tokenizer(text_after, return_tensors="pt").to(DEVICE)
    ids_after_ids = ids_after.input_ids
    text_len = ids_after_ids.shape[1]
    
    attn_mask = torch.ones((1, past_seq_len + text_len), dtype=torch.long, device=DEVICE)
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

    past_key_values   = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    next_token        = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
    del out, attn_mask, position_ids
 
    eos_token_id = tokenizer.eos_token_id
    print("=" * 60)
    answer = tokenizer.decode(next_token[0])

    for step in range(MAX_NEW_TOKENS):
        past_seq_len = past_key_values.get_seq_length()
        attn_mask = torch.ones((1, past_seq_len + 1), dtype=torch.long, device=DEVICE)
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
 
        answer += tokenizer.decode([token_id])
        del out, next_token_logits, attn_mask, position_ids
 
        if step % 50 == 0:
            torch.cuda.empty_cache()
 
    print("=" * 60)
    del past_key_values
    torch.cuda.empty_cache()
    
    return answer


def run_internvl_online_kv_stream_static(
    model,
    tokenizer,
    pixel_values: torch.Tensor,     # [num_image_tiles, 3, 448, 448]
    question: str,
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,             # 每次送入 ViT 的 tile 數
    chunk_size: int = 1024,          # LLM Chunked Prefill 的 token 數
):
    num_image_tiles = pixel_values.shape[0]
    total_vision_tokens = model.num_image_token * num_image_tiles
    
    print("\n[Step 1] Build prompt ...")
    prompt = build_prompt(tokenizer, model, question, num_image_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
    # print(f"text_before: {text_before}")
    # print(f"text_after: {text_after}")

    print(f"  ├─> num_image_token per tile : {model.num_image_token}")
    print(f"  ├─> num_image_tiles          : {num_image_tiles}")
    print(f"  └─> total vision tokens      : {total_vision_tokens}")

    # ── 先 tokenize text_before / text_after，才能算出 max_cache_len ──
    ids_before = tokenizer(text_before, return_tensors="pt").to(DEVICE) if text_before else None
    ids_after  = tokenizer(text_after, return_tensors="pt").to(DEVICE) if text_after else None

    len_before = ids_before.input_ids.shape[1] if ids_before is not None else 0
    len_after  = ids_after.input_ids.shape[1] if ids_after is not None else 0
    max_cache_len = len_before + total_vision_tokens + len_after + MAX_NEW_TOKENS + 8  # 留一點餘裕
    # print(f"len_before:    {len_before}")
    # print(f"len_after:     {len_after}")
    # print(f"max_cache_len: {max_cache_len}")

    # ── 關鍵改動：用 StaticCache 取代預設 DynamicCache ──
    past_key_values = StaticCache(
        config=model.language_model.config,
        max_batch_size=1,
        max_cache_len=max_cache_len,
        device=DEVICE,
        dtype=dtype,
    )

    # 關鍵修正：全長 attention_mask，1 代表「這個位置已經寫入有效 KV」
    full_attn_mask = torch.zeros((1, max_cache_len), dtype=torch.long, device=DEVICE)
    # 預先配置好 decode 階段會重複使用的固定大小 buffer，全部原地更新，不再逐步 new
    cache_pos_buf   = torch.zeros((max_cache_len,), dtype=torch.long, device=DEVICE)
    position_scalar = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)   # decode 用單 token position
    next_token_buf  = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)   # decode 用單 token id


    def step_prefill(inputs_embeds=None, input_ids=None, n_tokens=None):
        nonlocal past_key_values, full_attn_mask

        start = past_key_values.get_seq_length()
        # 用切片視圖而非重新 arange，減少一次配置
        cache_pos_buf[start:start + n_tokens].copy_(
            torch.arange(start, start + n_tokens, device=DEVICE, dtype=torch.long)
        )
        cache_pos = cache_pos_buf[start:start + n_tokens]
        full_attn_mask[:, start:start + n_tokens] = 1

        kwargs = dict(
            past_key_values=past_key_values,
            cache_position=cache_pos,
            attention_mask=full_attn_mask,
            # attention_mask=full_attn_mask[:, : start+n_tokens],
            use_cache=True,
            return_dict=True,
        )
        if inputs_embeds is not None:
            kwargs["inputs_embeds"] = inputs_embeds
        else:
            kwargs["input_ids"] = input_ids

        with torch.no_grad():
            out = model.language_model(**kwargs)
        return out

    # ── text_before ──
    print("\n[Step 2+3+4 fused] text_before prefill ...")
    if text_before:
        embeds_before = model.language_model.get_input_embeddings()(ids_before.input_ids)
        out = step_prefill(inputs_embeds=embeds_before, n_tokens=embeds_before.shape[1])
        del out, embeds_before
        print(f"  ├─> text_before injected, peak alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    token_buffer, buffered_token_count, tokens_done = [], 0, 0

    # ── streaming ViT + projector + chunked prefill ──
    print(f"\n[Streaming Loop] vit_batch={vit_batch}, chunk_size={chunk_size} ...")

    def flush_buffer():
        nonlocal token_buffer, buffered_token_count, tokens_done
        if not token_buffer:
            return
        vision_chunk = torch.cat(token_buffer, dim=1)
        chunk_len = vision_chunk.shape[1]

        out = step_prefill(inputs_embeds=vision_chunk, n_tokens=chunk_len)
        tokens_done += chunk_len
        del out, vision_chunk

        token_buffer.clear()
        buffered_token_count = 0

        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ├─> [LLM flush] {tokens_done:05d}/{total_vision_tokens} tokens, peak alloc={mem:.2f} GB")

    for i in range(0, num_image_tiles, vit_batch):
        tile_chunk = pixel_values[i:i + vit_batch].to(DEVICE, dtype=dtype)
        tile_tokens = encode_tile(model, tile_chunk, dtype=dtype)
        del tile_chunk
        
        B_tile = tile_tokens.shape[0]
        tile_tokens = tile_tokens.reshape(1, B_tile * model.num_image_token, -1)
        token_buffer.append(tile_tokens)
        buffered_token_count += tile_tokens.shape[1]
        
        if buffered_token_count >= chunk_size:
            flush_buffer()
    
    flush_buffer()
    print("  └─> Vision KV cache built (StaticCache, in-place)")

    # ── text_after prefill ──
    print("\n[Step 5] Text Prefill + Autoregressive Decode ...")
    out = step_prefill(input_ids=ids_after.input_ids, n_tokens=len_after)
    next_token_logits = out.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    del out, next_token_logits

    eos_token_id = tokenizer.eos_token_id
    answer = tokenizer.decode(next_token[0])

    # ★★★ decode loop：真正的高頻熱區，這裡才是要修的地方 ★★★
    for step in range(MAX_NEW_TOKENS):
        start = past_key_values.get_seq_length()

        # 原地更新 cache_position，不再每步 arange 配置新 tensor
        position_scalar.fill_(start)
        cache_pos_buf[start:start + 1].copy_(position_scalar.squeeze(0))
        cache_pos = cache_pos_buf[start:start + 1]

        full_attn_mask[:, start:start + 1] = 1

        # 原地更新 next_token 的 buffer，而不是每步用 unsqueeze 產生新 tensor
        next_token_buf.copy_(next_token)

        with torch.no_grad():
            out = model.language_model(
                input_ids=next_token_buf,
                cache_position=cache_pos,
                attention_mask=full_attn_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        next_token_logits = out.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        token_id = next_token.item()   # 這裡無法避免一次 GPU->CPU sync，但只有 1 個 scalar，成本很低
        if token_id == eos_token_id:
            del out, next_token_logits
            break
        answer += tokenizer.decode([token_id])
        del out, next_token_logits

    del past_key_values
    return answer

# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num",     type=int, default=48,  help="max dynamic tiles")
    parser.add_argument("--vit_batch",   type=int, default=4,   help="ViT micro-batch size")
    parser.add_argument("--chunk_size",  type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--batch_size",  type=int, default=1,   help="Image batch size")
    args = parser.parse_args()

    image = "4000x6000.jpg"
    question = "What is shown in this image in extreme detail?"

    dtype = torch.bfloat16

    torch.cuda.synchronize()
    t0 = time.time()

    # ── 1. 載入模型 ──
    tokenizer, model = build_model(args.model_name, dtype=dtype)
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"downsample_ratio         : {model.downsample_ratio}")
    print(f"select_layer             : {model.select_layer}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── 2. 影像前處理 ──
    pixel_values = load_image_tiles(image, input_size=448, max_num=args.max_num)
    print(f"Image loaded, pixel_values={pixel_values.shape}, batch_size={args.batch_size}")

    # ── 3. 批次 generate（同一張圖重複 batch_size 次） ──
    print(f"\n[Baseline] Running batch_chat() ...")
    ref_answers = generate_answer_standard(model, tokenizer, pixel_values, question, args.batch_size)
    print("\n[Baseline Answer] The first response:")
    print(ref_answers[0])
    print(f"Peak CUDA alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ── 4. Online KV Pipeline ──
    # answer = run_internvl_online_kv_stream(
    #     model=model,
    #     tokenizer=tokenizer,
    #     pixel_values=pixel_values,
    #     question=question,
    #     dtype=dtype,
    #     vit_batch=args.vit_batch,
    #     chunk_size=args.chunk_size,
    # )

    # model.language_model.config._attn_implementation = "sdpa"
    # answer = run_internvl_online_kv_stream_static(
    #     model=model,
    #     tokenizer=tokenizer,
    #     pixel_values=pixel_values,
    #     question=question,
    #     dtype=dtype,
    #     vit_batch=args.vit_batch,
    #     chunk_size=args.chunk_size,
    # )

    # print("\n[Online KV Answer]")
    # print(answer)
    # print(f"Peak CUDA alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # torch.cuda.synchronize()
    # elapsed = time.time() - t0
    # print(f"\nTotal time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
