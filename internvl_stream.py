
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
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, StaticCache
from internvl_preprocess import measure_peak_memory, build_model, load_image_tiles


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150


# ──────────────────────────────────────────────────────────────────────────────
# 標準 generate 參考路徑（用於正確性比較）
# ──────────────────────────────────────────────────────────────────────────────

def generate_answer_standard(model, tokenizer, pixel_values_list, questions):
    """走官方 model.batch_chat() 路徑。batch_chat 本身就是為不同圖片/不同 tile 數設計的，
    不需要 padding：把每張圖的 tiles 直接沿 batch 維度 cat 起來，用 num_patches_list
    告訴模型每張圖各自佔幾個 tile 即可。"""
 
    assert len(pixel_values_list) == len(questions)
 
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    num_patches_list = [pv.shape[0] for pv in pixel_values_list]
    pixel_values_batch = torch.cat(pixel_values_list, dim=0).to(DEVICE, dtype=model.dtype)
    questions_fmt = [f'<image>\n{q}' for q in questions]

    responses = model.batch_chat(
        tokenizer,
        pixel_values_batch,
        num_patches_list=num_patches_list,
        questions=questions_fmt,
        generation_config=generation_config,
    )
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
    pixel_values_list,          # list[Tensor]，長度=B，每個是 [N_tiles_b, 3, 448, 448]，N_tiles_b 可不同
    questions,                  # list[str]，長度=B（可以每張圖不同問題，也可以重複同一句）
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,         # 這裡是「攤平後」所有圖片 tile 一起算的 micro-batch 大小
    chunk_size: int = 1024,     # LLM Chunked Prefill 的 token 數
):
    B = len(pixel_values_list)
    assert B == len(questions)

    num_tiles_list = [pv.shape[0] for pv in pixel_values_list]
    max_tiles = max(num_tiles_list)
    num_image_token = model.num_image_token
    max_vision_len = max_tiles * num_image_token   # batch 內 vision token 對齊長度

    # ──────────────────────────────────────────────────────────────────
    # Step 1 : 建構 Prompt，設定 img_context_token_id
    # ──────────────────────────────────────────────────────────────────
    print("\n[Step 1] Build prompts (per-image, questions can be different) ...")
    text_before_list, text_after_list = [], []
    for question, n_tiles in zip(questions, num_tiles_list):
        prompt = build_prompt(tokenizer, model, question, n_tiles)
        text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
        text_before_list.append(text_before)
        text_after_list.append(text_after)

    print(f"  ├─> tiles per image : {num_tiles_list}")
    print(f"  └─> padded to max_tiles={max_tiles} ({max_vision_len} vision tokens/sample)")
    
    # ──────────────────────────────────────────────────────────────
    # Step 2+3+4 fused : text_before prefill（用 tokenizer padding 對齊不同長度的前綴）
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 2+3+4 fused] text_before prefill (padded) ...")
    tok_before = tokenizer(text_before_list, return_tensors="pt", padding=True, padding_side="right")
    ids_before = tok_before.input_ids.to(DEVICE)          # [B, L_before]
    mask_before = tok_before.attention_mask.to(DEVICE)    # [B, L_before]，1=真實 token, 0=padding

    embeds_before = model.language_model.get_input_embeddings()(ids_before)

    past_key_values = None
    with torch.no_grad():
        out = model.language_model(
            inputs_embeds=embeds_before,
            attention_mask=mask_before,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    past_key_values = out.past_key_values
    running_mask = mask_before   # 之後每次 flush 都要把新 token 的 mask 接在後面
    del out, embeds_before, ids_before
    
    torch.cuda.empty_cache()
    print(f"  └─> text_before injected, peak alloc={torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    # ──────────────────────────────────────────────────────────────
    # 準備每張圖各自 padding 好的 vision token 全序列 + 對應 mask
    # 真實 tile 統一攤平做 ViT（省算力），算完再切回各自樣本、padding 到 max_vision_len
    # ──────────────────────────────────────────────────────────────
    print(f"\n[Streaming Loop] vit_batch={vit_batch}, chunk_size={chunk_size}, B={B} (不同圖) ...")
    
    # 攤平：把所有圖片的真實 tile 串成一條 flat tensor，並記錄每個 tile 屬於哪張圖
    flat_tiles = torch.cat(pixel_values_list, dim=0)          # [sum(N_tiles_b), 3, 448, 448]
    owner = []                                                 # owner[k] = 第 k 個 flat tile 屬於哪個 batch index
    for b, n in enumerate(num_tiles_list):
        owner.extend([b] * n)
 
    D = model.mlp1[-1].out_features if hasattr(model.mlp1[-1], "out_features") else None
    
    # 每個樣本一個 list，蒐集自己的 tile_tokens，最後再 pad
    per_sample_tokens = [[] for _ in range(B)]

    for i in range(0, flat_tiles.shape[0], vit_batch):
        chunk = flat_tiles[i:i + vit_batch].to(DEVICE, dtype=dtype)
        owner_chunk = owner[i:i + vit_batch]
 
        tile_tokens = encode_tile(model, chunk, dtype=dtype)   # [b_t, num_image_token, D]
        del chunk
        torch.cuda.empty_cache()
 
        for j, b in enumerate(owner_chunk):
            per_sample_tokens[b].append(tile_tokens[j:j+1])    # [1, num_image_token, D]
 
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ├─> [ViT+Proj] flat tile {i}~{min(i+vit_batch, flat_tiles.shape[0])}"
              f"/{flat_tiles.shape[0]} done, peak alloc={mem:.2f} GB")
 
    del flat_tiles
    
    # 組出每個樣本的 [1, N_tiles_b*num_image_token, D]，再 pad 到 [1, max_vision_len, D]
    padded_vision = []
    vision_mask_rows = []
    for b in range(B):
        toks = torch.cat(per_sample_tokens[b], dim=1)   # [1, N_tiles_b*num_image_token, D]
        real_len = toks.shape[1]
        pad_len = max_vision_len - real_len
        if pad_len > 0:
            pad_block = torch.zeros((1, pad_len, toks.shape[-1]), dtype=toks.dtype, device=toks.device)
            toks = torch.cat([toks, pad_block], dim=1)
        padded_vision.append(toks)
        vision_mask_rows.append(
            torch.cat([
                torch.ones(real_len, dtype=torch.long),
                torch.zeros(pad_len, dtype=torch.long),
            ])
        )
    vision_tensor = torch.cat(padded_vision, dim=0)              # [B, max_vision_len, D]
    vision_mask = torch.stack(vision_mask_rows, dim=0).to(DEVICE)  # [B, max_vision_len]
    del padded_vision, per_sample_tokens
    
    # ── 分 chunk 灌進 LLM（沿用 chunked prefill 的精神，只是這次資料已經對齊好了）──
    tokens_done = 0
    for i in range(0, max_vision_len, chunk_size):
        chunk = vision_tensor[:, i:i + chunk_size, :]           # [B, c, D]
        chunk_mask = vision_mask[:, i:i + chunk_size]            # [B, c]
        c = chunk.shape[1]
 
        past_seq_len = past_key_values.get_seq_length()
        running_mask = torch.cat([running_mask, chunk_mask], dim=1)
        position_ids = torch.arange(
            past_seq_len, past_seq_len + c, dtype=torch.long, device=DEVICE
        ).unsqueeze(0).expand(B, -1)
 
        with torch.no_grad():
            out = model.language_model(
                inputs_embeds=chunk,
                attention_mask=running_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values = out.past_key_values
        tokens_done += c
        del out, chunk, chunk_mask, position_ids
        torch.cuda.empty_cache()
 
        print(f"  ├─> [LLM flush] {tokens_done:05d}/{max_vision_len} vision tokens injected "
              f"(padded, B={B}), alloc={torch.cuda.memory_allocated()/1e9:.2f} GB")
 
    del vision_tensor, vision_mask
    print("  └─> Vision KV cache built (per-image padded)")
    
    # ──────────────────────────────────────────────────────────────
    # Step 5 : text_after prefill + 自迴歸解碼（padding + EOS mask）
    # ──────────────────────────────────────────────────────────────
    print("\n[Step 5] Text Prefill + Autoregressive Decode ...")
    tok_after = tokenizer(text_after_list, return_tensors="pt", padding=True, padding_side="right")
    ids_after = tok_after.input_ids.to(DEVICE)
    mask_after = tok_after.attention_mask.to(DEVICE)

    past_seq_len = past_key_values.get_seq_length()
    running_mask = torch.cat([running_mask, mask_after], dim=1)
    position_ids = torch.arange(
        past_seq_len, past_seq_len + ids_after.shape[1], dtype=torch.long, device=DEVICE
    ).unsqueeze(0).expand(B, -1)
 
    with torch.no_grad():
        out = model.language_model(
            input_ids=ids_after,
            attention_mask=running_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
    past_key_values   = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    next_token        = torch.argmax(next_token_logits, dim=-1, keepdim=True)   # [B, 1]
    del out, ids_after, mask_after, position_ids

    eos_token_id = tokenizer.eos_token_id
    answers = [tokenizer.decode(next_token[b]) for b in range(B)]
    finished = (next_token.squeeze(-1) == eos_token_id)

    for step in range(MAX_NEW_TOKENS):
        if finished.all():
            break
        past_seq_len = past_key_values.get_seq_length()
        running_mask = torch.cat(
            [running_mask, torch.ones((B, 1), dtype=torch.long, device=DEVICE)], dim=1
        )
        position_ids = torch.full((B, 1), past_seq_len, dtype=torch.long, device=DEVICE)
 
        with torch.no_grad():
            out = model.language_model(
                input_ids=next_token,
                attention_mask=running_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
        past_key_values   = out.past_key_values
        next_token_logits = out.logits[:, -1, :]
        next_token        = torch.argmax(next_token_logits, dim=-1, keepdim=True)
 
        just_finished = (next_token.squeeze(-1) == eos_token_id)
        for b in range(B):
            if not finished[b]:
                answers[b] += tokenizer.decode([next_token[b].item()])
        finished = finished | just_finished
        del out, next_token_logits, position_ids
 
        if step % 50 == 0:
            torch.cuda.empty_cache()
 
    del past_key_values
    torch.cuda.empty_cache()
    return answers   # list[str]，長度 = B，各自對應各自的圖片


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

    # ── 2. 影像前處理（同一張圖重複 batch_size 次）──
    image_paths = [image] * args.batch_size
    pixel_values_list = [load_image_tiles(p, input_size=448, max_num=args.max_num) for p in image_paths]
    questions = [question] * len(image_paths)
    print(f"Loaded {len(image_paths)} different images, "
          f"tiles per image = {[pv.shape[0] for pv in pixel_values_list]}")

    # # ── 3. 批次 generate ──
    # print(f"\n[Baseline] Running batch_chat() ...")
    # with measure_peak_memory("baseline"):
    #     ref_answers = generate_answer_standard(model, tokenizer, pixel_values_list, questions)
    # print("\n[Baseline Answer] The first response:")
    # print(ref_answers[0])

    # ── 4. Online KV Pipeline ──
    print(f"\n[Online Stream] Running online_kv_streaming ...")
    with measure_peak_memory("online_kv_stream"):
        answers = run_internvl_online_kv_stream(
            model, tokenizer, pixel_values_list, questions,
            dtype=dtype,
            vit_batch=args.vit_batch,
            chunk_size=args.chunk_size,
        )

    # # model.language_model.config._attn_implementation = "sdpa"
    # # answer = run_internvl_online_kv_stream_static(
    # #     model=model,
    # #     tokenizer=tokenizer,
    # #     pixel_values=pixel_values,
    # #     question=question,
    # #     dtype=dtype,
    # #     vit_batch=args.vit_batch,
    # #     chunk_size=args.chunk_size,
    # # )

    # print("\n[Online KV Answer]")
    # for i, (p, a) in enumerate(zip(image_paths, answers)):
    #     print(f"\n{i} -> [{p}]\n{a}")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"\nTotal time: {elapsed:.2f} s")


if __name__ == "__main__":
    main()
