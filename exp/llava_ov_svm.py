# ══════════════════════════════════════════════════════════════════════════════
# Streaming Memory Bank on LLaVA-OneVision-Qwen2  (grid-aware 版)
#   port 自 internvl_svm.py 的 run_online_kv_with_memory_bank()
#
# ── 跟 InternVL 版本的關鍵架構差異 ──────────────────────────────────────────
#
# 1. Tile 的定義：
#      InternVL : 每張圖動態切成 N 個 448x448 tile + 1 個 thumbnail（放最後），
#                 每個 tile 固定輸出 256 個 token。
#      LLaVA-OV : anyres 前處理把每張圖切成 N 個 "patch"：patch[0] 是整圖縮圖
#                 （base image，放最前面），patch[1:] 是 anyres 網格裁出的
#                 crop，依 row-major 排列。每個 patch 固定輸出
#                 (image_size // patch_size)**2 個 token（7b-ov-hf 預設是
#                 SigLIP 384px / patch14 → 27*27 = 729 token/patch）。
#      → 把每個 "patch" 當作 TileStreamingMemoryBank 的一個 "tile"，
#        num_image_token 從 256 換成 729，protect_position 用 "first"
#        （base image 永遠保留，對應 InternVL 保護 thumbnail 的精神）。
#
# 2. 官方 `vision_aspect_ratio="anyres_max_9"` 這個設定，代表模型從來沒有
#    在訓練時看過「超過 9 個 crop 直接攤平串接」這種輸入型態——crop 數只要超過
#    9，官方一定會用雙線性內插把整個網格強制壓回 ~6561 token 的固定上限，這個
#   「連續內插、每個角落都保留一點模糊資訊」的表示，跟「離散選幾個完整 crop、
#    其他角落完全空白」在分佈上是兩回事，即使 token 數接近，模型也認不得。
#
# 所以這版改弦易轍：**不再自己重刻 unpad/interpolate，直接呼叫官方
# `model.model.pack_image_features()`**，拿到跟官方 generate() 一模一樣、
# 保證 in-distribution 的 packed 序列（本來就會處理 unpad + 內插 + newline，
# 不管 crop 數多寡都在模型訓練時看過的分佈內）。
#
# Memory bank 的角色因此往後退一步：不再負責「決定哪些 crop 存活」，而是在
# 官方 pack 完、已經是正確表示之後，**以「攤平後的一列（含結尾的 newline，
# 長度 = curr_width+1）」為 tile 單位**，用 TileStreamingMemoryBank 做進一步
# 壓縮（只有 budget < 官方 pack 出來的長度時才需要）。這個「一列 = 一個
# tile」的抽象，跟 InternVL 的「一個 448x448 tile = 一個 tile」、上一版
# LLaVA-OV 的「一個 patch = 一個 tile」是同一個類別（TileStreamingMemoryBank）
# 的三種不同實例化，不需要為 LLaVA-OV 另外寫一個 eviction 演算法——這就是這裡
# 說的「通用化」：streaming eviction 這件事跟「tile 的內容從哪裡來」是解耦的。
#
# 跟官方 100% 對齊的部分：unpad、interpolate、newline 位置——全部直接呼叫官方
# 函式，不是照抄數學重寫。
# 跟官方不一樣、屬於我們自己壓縮策略的部分：budget < pack 出來的長度時，用
# score-based eviction 再砍掉一些「列」，這步官方沒有，是我們疊加上去、拿來
# 驗證 memory bank 本身有沒有用的部分。
# ══════════════════════════════════════════════════════════════════════════════

import os
import gc
import math
import time
import argparse
import json
import torch
from exp.llava_ov_preprocess import measure_peak_memory, build_model, load_image_patches
from exp.internvl_memory_bank import TileStreamingMemoryBank
from transformers.models.llava_onevision.modeling_llava_onevision import (
    get_anyres_image_grid_shape,
    unpad_image,
)
from exp.llava_ov_preprocess import measure_peak_memory, build_model, load_image_patches
from exp.internvl_memory_bank import TileStreamingMemoryBank


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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
# def build_prompt(tokenizer, question: str) -> str:
#     messages = [
#         {
#             "role": "user", 
#             "content": f"{IMAGE_TOKEN}\n{question}"
#         }
#     ]
#     print(messages)
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     return prompt


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


# ──────────────────────────────────────────────────────────────────────────────
# 整合進 online KV pipeline
# ──────────────────────────────────────────────────────────────────────────────
def run_online_kv_with_memory_bank(
    model,
    processor,
    pixel_values_list,        # list[Tensor]，每個 [N_patches_b, 3, H, W]，來自 load_image_patches
    image_sizes_list,         # list[Tensor([H, W])]，每個原圖尺寸，同樣來自 load_image_patches
    datasets,
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,
    chunk_size: int = 1024,
    budget: int = 1024,       # ← 固定 memory budget（每張圖各自的 vision token 上限）
    score_fn: str = "l2_norm",
    merge_mode: str = "evict",
):
    questions = [dataset["question"] for dataset in datasets]
    tokenizer = processor.tokenizer
    B = len(pixel_values_list)
    assert B == len(questions) == len(image_sizes_list)
 
    num_patches_list = [pv.shape[0] for pv in pixel_values_list]
    num_image_token = num_image_tokens_per_patch(model)   # 例如 729
    if budget < num_image_token:
        raise ValueError(
            f"budget={budget} 小於單一 patch 的 token 數 ({num_image_token})，"
            f"base image 本身就無法塞進這個 budget，這個實驗設定不可行。"
        )
 
    print(f"\n[Memory Bank] budget={budget}  score_fn={score_fn}  mode={merge_mode}")
    print(f"  patches per image (before compression) : {num_patches_list}")
    print(f"  raw vision tokens per image             : "
          f"{[n * num_image_token for n in num_patches_list]}")

    # ── Step 1 : prompt ──
    text_before_list, text_after_list = [], []
    for i, question in enumerate(questions):
        # prompt = build_prompt(tokenizer, question)
        prompt = build_prompt(processor, question)
        text_before, text_after = split_prompt_at_vision(prompt)
        text_before_list.append(text_before)
        text_after_list.append(text_after)

    # ── Step 2 : text_before prefill ──
    tok_before = tokenizer(text_before_list, return_tensors="pt", padding=True, padding_side="right")
    ids_before = tok_before.input_ids.to(DEVICE)
    mask_before = tok_before.attention_mask.to(DEVICE)
    embeds_before = model.get_input_embeddings()(ids_before)
 
    past_key_values = None
    with torch.no_grad():
        out = model.model.language_model(
            inputs_embeds=embeds_before, attention_mask=mask_before,
            past_key_values=past_key_values, use_cache=True, return_dict=True,
        )
    past_key_values = out.past_key_values
    running_mask = mask_before
    del out, embeds_before, ids_before
    torch.cuda.empty_cache()

    # ── Step 3 : per-patch SigLIP（micro-batch 串流，控制 ViT 階段的 peak memory）
    #    這一步跟 budget 大小、要不要 eviction 無關，永遠都要跑 ──
    flat_patches = torch.cat(pixel_values_list, dim=0)   # [sum(N_patches_b), 3, H, W]
    owner = []
    for b, n in enumerate(num_patches_list):
        owner.extend([b] * n)
 
    D_llm = model.config.text_config.hidden_size
    per_image_tokens = [[] for _ in range(B)]   # 依原始 patch 順序，收集每張圖各自的 patch tokens
 
    for i in range(0, flat_patches.shape[0], vit_batch):
        chunk = flat_patches[i:i + vit_batch].to(DEVICE, dtype=dtype)
        owner_chunk = owner[i:i + vit_batch]
 
        patch_tokens = encode_patch(model, chunk, dtype=dtype)   # [b_t, num_image_token, D]
        del chunk
        torch.cuda.empty_cache()
 
        for j, b in enumerate(owner_chunk):
            per_image_tokens[b].append(patch_tokens[j])
 
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ├─> [ViT] flat patch {i}~{min(i + vit_batch, flat_patches.shape[0])}"
              f"/{flat_patches.shape[0]} done, peak alloc={mem:.2f} GB")
 
    del flat_patches
    per_image_tokens = [torch.stack(toks, dim=0) for toks in per_image_tokens]  # 每個 [N_patches_b, num_image_token, D]

    # ── Step 4 : 官方 pack_image_features（unpad + 必要時內插 + image_newline）
    #    這一步保證結果 in-distribution，不管單張圖的 crop 數多寡 ──
    image_newline = model.model.image_newline   # [D_llm]
    with torch.no_grad():
        q_tok = tokenizer(questions, return_tensors="pt", padding=True).to(DEVICE)
        q_embeds_all = model.get_input_embeddings()(q_tok.input_ids)
        q_mask = q_tok.attention_mask.unsqueeze(-1).float()
        question_embeds = (q_embeds_all * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-6)
        question_embeds = question_embeds.to(dtype)
 
        packed_list, feature_lens = model.model.pack_image_features(
            per_image_tokens,
            torch.stack(image_sizes_list).to(DEVICE),
            image_newline=image_newline,
        )
    del per_image_tokens
 
    print("\n[Pack] official pack_image_features() output (unpad + interpolate + newline done):")
    for b in range(B):
        print(f"  image {b}: packed_len={packed_list[b].shape[0]}  (budget={budget})")

    # ── Step 5 : 只有在官方 pack 出來的長度還是超過 budget 時，才用
    #    TileStreamingMemoryBank 在「一列（含 newline）」的粒度上做進一步淘汰。
    #    Tile 的定義換了，但淘汰演算法（score_fn / merge_mode）完全沿用
    #    TileStreamingMemoryBank，跟 InternVL 共用同一份程式碼。──
    finalized, all_stats = [], []
    for b in range(B):
        packed_b = packed_list[b]
        if packed_b.shape[0] <= budget:
            finalized.append(packed_b)
            all_stats.append({
                "final_size": packed_b.shape[0], "total_seen": packed_b.shape[0],
                "total_dropped": 0, "compression_ratio": 1.0,
                "rows_kept": None, "rows_total": None,
            })
            continue
 
        if num_patches_list[b] <= 1:
            # 只有 base image、沒有 anyres 網格：packed_b 就是 base(+可能 1 個
            # trailing newline)，沒有「列」可以再切，budget 再小也無法進一步淘汰。
            finalized.append(packed_b)
            all_stats.append({
                "final_size": packed_b.shape[0], "total_seen": packed_b.shape[0],
                "total_dropped": 0, "compression_ratio": 1.0,
                "rows_kept": None, "rows_total": None,
            })
            continue

        grid_shape = get_anyres_image_grid_shape(
            image_sizes_list[b], model.config.image_grid_pinpoints, model.config.vision_config.image_size
        )
        num_rows, row_width = compute_packed_row_layout(model, image_sizes_list[b], grid_shape)
        base_tokens = packed_b[:num_image_token]
        grid_tokens = packed_b[num_image_token:]
        assert grid_tokens.shape[0] == num_rows * row_width, (
            f"row layout 算出來的長度 ({num_rows}x{row_width}={num_rows * row_width}) 跟官方 "
            f"pack_image_features 實際吐出來的長度 ({grid_tokens.shape[0]}) 對不上，"
            f"代表 compute_packed_row_layout 的數學跟官方版本不一致，需要重新核對。"
        )
        rows = grid_tokens.view(num_rows, row_width, D_llm)
 
        bank = TileStreamingMemoryBank(
            capacity=budget - num_image_token, dim=D_llm, device=DEVICE, dtype=dtype,
            score_fn=score_fn, mode=merge_mode,
            num_image_token=row_width,
            question_embed=question_embeds[b],
            protected_tiles=0,   # base image 已經在迴圈外處理，bank 只管網格列
        )
        for r in range(num_rows):
            bank.add_tile(rows[r])   # 評分含這一列的 newline token，權重很小可忽略
 
        final_grid_tokens, stats = bank.finalize()   # flat 模式：newline 已經內建在每個 tile 裡，不用再插
        finalized.append(torch.cat([base_tokens, final_grid_tokens], dim=0))
 
        # bank 只管網格列，final_size/total_seen 預設不含 base，這裡補回去，
        # 讓 log 顯示的是「這張圖真正餵給 LLM 的總 token 數」而不是只有網格部分。
        stats["final_size"] += base_tokens.shape[0]
        stats["total_seen"] += base_tokens.shape[0]
        stats["rows_kept"] = stats["grid_patches_kept"]
        stats["rows_total"] = num_rows
        all_stats.append(stats)

    print("\n[Memory Bank Validation] final size for each image vs budget:")
    for b in range(B):
        stats = all_stats[b]
        row_info = (f"rows_kept={stats['rows_kept']}/{stats['rows_total']}"
                    if stats["rows_kept"] is not None else "rows_kept=N/A（沒有網格可淘汰）")
        print(f"  image {b}: raw={stats['total_seen']:5d} -> final={stats['final_size']:5d}  "
              f"(budget={budget}, compression={stats['compression_ratio']:.2f}x)  {row_info}")
        assert stats["final_size"] <= max(budget, num_image_token), (
            "final_size 超過 budget，代表 eviction 換算出了問題，需要檢查"
        )
 
    del packed_list

    # ── padding 到 batch 內最大實際長度 ──
    max_len = max(t.shape[0] for t in finalized)
    padded_vision, vision_mask_rows = [], []
    for toks in finalized:
        real_len = toks.shape[0]
        pad_len = max_len - real_len
        if pad_len > 0:
            pad_block = torch.zeros((pad_len, toks.shape[-1]), dtype=toks.dtype, device=toks.device)
            toks = torch.cat([toks, pad_block], dim=0)
 
        padded_vision.append(toks.unsqueeze(0))
        vision_mask_rows.append(torch.cat([
            torch.ones(real_len, dtype=torch.long),
            torch.zeros(pad_len, dtype=torch.long),
        ]))
 
    vision_tensor = torch.cat(padded_vision, dim=0)
    vision_mask = torch.stack(vision_mask_rows, dim=0).to(DEVICE)
    del finalized
 
    # ── Online KV Builder：分 chunk 灌進 LLM ──
    tokens_done = 0
    for i in range(0, max_len, chunk_size):
        vchunk = vision_tensor[:, i:i + chunk_size, :]
        mchunk = vision_mask[:, i:i + chunk_size]
        c = vchunk.shape[1]
 
        past_seq_len = past_key_values.get_seq_length()
        running_mask = torch.cat([running_mask, mchunk], dim=1)
        position_ids = torch.arange(
            past_seq_len, past_seq_len + c, dtype=torch.long, device=DEVICE
        ).unsqueeze(0).expand(B, -1)
 
        with torch.no_grad():
            out = model.model.language_model(
                inputs_embeds=vchunk, attention_mask=running_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=True, return_dict=True,
            )
        past_key_values = out.past_key_values
        tokens_done += c
        del out, vchunk, mchunk, position_ids
        torch.cuda.empty_cache()
        print(f"  ├─> [LLM flush] {tokens_done}/{max_len} (memory-bank-compressed) "
              f"vision tokens injected, alloc={torch.cuda.memory_allocated()/1e9:.2f} GB")
 
    del vision_tensor, vision_mask
 
    # ── text_after + decode（language_model 是 base model，logits 要自己接 lm_head）──
    tok_after = tokenizer(text_after_list, return_tensors="pt", padding=True, padding_side="right")
    ids_after = tok_after.input_ids.to(DEVICE)
    mask_after = tok_after.attention_mask.to(DEVICE)
 
    past_seq_len = past_key_values.get_seq_length()
    running_mask = torch.cat([running_mask, mask_after], dim=1)
    position_ids = torch.arange(
        past_seq_len, past_seq_len + ids_after.shape[1], dtype=torch.long, device=DEVICE
    ).unsqueeze(0).expand(B, -1)
 
    with torch.no_grad():
        out = model.model.language_model(
            input_ids=ids_after, attention_mask=running_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, return_dict=True,
        )
        next_token_logits = model.lm_head(out.last_hidden_state[:, -1, :])
    past_key_values = out.past_key_values
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    del out, ids_after, mask_after, position_ids, next_token_logits
 
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
            out = model.model.language_model(
                input_ids=next_token, attention_mask=running_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=True, return_dict=True,
            )
            next_token_logits = model.lm_head(out.last_hidden_state[:, -1, :])
        past_key_values = out.past_key_values
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
 
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
    return clean_answer(datasets, answers), all_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
def save_results_incremental(output_path: str, results: dict):
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="llava-hf/llava-onevision-qwen2-7b-ov-hf")
    parser.add_argument("--vit_batch", type=int, default=4, help="ViT micro-batch size")
    parser.add_argument("--chunk_size", type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--batch_size", type=int, default=1, help="Image batch size")
    parser.add_argument("--budget", type=int, default=1024, help="The maximum number of vision tokens per image")
    parser.add_argument(
        "--score_fn", type=str, default="info_density",
        choices=["l2_norm", "info_density", "random"],
    )
    parser.add_argument(
        "--merge_mode", type=str, default="evict",
        choices=["fifo", "evict"],
    )
    parser.add_argument("--run_stream", action="store_true", help="run online KV pipeline")
    parser.add_argument("--run_standard", action="store_true",
                            help="run the official (uncompressed) HF generate() path to produce reference answers")

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_llava_ov.json")
    args = parser.parse_args()

    dtype = torch.bfloat16

    torch.cuda.synchronize()
    t0 = time.time()

    # ── 1. 載入模型 ──
    processor, model = build_model(args.model_name, dtype=dtype)
    print(f"num_image_token per patch : {num_image_tokens_per_patch(model)}")
    print(f"vision_feature_layer      : {model.config.vision_feature_layer}")
    print(f"vision_feature_select     : {model.config.vision_feature_select_strategy}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    # ── 2. 載入圖像與問題 ──
    if args.use_ds:
        from exp.llava_ov_preprocess import load_mmmu
        datasets = load_mmmu(num_image=args.num_images)
    else:
        from exp.llava_ov_preprocess import load_local
        datasets = load_local(args.image, args.batch_size, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process (batch_size={args.batch_size})")


    # ── 3. 讀取舊的 output_results.json（如果存在），支援中斷後繼續跑 ──
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        if "references" not in output_results: output_results["references"] = []
        if "candidates" not in output_results: output_results["candidates"] = {}
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', {}))} images already done)")
    else:
        output_results = {"references": [], "candidates": {}}

    tag = f"budget={args.budget}_{args.merge_mode}_{args.score_fn}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"\nAll done. Total time: {elapsed:.2f} s")
        return

    # ── 4. 逐批處理 ──
    for i in range(0, len(datasets), args.batch_size):
        batch_datasets = datasets[i:i + args.batch_size]

        print(f"\n{'='*70}")
        print(f"[{i+1}~{i+len(batch_datasets)}/{len(datasets)}] Processing Batch...")
        print(f"{'='*70}")
 
        torch.cuda.reset_peak_memory_stats()

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                print(f"\n[Standard] Running official (uncompressed) generate() path ...")
                with measure_peak_memory("llava_ov_standard_generate"):
                    ref_answers = generate_answer_standard(model, processor, batch_datasets)
                    output_results["references"] += ref_answers
                
                # print("\n[Baseline Answer]")
                # for i, answer in enumerate(ref_answers):
                #     print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")


            if args.run_stream:
                if tag not in output_results["candidates"]: output_results["candidates"][tag] = []

                pixel_values_list, image_sizes_list = load_image_patches(processor, batch_datasets)
                print(f"  patches per image = {[pv.shape[0] for pv in pixel_values_list]}")

                print(f"\n[Online Stream] Running online kv stream with memory bank ...")
                with measure_peak_memory("llava_ov_online_kv_memory_bank"):
                    answers, all_stats = run_online_kv_with_memory_bank(
                        model, processor, pixel_values_list, image_sizes_list, batch_datasets,
                        dtype=dtype,
                        vit_batch=args.vit_batch,
                        chunk_size=args.chunk_size,
                        budget=args.budget,
                        merge_mode=args.merge_mode,
                        score_fn=args.score_fn,
                    )
                    output_results["candidates"][tag] += answers

                # print("\n[Online KV Answer]")
                # for i, answer in enumerate(answers):
                #     print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")

                del pixel_values_list, image_sizes_list

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on batch: {e}")
            raise
 
        gc.collect()
        torch.cuda.empty_cache()

        if args.save:
            save_results_incremental(args.output_json, output_results)
            print(f"\n[Saved] {args.output_json} updated "
                  f"({len(output_results['references'])}/{len(datasets)} images done)")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"\nAll done. Total time: {elapsed:.2f} s")
 
    if args.save:
        print(f"Results saved to {args.output_json}")



if __name__ == "__main__":
    main()
