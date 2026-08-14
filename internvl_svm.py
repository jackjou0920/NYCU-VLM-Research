# ══════════════════════════════════════════════════════════════════════════════
# Streaming Memory Bank  (固定 budget)
#   + Importance-aware Eviction
#   + Progressive Merge
#   + Online KV Construction
# ══════════════════════════════════════════════════════════════════════════════
import glob
import os
import gc
import time
import argparse
import json
import torch
from internvl_preprocess import measure_peak_memory, build_model, load_image_tiles
from internvl_memory_bank import TileStreamingMemoryBank


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
# 整合進你的 online KV pipeline：
#   for each sample -> 每張圖有自己獨立的 memory bank
#   tile 進來 -> encode_tile -> bank.add_tile -> (budget 自動守住)
#   全部 tile 處理完 -> bank.finalize() 拿到 <= budget 的最終 vision token
#   -> 跟之前一樣做 padding-to-max（但這次 max 是被 budget 封頂過的，不會再隨
#      tile 數暴增）-> chunked prefill 進 LLM（Online KV Builder，沿用你原本邏輯）
# ──────────────────────────────────────────────────────────────────────────────
def run_online_kv_with_memory_bank(
    model,
    tokenizer,
    pixel_values_list,        # list[Tensor]，每個 [N_tiles_b, 3, 448, 448]
    questions,
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,
    chunk_size: int = 1024,
    budget: int = 1024,       # ← 固定 memory budget（每張圖各自的 vision token 上限）
    score_fn: str = "l2_norm",   # "l2_norm" | "info_density" | "random"
    merge_mode: str = "evict",   # "none" | "evict"
):
    B = len(pixel_values_list)
    assert B == len(questions)

    num_tiles_list = [pv.shape[0] for pv in pixel_values_list]
    num_image_token = model.num_image_token  # num_image_token=256
 
    print(f"\n[Memory Bank] budget={budget}  score_fn={score_fn}  mode={merge_mode}")
    print(f"  tiles per image (before compression): {num_tiles_list}")
    print(f"  raw vision tokens per image          : "
          f"{[n * num_image_token for n in num_tiles_list]}")

    # ── Step 1 : prompt（先用固定的 budget 當作 num_image_tiles 的替代值來建 prompt，
    #             因為最終灌進去的 vision token 數就是 budget 封頂後的結果）──
    text_before_list, text_after_list = [], []
    for question in questions:
        # 用 budget/num_image_token 換算「等效 tile 數」去 build_prompt，
        # 確保 prompt 裡 <IMG_CONTEXT> 的數量跟最終真正塞進去的 token 數一致
        equiv_tiles = -(-budget // num_image_token)  # ceil division
        print(f"[question] {question} -> equiv_tiles={equiv_tiles}")
        
        prompt = build_prompt(tokenizer, model, question, equiv_tiles)
        text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
        text_before_list.append(text_before)
        text_after_list.append(text_after)

    # ── Step 2 : text_before prefill ──
    tok_before = tokenizer(text_before_list, return_tensors="pt", padding=True, padding_side="right")
    ids_before = tok_before.input_ids.to(DEVICE)          # [B, L_before]
    mask_before = tok_before.attention_mask.to(DEVICE)    # [B, L_before]，1=真實 token, 0=padding
    embeds_before = model.language_model.get_input_embeddings()(ids_before)

    past_key_values = None
    with torch.no_grad():
        out = model.language_model(
            inputs_embeds=embeds_before, attention_mask=mask_before,
            past_key_values=past_key_values, use_cache=True, return_dict=True,
        )
    past_key_values = out.past_key_values
    running_mask = mask_before
    del out, embeds_before, ids_before
    torch.cuda.empty_cache()

    # ── Step 3 : per-tile ViT + 各自的 Streaming Memory Bank ──
    # 攤平：把所有圖片的真實 tile 串成一條 flat tensor，並記錄每個 tile 屬於哪張圖
    flat_tiles = torch.cat(pixel_values_list, dim=0)  # [sum(N_tiles_b), 3, 448, 448]
    owner = []                                        # owner[k] = 第 k 個 flat tile 屬於哪個 batch index
    for b, n in enumerate(num_tiles_list):
        owner.extend([b] * n)
    
    D_llm = model.mlp1[-1].out_features

    # ── 新增：算出每個 question 的 embedding（跟 vision token 同一個 LLM embedding space）──
    with torch.no_grad():
        q_tok = tokenizer(questions, return_tensors="pt", padding=True).to(DEVICE)
        q_embeds_all = model.language_model.get_input_embeddings()(q_tok.input_ids)  # [B, L, D_llm]
        q_mask = q_tok.attention_mask.unsqueeze(-1).float()                          # [B, L, 1]
        # mean-pool，忽略 padding
        question_embeds = (q_embeds_all * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-6)
        question_embeds = question_embeds.to(dtype)  # [B, D_llm]

    banks = [
        TileStreamingMemoryBank(
            capacity=budget, dim=D_llm, device=DEVICE, dtype=dtype,
            score_fn=score_fn, mode=merge_mode,
            question_embed=question_embeds[b],   # ← 傳入真正的向量，不再是空字串
        ) for b in range(B)
    ]

    for i in range(0, flat_tiles.shape[0], vit_batch):
        chunk = flat_tiles[i:i + vit_batch].to(DEVICE, dtype=dtype)
        owner_chunk = owner[i:i + vit_batch]
 
        tile_tokens = encode_tile(model, chunk, dtype=dtype)  # [b_t, num_image_token, D]
        del chunk
        torch.cuda.empty_cache()
 
        for j, b in enumerate(owner_chunk):
            # 這一個 tile 的 num_image_token 個 token，一起丟進對應樣本的 memory bank
            banks[b].add_tile(tile_tokens[j])   # [num_image_token, D]
 
        mem = torch.cuda.max_memory_allocated() / 1e9
        print(f"  ├─> [ViT+Bank] flat tile {i}~{min(i+vit_batch, flat_tiles.shape[0])}"
              f"/{flat_tiles.shape[0]} done, peak alloc={mem:.2f} GB")
            #   f"bank sizes={[bk.tokens.shape[0] for bk in banks]}")
 
    del flat_tiles

    # ── 驗證 budget 真的被守住（Phase 1 的核心證明）──
    print("\n[Memory Bank Validation] final size for each image vs budget:")
    finalized, all_stats = [], []
    for b in range(B):
        toks, stats = banks[b].finalize()
        finalized.append(toks)
        all_stats.append(stats)
        print(f"  image {b}: raw={stats['total_seen']:5d} -> final={stats['final_size']:5d} "
              f"(budget={budget}, compression={stats['compression_ratio']:.2f}x, "
              f"dropped/merged={stats['total_dropped']})")
        assert stats["final_size"] <= budget, "The budget has been violated; check the add_tile logic"

    # ── padding 到 batch 內最大實際長度（現在被 budget 封頂，不會再暴增）──
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
    
    vision_tensor = torch.cat(padded_vision, dim=0)                # [B, max_len, D]
    vision_mask = torch.stack(vision_mask_rows, dim=0).to(DEVICE)  # [B, max_len]

    # ── Online KV Builder：跟之前一樣分 chunk 灌進 LLM ──
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
            out = model.language_model(
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

    # ── text_after + decode ──
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
            input_ids=ids_after, attention_mask=running_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, return_dict=True,
        )
    past_key_values   = out.past_key_values
    next_token_logits = out.logits[:, -1, :]
    next_token        = torch.argmax(next_token_logits, dim=-1, keepdim=True)
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
                input_ids=next_token, attention_mask=running_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=True, return_dict=True,
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
 
    return answers, all_stats


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def save_results_incremental(output_path: str, results: dict):
    """每處理完一張圖就整份重寫一次 JSON。圖片數量通常不多（十幾二十張），
    整份重寫的成本可忽略，但換來的是「跑到一半 OOM 也不會流失前面的結果」。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num",     type=int, default=48,  help="max dynamic tiles")
    parser.add_argument("--vit_batch",   type=int, default=4,   help="ViT micro-batch size")
    parser.add_argument("--chunk_size",  type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--batch_size",  type=int, default=1,   help="Image batch size")
    parser.add_argument("--budget",      type=int, default=1024, help="The maximum number of vision tokens per image")
    parser.add_argument(
        "--score_fn", type=str, default="info_density",
        choices=["l2_norm", "info_density", "random"],
    )
    parser.add_argument(
        "--merge_mode", type=str, default="evict",
        choices=["fifo", "evict"],
    )
    parser.add_argument("--image",       type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results.json")
    parser.add_argument("--run_stream", action="store_true", help="run online KV pipeline")
    args = parser.parse_args()

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

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    # ── 2. 找出所有圖片路徑（先只收集路徑，不要一次把全部圖片前處理/載進記憶體）──
    if os.path.isdir(args.image):
        image_paths = sorted(sum(
            [glob.glob(os.path.join(args.image, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []
        ))
        if not image_paths:
            raise FileNotFoundError(f"No images in {args.image}")
    else:
        image_paths = [args.image] * args.batch_size
    print(f"Found {len(image_paths)} image(s) to process (batch_size={args.batch_size})")

    # ── 3. 讀取舊的 output_results.json（如果存在），支援中斷後繼續跑 ──
    if os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        if "references" not in output_results: output_results["references"] = []
        if "candidates" not in output_results: output_results["candidates"] = {}
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', {}))} images already done)")
    else:
        output_results = {"references": [], "candidates": {}}

    tag = f"budget={args.budget}_{args.merge_mode}_{args.score_fn}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(image_paths):
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        print(f"\nAll done. Total time: {elapsed:.2f} s")
        return

    # ── 4. 逐張（或依 batch_size 分小批）處理，避免一次把全部圖片攤在同個 batch ──
    for i in range(0, len(image_paths), args.batch_size):
        batch_paths = image_paths[i:i + args.batch_size]

        print(f"\n{'='*70}")
        print(f"[{i+1}~{i+len(batch_paths)}/{len(image_paths)}] Processing: {batch_paths}")
        print(f"{'='*70}")

        # 只在真正要處理這一小批時才做影像前處理，處理完就可以釋放
        pixel_values_list = [
            load_image_tiles(p, input_size=448, max_num=args.max_num) for p in batch_paths
        ]
        questions = [question] * len(batch_paths)
        print(f"  tiles per image = {[pv.shape[0] for pv in pixel_values_list]}")

        # 每張圖（或每個小批次）開始前重置記憶體統計，量測才不會被前一批污染
        torch.cuda.reset_peak_memory_stats()

        try:
            if len(output_results["references"]) < len(image_paths):
                # ── Baseline ──
                print(f"\n[Baseline] Running batch_chat() ...")
                with measure_peak_memory(f"baseline_{i}"):
                    ref_answers = generate_answer_standard(model, tokenizer, pixel_values_list, questions)
                    output_results["references"] += ref_answers
        
                print("\n[Baseline Answer]")
                for i, (path, answer) in enumerate(zip(batch_paths, ref_answers)):
                    print(f"\n{i} -> [{path}]\n{answer}")

            # ── (可選) Online KV Memory Bank，跟 baseline 用同一批圖片比較 ──
            if args.run_stream:
                # ── 4. Online KV Pipeline ──
                if tag not in output_results["candidates"]: output_results["candidates"][tag] = []
                
                print(f"\n[Online Stream] Running online kv stream with memory bank ...")
                with measure_peak_memory("online_kv_memory_bank"):
                    answers, all_stats = run_online_kv_with_memory_bank(
                        model, tokenizer, pixel_values_list, questions,
                        dtype=dtype,
                        vit_batch=args.vit_batch,
                        chunk_size=args.chunk_size,
                        budget=args.budget,
                        merge_mode=args.merge_mode,
                        score_fn=args.score_fn
                    )
                    output_results["candidates"][tag] += answers

                # print("\n[Online KV Answer]")
                # for i, (path, answer) in enumerate(zip(batch_paths, answers)):
                #     print(f"\n{i} -> [{path}]\n{answer}")

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on batch {batch_paths}: {e}")
            raise

        
        # ── 這一批處理完，立刻釋放，避免殘留累積到下一批 ──
        del pixel_values_list
        gc.collect()
        torch.cuda.empty_cache()
    
        # ── 每處理完一批就存檔一次，不用等全部 10 張都跑完 ──
        if args.save:
            save_results_incremental(args.output_json, output_results)
            print(f"\n[Saved] {args.output_json} updated "
                f"({len(output_results['references'])}/{len(image_paths)} images done)")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"\nAll done. Total time: {elapsed:.2f} s")

    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
