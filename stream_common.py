"""
共用骨架：跟模型無關的 online KV streaming 主流程。
 
這裡的每一步都是原本兩支 script（internvl_svm.py / llava_ov_svm.py）裡
逐行相同的部分：
    Step A. text_before prefill
    Step B. padding-to-max-len 组装 vision tensor/mask
    Step C. Online KV Builder：分 chunk 灌进 LLM
    Step D. text_after + 自回归 decode loop
 
真正模型相关的部分（tile/patch → bank、baseline 生成、lm forward 细节）
全部透過 adapter（見 stream_adapters.py）注入，這支檔案完全不 import
transformers 的模型类别，也不 if isinstance(model, ...)。
 
【重要】llava_ov_svm.py 目前的流程已確認沒問題，這支共用骨架的行為是照抄它
逐行一樣的部分抽出來，语意上不改变原本任何一步的顺序或数学；LlavaOVAdapter
只是把原本函式包一层，不重写内部逻辑。
"""
from __future__ import annotations


import torch

from stream_adapters import StreamModelAdapter, DEVICE, mean_pool_question_embed


def pad_and_stack_vision(finalized: list[torch.Tensor]):
    """把每張圖各自 <= budget 的 vision token 序列 padding 到 batch 內最大長度。
    跟原本兩支 script 裡的邏輯逐行相同。"""
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
    return vision_tensor, vision_mask, max_len


def run_online_kv_with_memory_bank(
    model,
    adapter: StreamModelAdapter,
    pixel_values_list,          # list[Tensor]，語意由 adapter 決定（InternVL: tiles；LLaVA-OV: patches）
    questions: list[str],
    image_sizes_list=None,      # LLaVA-OV 需要，InternVL 传 None
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,
    chunk_size: int = 1024,
    budget: int = 1024,
    score_fn: str = "info_density",
    merge_mode: str = "evict",
    max_new_tokens: int = 300,
):
    B = len(pixel_values_list)
    assert B == len(questions)
    tokenizer = adapter.get_tokenizer()

    # ── Step A0：question embedding（算法一样，共用）──
    question_embeds = mean_pool_question_embed(
        adapter.get_input_embeddings(model), tokenizer, questions, dtype,
    )

    # ── Step A1：prompt 切分（各自 build_text_segments，回傳統一介面）──
    text_before_list, text_after_list = adapter.build_text_segments(questions)

    # ── Step A2：text_before prefill（共用）──
    tok_before = tokenizer(text_before_list, return_tensors="pt", padding=True, padding_side="right")
    ids_before = tok_before.input_ids.to(DEVICE)
    mask_before = tok_before.attention_mask.to(DEVICE)
    embeds_before = adapter.get_input_embeddings(model)(ids_before)
 
    out = adapter.lm_prefill(
        model, inputs_embeds=embeds_before, attention_mask=mask_before,
        past_key_values=None, use_cache=True, return_dict=True,
    )
    past_key_values = out.past_key_values
    running_mask = mask_before
    del out, embeds_before, ids_before
    torch.cuda.empty_cache()

    # ── Step B：模型專屬的 encode + bank（真正的分歧點，見各自 adapter）──
    # 注意：budget 合法性檢定（final_size <= budget 或 <= max(budget, tile_size)）
    # 已經在各自 adapter.encode_and_bank() 內部做過、語意跟原本兩支 script 完全
    # 一致（LLaVA-OV 的 base image 單獨一個 patch 可能撐破 budget，所以那邊用
    # max(budget, num_image_token)；InternVL 用嚴格 <= budget）。這裡不重複
    # 加一個通用 assert，因為兩邊"合法上限"的定義本來就不同，硬套一個共用
    # 斷言反而會寫出沒有實際檢查力的假斷言。
    finalized, all_stats = adapter.encode_and_bank(
        model, pixel_values_list, image_sizes_list,
        question_embeds, budget, score_fn, merge_mode, vit_batch, dtype,
    )

    # ── Step C：padding + Online KV chunked injection（共用）──
    vision_tensor, vision_mask, max_len = pad_and_stack_vision(finalized)
    del finalized

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
 
        out = adapter.lm_prefill(
            model, inputs_embeds=vchunk, attention_mask=running_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, return_dict=True,
        )
        past_key_values = out.past_key_values
        tokens_done += c
        del out, vchunk, mchunk, position_ids
        torch.cuda.empty_cache()
        print(f"  ├─> [LLM flush] {tokens_done}/{max_len} vision tokens injected, "
              f"alloc={torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")
 
    del vision_tensor, vision_mask

    # ── Step D：text_after + decode（共用，forward 细节交给 adapter.lm_decode_step）──
    tok_after = tokenizer(text_after_list, return_tensors="pt", padding=True, padding_side="right")
    ids_after = tok_after.input_ids.to(DEVICE)
    mask_after = tok_after.attention_mask.to(DEVICE)
 
    past_seq_len = past_key_values.get_seq_length()
    running_mask = torch.cat([running_mask, mask_after], dim=1)
    position_ids = torch.arange(
        past_seq_len, past_seq_len + ids_after.shape[1], dtype=torch.long, device=DEVICE
    ).unsqueeze(0).expand(B, -1)
 
    next_token_logits, past_key_values = adapter.lm_decode_step(
        model, input_ids=ids_after, attention_mask=running_mask,
        position_ids=position_ids, past_key_values=past_key_values,
        use_cache=True, return_dict=True,
    )
    next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
    del ids_after, mask_after, position_ids, next_token_logits
 
    eos_token_id = tokenizer.eos_token_id
    answers = [tokenizer.decode(next_token[b]) for b in range(B)]
    finished = (next_token.squeeze(-1) == eos_token_id)

    for step in range(max_new_tokens):
        if finished.all():
            break
        past_seq_len = past_key_values.get_seq_length()
        running_mask = torch.cat(
            [running_mask, torch.ones((B, 1), dtype=torch.long, device=DEVICE)], dim=1
        )
        position_ids = torch.full((B, 1), past_seq_len, dtype=torch.long, device=DEVICE)
 
        next_token_logits, past_key_values = adapter.lm_decode_step(
            model, input_ids=next_token, attention_mask=running_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            use_cache=True, return_dict=True,
        )
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
 
        just_finished = (next_token.squeeze(-1) == eos_token_id)
        for b in range(B):
            if not finished[b]:
                answers[b] += tokenizer.decode([next_token[b].item()])
        finished = finished | just_finished
        del next_token_logits, position_ids
        if step % 50 == 0:
            torch.cuda.empty_cache()
 
    del past_key_values
    torch.cuda.empty_cache()
 
    return adapter.clean_answer(questions, answers), all_stats
