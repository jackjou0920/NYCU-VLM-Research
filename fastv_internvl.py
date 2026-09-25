"""
獨立、不改動任何既有檔案的「FastV × InternVL」執行腳本。

參考：https://github.com/pkunlp-icler/FastV（實際讀了官方兩份 modeling_llama.py
fork 才寫這個檔案，剪枝『決策』本身逐行對照官方碼，見 fastv_core.py 開頭的詳細
說明與行號）。跟 sparsevlm_internvl.py 走同一條路：靠手動逐層呼叫 Qwen3Model
的 decoder layers，在指定層插入 fastv_core.py 的剪枝邏輯，讓視覺 token 真的在
LLM decoder 內部被剔除——只「讀取」internvl_stream.py / sparsevlm_internvl.py
既有的公用函式，完全沒有修改任何一行既有檔案。輸出的 JSON 格式跟
internvl_svm.py / sparsevlm_internvl.py 一致，可以直接丟給既有的
evaluate_anls.py / evaluate_hrbench.py / bench_metrics.py 用。

用法範例：
    python fastv_internvl.py \
        --image img_datasets/img01.jpg --budget 512 --agg_layer 2 \
        --run_standard --run_fastv --save

【跟官方原始碼的關鍵落差，如實記錄——這是本檔案唯一「補上官方沒做完」的部分】
  官方 repo 有兩份實作（見 fastv_core.py 開頭引用的行號）：
    (a) attention-mask 版：靠把被丟 token 的 attention 設 -inf 等效模擬剪枝，
        tensor shape 不變，天生相容 KV cache + model.generate() 多步 decode，
        但『不會』實際縮短序列，量不出真正的 peak memory / KV cache 下降。
    (b) inplace 版：真的縮短 hidden_states/position_ids，才會有真正的記憶體/
        FLOPs 下降，但官方自己在 --fast-v-inplace 的 help 字串承認「no
        supported kv cache yet」，官方腳本只拿它做單次 forward 的 latency
        量測，或 max_new_tokens=1 的單字母選擇題（AOKVQA），從未在官方 repo
        裡跑過『真正多個 token 的自迴歸生成』。
  這支腳本要跟你的 streaming 方法比較 peak memory / KV cache MB（不是只看
  accuracy），需要 (b) 的真實縮短效果，但也需要撐過完整的多 token 生成
  （HR-Bench/DocVQA 都不只吐一個字）。剪枝『決策』完全照抄官方（b）版的邏輯
  （fastv_core.py 的 fastv_select_keep_indices，逐行對照官方碼），連
  position_ids 也照官方（b）版保留剪枝前的絕對位置、不重新編號（見
  fastv_prefill() 裡的說明——這點會影響 RoPE 相位，跟 sparsevlm_internvl.py
  的做法不同，不能照搬）。但『剪枝之後怎麼讓不同層的 KV cache 各自維護正確
  長度、支援多步 decode』官方沒做——這裡借用 sparsevlm_internvl.py 已經驗證
  過的核心想法（DynamicCache 天生允許各層長度不同），但改寫成 fastv_decode_
  step()：因為 FastV 的 position_ids 保留絕對位置（不像 SparseVLM 每個
  stage 後重新從 0 編號），不能直接沿用 sparsevlm_decode_step() 用『每層
  cache 長度』反推下一個 token 位置的算法，細節見 fastv_decode_step() 的
  docstring。這是補齊官方承認的工程缺口，不是更改 FastV 的剪枝演算法本身。
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from internvl_stream import (
    MAX_NEW_TOKENS, build_prompt, clean_answer, generate_answer_standard,
    split_prompt_at_vision, encode_tile, build_model, load_image_tiles,
    DEVICE, measure_peak_memory, dedup_hrbench_open,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
)
# 這幾個 helper 跟演算法本身無關（attn_implementation 切換 / KV cache MB 估算 /
# eos token / 剪枝前的初始幾何），sparsevlm_internvl.py 已經寫好、驗證過，直接
# import 沿用，不重複一份。decode step『不』沿用 sparsevlm_decode_step()——
# 原因見 fastv_decode_step() 的 docstring：兩個方法的 position_ids 語意不同，
# 直接借用會算出錯誤的 RoPE 位置。
from sparsevlm_internvl import (
    _set_attn_implementation, _get_eos_token_id,
    estimate_kv_cache_mb, estimate_prompt_token_lengths,
    _segment_geometry,
)
from fastv_core import fastv_select_keep_indices


# ──────────────────────────────────────────────────────────────────────────
# Prefill：手動逐層跑，在 agg_layer 插入 FastV 的 one-shot 剪枝
# ──────────────────────────────────────────────────────────────────────────
def _layer_forward_capture_last_row(
    decoder_layer, hidden_states, position_ids, position_embeddings,
    past_key_values, layer_idx, cache_position,
):
    """手動重現 Qwen3DecoderLayer.forward() 的 pre-norm block，跟
    sparsevlm_internvl.py 的 sparsevlm_prefill() 同一招：attn_out 走真正的
    F.scaled_dot_product_attention 融合 kernel（不會材料化整個 [H,L,L]
    attention matrix），打分要用的 attention weights 另外手算一次——但這裡
    只算『最後一個 query 位置』那一列（[H, L]），比 sparsevlm_internvl.py
    只算文字 token 那幾列還更省，因為 FastV 官方碼本來就只用最後一個 token
    的 attention（見 fastv_core.py 的 fastv_select_keep_indices() docstring，
    逐行對應官方碼的 `[-1]`）。

    回傳 (new_hidden_states, last_row_attn[H, L])。
    """
    residual = hidden_states
    normed = decoder_layer.input_layernorm(hidden_states)

    self_attn = decoder_layer.self_attn
    bsz, q_len, _ = normed.shape
    hshape = (bsz, q_len, -1, self_attn.head_dim)
    query_states = self_attn.q_norm(self_attn.q_proj(normed).view(hshape)).transpose(1, 2)
    key_states = self_attn.k_norm(self_attn.k_proj(normed).view(hshape)).transpose(1, 2)
    value_states = self_attn.v_proj(normed).view(hshape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
    cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
    key_states, value_states = past_key_values.update(key_states, value_states, layer_idx, cache_kwargs)

    key_full = repeat_kv(key_states, self_attn.num_key_value_groups)
    value_full = repeat_kv(value_states, self_attn.num_key_value_groups)

    attn_out = F.scaled_dot_product_attention(
        query_states, key_full, value_full, attn_mask=None,
        is_causal=True, scale=self_attn.scaling,
    )
    attn_out = attn_out.transpose(1, 2).reshape(bsz, q_len, -1).contiguous()
    attn_out = self_attn.o_proj(attn_out)

    # 只對「序列裡最後一個 query 位置」算 attention weights：這個 query 對
    # 序列中每個位置（含自己）天然全部可見，causal mask 自動滿足，不需要
    # 額外套遮罩——對應官方碼 last_layer_attention_avg_last_tok 的 `[-1]`。
    last_query = query_states[:, :, -1:, :]                                     # [B, H, 1, d]
    last_logits = (last_query @ key_full.transpose(-2, -1)) * self_attn.scaling  # [B, H, 1, L]
    last_row_attn = last_logits.softmax(dim=-1).to(query_states.dtype)[0, :, 0, :]  # [H, L]

    hidden_states = residual + attn_out
    residual = hidden_states
    hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
    hidden_states = decoder_layer.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states, last_row_attn


def fastv_prefill(
    model, inputs_embeds: torch.Tensor, agg_layer: int, keep_count: int,
    vis_start: int, vis_len: int,
):
    """回傳 (hidden_states [1, L_final, D], past_key_values, final_vis_len, debug_stats)。

    跟 sparsevlm_internvl.py 的 sparsevlm_prefill() 同一招手動逐層呼叫
    Qwen3Model 的 decoder layers：FastV 的剪枝要在 LLM decoder 內部『特定
    一層』把序列變短，之後所有層都要沿用縮短後的長度，一般
    model.language_model(...) 整段呼叫做不到這個粒度。

    跟官方碼一致的層序安排（見 fastv_core.py 開頭引用的官方行號）：
      - layer_idx < agg_layer-1：正常跑（fast attn 實作，不算 attention weights）。
      - layer_idx == agg_layer-1：額外手算『最後一個 query 位置』的 attention，
        留給下一層剪枝用（對應官方碼 `if layer_idx == FASTV_k - 1:
        output_attentions = True`）。
      - layer_idx == agg_layer：用上一層算好的 attention 決定 keep_indexs，
        把 hidden_states 縮短後才餵進這一層（對應官方碼
        `elif layer_idx==FASTV_k:` 那個分支）。
      - layer_idx > agg_layer：正常跑，序列長度已經是縮短後的。
    """
    if agg_layer < 1:
        raise ValueError(
            "官方碼只在 AGG_LAYER>=1 時有明確定義的剪枝邏輯（用『上一層』的"
            "attention 決定 keep_indexs）；AGG_LAYER==0 官方碼在 token-mask 版本"
            "改用隨機 attention mask（modeling_llama.py 的 `else:` 分支，"
            "rand_image_attention_mask），沒有代表性，也没有拿去跑過任何一個"
            "官方 eval 腳本（rank_list/Ks 都固定 K=2），這裡不實作這個邊界情況，"
            "agg_layer 請用 >=1（論文與官方腳本預設都是 2）。"
        )

    lm_model = model.language_model.model  # Qwen3Model
    device = inputs_embeds.device
    past_key_values = DynamicCache(config=lm_model.config)

    hidden_states = inputs_embeds
    total_len = hidden_states.shape[1]
    position_ids, cache_position = _segment_geometry(hidden_states, device)
    causal_mask = create_causal_mask(
        config=lm_model.config, input_embeds=hidden_states, attention_mask=None,
        cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
    )
    position_embeddings = lm_model.rotary_emb(hidden_states, position_ids)

    captured_attn = None
    final_vis_len = vis_len
    debug_stats = {"vis_len_before": vis_len}

    for layer_idx, decoder_layer in enumerate(lm_model.layers):
        if layer_idx == agg_layer:
            keep_indices = fastv_select_keep_indices(
                captured_attn, vis_start, vis_len, total_len, keep_count, device,
            )
            hidden_states = hidden_states[:, keep_indices, :]
            final_vis_len = min(keep_count, vis_len)
            debug_stats["vis_len_after"] = final_vis_len
            debug_stats["dropped"] = vis_len - final_vis_len

            # position_ids 用 keep_indices 本身（剪枝前的絕對位置，被丟的位置
            # 中間會有間隙）——對應官方碼 `position_ids = keep_indexs.unsqueeze(0)`
            # （見 fastv_modeling_llama.py 第 1058 行），不是重新從 0 連續編號。
            # 這不是無關緊要的小細節：官方的「attention-mask 版」（用來測
            # accuracy 的那個版本，見 fastv_core.py 開頭說明）從頭到尾沒有縮短
            # 序列，被丟的 token 的位置天然還在，RoPE 用的是原始絕對位置；
            # 「inplace 版」（這裡用的版本）如果改成連續重編號，等於用了一組
            # 跟 mask 版不同的 RoPE 相位，會讓兩個官方版本在數學上不再等價
            # ——這正是官方拿 mask 版的 accuracy 數字去代表 inplace 版效果的
            # 前提（見 fastv_internvl.py 開頭的落差說明）。cache_position 沒有
            # 這個問題：DynamicCache 只是依呼叫順序 append，不看 cache_position
            # 的實際數值決定寫入位置，用連續 0..new_len-1 即可，跟官方碼
            # `cache_position[:new_seq_length]`（在 past_seen_tokens=0 時
            # 同樣等於 arange(0, new_seq_length)）行為一致。
            new_seq_length = hidden_states.shape[1]
            cache_position = torch.arange(new_seq_length, device=device)
            position_ids = keep_indices.unsqueeze(0)
            causal_mask = create_causal_mask(
                config=lm_model.config, input_embeds=hidden_states, attention_mask=None,
                cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
            )
            position_embeddings = lm_model.rotary_emb(hidden_states, position_ids)

            hidden_states = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=True,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )
        elif layer_idx == agg_layer - 1:
            hidden_states, captured_attn = _layer_forward_capture_last_row(
                decoder_layer, hidden_states, position_ids, position_embeddings,
                past_key_values, layer_idx, cache_position,
            )
        else:
            hidden_states = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=True,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )

    hidden_states = lm_model.norm(hidden_states)
    return hidden_states, past_key_values, final_vis_len, debug_stats


# ──────────────────────────────────────────────────────────────────────────
# Decode：不能沿用 sparsevlm_internvl.py 的 sparsevlm_decode_step()
# ──────────────────────────────────────────────────────────────────────────
def fastv_decode_step(model, hidden_states_1tok: torch.Tensor, past_key_values, next_position: int, device):
    """跟 sparsevlm_decode_step() 表面很像（都要逐層跑，因為剪枝層跟非剪枝層
    的 KV cache 長度不同），但 position_ids 的算法不能照搬：

    SparseVLM（sparsevlm_core.py 的 V1 設計）每個 pruning stage 後都把
    position_ids 從 0 重新編號，所以『這一層 cache 目前的長度』永遠等於
    『這一層目前用到的最大絕對位置 + 1』，sparsevlm_decode_step() 才能直接拿
    `past_key_values.get_seq_length(layer_idx)` 當下一個 token 的 position_id。

    FastV 官方碼（見 fastv_prefill() 裡 `position_ids = keep_indices.
    unsqueeze(0)` 那段說明）刻意『不』重新編號，保留剪枝前的絕對位置——
    代表剪枝層的 cache 長度雖然比較短，但序列裡最後一個 token（一定被保留，
    因為 keep_indices 永遠包含 image 區段之後的所有文字 token）的絕對位置
    仍然是 total_len-1，不是 new_seq_length-1。如果沿用
    sparsevlm_decode_step() 的算法，剪枝層會把下一個 token 的 position_id
    算成 new_seq_length（遠小於正確值 total_len），RoPE 相位跟非剪枝層對不
    起來，是一個會讓生成結果錯誤但不會報錯的 silent bug。

    修正：position_id 由呼叫端傳入一個『所有層共用』的絕對位置計數器
    （next_position，從 total_len 開始每步 +1，見 generate_fastv()），不再
    從個別層的 cache 長度反推。cache_position 維持跟 sparsevlm_decode_step()
    一樣用『這一層目前 cache 長度』——DynamicCache.update() 只依呼叫順序
    append，不看 cache_position 的實際數值決定寫入位置，這部分兩個方法可以
    共用同一招。
    """
    lm_model = model.language_model.model
    h = hidden_states_1tok
    position_ids = torch.tensor([[next_position]], device=device)
    for decoder_layer in lm_model.layers:
        layer_cache_len = past_key_values.get_seq_length(decoder_layer.self_attn.layer_idx)
        cache_position = torch.tensor([layer_cache_len], device=device)
        position_embeddings = lm_model.rotary_emb(h, position_ids)
        h = decoder_layer(
            h, attention_mask=None, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=True,
            cache_position=cache_position, position_embeddings=position_embeddings,
        )
    h = lm_model.norm(h)
    return model.language_model.lm_head(h)


# ──────────────────────────────────────────────────────────────────────────
# 頂層：build prompt → 灌 vision → fastv_prefill → greedy decode
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_fastv(
    model, tokenizer, pixel_values: torch.Tensor, question: str,
    budget: int, agg_layer: int = 2,
    vit_batch: int = 4, max_new_tokens: int = MAX_NEW_TOKENS,
):
    """budget 直接對應官方的 ATTENTION_RANK（保留幾個 image token），不像
    SparseVLM 的 λ 需要另外校準——FastV 一次到位，final_vis_tokens 恆等於
    min(budget, raw_vis_tokens)，沒有「自然壓到接近 budget」這種資料相依行為。
    """
    device = DEVICE
    dtype = model.dtype
    num_tiles = pixel_values.shape[0]

    prompt = build_prompt(tokenizer, model, question, num_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)

    embed_fn = model.language_model.get_input_embeddings()
    ids_before = tokenizer(text_before, return_tensors="pt").input_ids.to(device)
    ids_after = tokenizer(text_after, return_tensors="pt").input_ids.to(device)
    embeds_before = embed_fn(ids_before)
    embeds_after = embed_fn(ids_after)

    vision_chunks = []
    for i in range(0, num_tiles, vit_batch):
        chunk = pixel_values[i:i + vit_batch].to(device, dtype=dtype)
        vision_chunks.append(encode_tile(model, chunk, dtype=dtype))
        del chunk
    vision_tokens = torch.cat(vision_chunks, dim=0).reshape(1, -1, embeds_before.shape[-1])

    inputs_embeds = torch.cat([embeds_before, vision_tokens, embeds_after], dim=1)
    vis_start = embeds_before.shape[1]
    vis_len = vision_tokens.shape[1]
    total_len = inputs_embeds.shape[1]

    hidden_states, past_key_values, final_vis_len, debug_stats = fastv_prefill(
        model, inputs_embeds, agg_layer=agg_layer, keep_count=budget,
        vis_start=vis_start, vis_len=vis_len,
    )

    num_layers = model.language_model.config.num_hidden_layers
    seq_lens_per_layer = [past_key_values.get_seq_length(i) for i in range(num_layers)]
    kv_cache_mb = estimate_kv_cache_mb(model, seq_lens_per_layer)

    logits = model.language_model.lm_head(hidden_states[:, -1, :])
    next_id = logits.argmax(dim=-1)   # greedy，跟 baseline 的 do_sample=False 對齊

    eos_id = _get_eos_token_id(tokenizer, model)
    generated = [next_id.item()]
    # next_position：所有層共用的絕對位置計數器，從 total_len 開始（prefill
    # 最後一個 token 的絕對位置是 total_len-1，見 fastv_decode_step() docstring
    # 為什麼不能用每層自己的 cache 長度反推）。
    next_position = total_len
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        next_embed = embed_fn(next_id.view(1, 1))
        logits = fastv_decode_step(model, next_embed, past_key_values, next_position, device)
        next_id = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_id.item())
        next_position += 1

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    stats = {
        "raw_vis_tokens": vis_len,
        "final_vis_tokens": final_vis_len,
        "budget": budget,
        "agg_layer": agg_layer,
        "compression_ratio": round(vis_len / max(final_vis_len, 1), 3),
        "debug_stats": debug_stats,
        "kv_cache_mb": round(kv_cache_mb, 2),
    }
    return answer, stats


# ──────────────────────────────────────────────────────────────────────────
# CLI：跟 sparsevlm_internvl.py 同款輸出格式，方便沿用既有評測腳本
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num", type=int, default=48, help="max dynamic tiles")
    parser.add_argument("--vit_batch", type=int, default=4)
    parser.add_argument("--budget", type=int, default=512,
                         help="ATTENTION_RANK：剪枝那一層之後保留的 image token 數"
                              "（直接對應官方 --fast-v-attention-rank，沒有校準，"
                              "final_vis_tokens 恆等於 min(budget, raw_vis_tokens)）")
    parser.add_argument("--agg_layer", type=int, default=2,
                         help="AGG_LAYER：第幾層之後剪枝（0-indexed），官方預設 2，"
                              "所有官方 eval 腳本都固定用這個值")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    parser.add_argument("--run_standard", action="store_true", help="跑未壓縮的 baseline（model.batch_chat）")
    parser.add_argument("--run_fastv", action="store_true", help="跑 FastV")

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--dataset", type=str, default="hrbench", choices=["docvqa", "infovqa", "hrbench"],
                        help="which HF dataset to load when --use_ds is set")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_fastv.json")
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    # ── 1. 載入模型 ──
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    baseline_attn_impl = model.language_model.config._attn_implementation
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")

    torch.cuda.synchronize(DEVICE)
    print(f"Load model time: {time.time() - t0:.2f} s")

    # ── 2. 載入圖像與問題（跟 sparsevlm_internvl.py 同一套 --use_ds/--dataset 選項）──
    if args.use_ds:
        from process_common import load_docvqa, load_infovqa, load_hrbench
        if args.dataset == "docvqa":
            datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                    num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
        elif args.dataset == "infovqa":
            # InfoVQA validation（HF split 名為 "val"）：同 internvl_stream.py
            datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                    num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
        elif args.dataset == "hrbench":
            datasets = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT,
                                     num_image=args.num_images, prompt_mode=HRBENCH_PROMPT)
            if HRBENCH_PROMPT == "open":
                datasets = dedup_hrbench_open(datasets)
        else:
            return
    else:
        from process_common import load_local
        datasets = load_local(args.image, batch_size=1, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process")

    # ── 3. 讀取舊的 output_results.json（如果存在），支援中斷後繼續跑 ──
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        output_results.setdefault("references", [])
        output_results.setdefault("candidates", {})
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', {}))} images already done)")
    else:
        output_results = {"references": [], "candidates": {}}

    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
             "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
             "answers": d.get("answers", [])}
            for k, d in enumerate(datasets)
        ]

    tag = f"fastv_budget={args.budget}_k={args.agg_layer}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
        return

    # ── 4. 逐張處理：跟 SparseVLM 一樣，剪枝決策依賴單張圖的完整 attention，
    #        不同圖片剪完的序列長度天生不同，固定一次一張 ──
    for i, d in enumerate(datasets):
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(datasets)}] Processing image...")
        print(f"{'='*70}")

        pixel_values = load_image_tiles(d["image"], input_size=448, max_num=args.max_num)
        print(f"  tiles = {pixel_values.shape[0]}  raw_vis_tokens = {pixel_values.shape[0] * model.num_image_token}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        baseline_kv_mb = None
        fastv_kv_mb = None

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                _set_attn_implementation(model, baseline_attn_impl)

                num_layers = model.language_model.config.num_hidden_layers
                tb_len, v_len, ta_len = estimate_prompt_token_lengths(
                    tokenizer, model, d["question"], pixel_values.shape[0])
                total_len = tb_len + v_len + ta_len
                baseline_kv_mb = estimate_kv_cache_mb(model, [total_len] * num_layers)

                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("fastv_script_standard_generate"):
                    ref = generate_answer_standard(
                        model, tokenizer, [pixel_values], [d["question"]],
                        generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                    )
                ref = clean_answer(ref)
                output_results["references"].append(ref[0])
                print(f"  [KV cache, prompt-only] {baseline_kv_mb:.1f} MB "
                      f"({total_len} tokens x {num_layers} layers)")
                print(f"\n[Baseline Answer]\n{ref[0]}")

            if args.run_fastv:
                output_results["candidates"].setdefault(tag, [])

                print(f"\n[FastV] Running one-shot pruning at layer {args.agg_layer} "
                      f"(keep={args.budget}) ...")
                with measure_peak_memory("fastv_script_fastv_generate"):
                    answer, stats = generate_fastv(
                        model, tokenizer, pixel_values, d["question"],
                        budget=args.budget, agg_layer=args.agg_layer,
                        vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                    )
                fastv_kv_mb = stats["kv_cache_mb"]
                print(f"  {stats['raw_vis_tokens']} -> {stats['final_vis_tokens']} tokens "
                      f"(compression={stats['compression_ratio']}x)  debug_stats={stats['debug_stats']}")
                print(f"  [KV cache, prompt-only] {fastv_kv_mb:.1f} MB")
                output_results["candidates"][tag].append(answer)
                print(f"\n[FastV Answer]\n{answer}")

            if baseline_kv_mb is not None and fastv_kv_mb is not None:
                reduction = 100 * (1 - fastv_kv_mb / baseline_kv_mb)
                print(f"\n[KV cache, prompt-only] baseline={baseline_kv_mb:.1f} MB  "
                      f"fastv={fastv_kv_mb:.1f} MB  ({reduction:.1f}% 減少)")

            del pixel_values

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on image {i}: {e}")
            raise

        gc.collect()
        torch.cuda.empty_cache()

        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(output_results, f, indent=2, ensure_ascii=False)
            print(f"\n[Saved] {args.output_json} updated "
                  f"({len(output_results['references'])}/{len(datasets)} images done)")

    torch.cuda.synchronize(DEVICE)
    print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
