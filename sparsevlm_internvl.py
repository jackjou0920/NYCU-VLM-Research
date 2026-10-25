"""
獨立、不改動任何既有檔案的「SparseVLM × InternVL」執行腳本。

跟 run_internvl.sh / internvl_svm.py 走的是完全不同的一條路徑：那邊靠
TileStreamingMemoryBank 在進 LLM 之前（ViT 輸出端）一次性決定 token 數，
整段呼叫 model.language_model(...) 跑到底；這裡改成手動逐層呼叫 Qwen3Model
的 decoder layers，在 SparseVLMPruner.pruning_loc 指定的幾層插入
sparsevlm_core.py 的剔除/合併邏輯，讓視覺 token 真的在 LLM decoder 內部
漸進式被剔除——這是 SparseVLM 論文的核心機制，沒辦法只靠一個 score_fn
插進既有 pipeline，必須自己控制每一層的 attention_mask / position_ids /
rotary embedding 怎麼跟著序列長度縮短（見 sparsevlm_core.py 開頭的說明）。

只「讀取」既有模組的公用函式（build_model / load_image_tiles / build_prompt /
split_prompt_at_vision / encode_tile / clean_answer / measure_peak_memory /
load_local 等），完全沒有修改任何一行既有檔案。輸出的 JSON 格式
（{"references": [...], "candidates": {tag: [...]}}）跟 internvl_svm.py
一致，output_json 可以直接丟給既有的 evaluate_anls.py / evaluate_hrbench.py /
bench_metrics.py 用，不用改評測腳本。

用法範例：
    python sparsevlm_internvl.py \
        --image img_datasets/img01.jpg --budget 512 --lam 0.1 \
        --run_standard --run_sparsevlm --save

peak memory 量測跟 internvl_stream.py 用同一個 measure_peak_memory()
context manager 包住 baseline 跟 sparsevlm 各自的生成過程，數字可以直接
跟你原本方法的 output log 對照，不需要额外校準。

【KV cache 記憶體量測，對應 SparseVLM 論文報的效率數字】
  論文報的「KV cache 從 302.4MB 降到 100.8MB（192/576 token，約 67%）」量的
  是 KV cache tensor 本身的大小，不是整個 process 的 peak_allocated——後者
  被固定的模型權重（InternVL3.5-8B 約 17GB）當分母稀釋，同樣的壓縮比例
  反映到 peak_allocated 上只會是一個很小的百分比，兩個指標不是同一把尺
  （detail 見 estimate_kv_cache_mb() 的 docstring）。這個腳本額外印一組
  「KV cache MB」數字，用模型架構＋實際序列長度直接算出來，排除模型權重，
  才是跟論文同一種量測方式的數字。

【已知限制，如實記錄】
  - 剪枝那幾層（pruning_loc）的 attn_out 走 PyTorch 真正的融合 kernel
    （F.scaled_dot_product_attention，有裝 flash_attn 會自動走
    FlashAttention2），只有「打分要用的 attention weights」另外手算一次，
    而且只算文字 token query 那幾列（O(L_t * L)，不是 O(L^2)）——不是整層
    切 eager，其餘 33 層 + 整個 decode loop 完全沿用你模型原本的
    attn_implementation，速度/記憶體特性應該跟你原本 pipeline 一致。
  - batch_size 固定為 1（一次處理一張圖）：SparseVLM 的剔除決策是「這一層
    的完整 attention 矩陣」算出來的，不同圖片剪枝後的序列長度天生不同，
    没辦法像 baseline 那樣同 batch 內 padding 對齊，所以這裡沒有做 batch
    化，跟 internvl_svm.py 可以 --batch_size > 1 不同。
  - lam（Eq.7 的 λ）沒有論文/官方數值可以抄，需要你自己先用
    --probe_lam 掃一輪看最終 token 數落在哪裡，再決定要用的 lam（細節見
    sparsevlm_core.py 的 rank_based_stage_removal() docstring）。
"""
from __future__ import annotations

import argparse
import gc
import importlib
import json
import os
import time

import torch
from process_common import get_image
import torch.nn.functional as F
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv

from internvl_stream import (
    MAX_NEW_TOKENS, build_prompt, clean_answer, generate_answer_standard,
    split_prompt_at_vision, encode_tile, build_model, load_image_tiles,
    DEVICE, measure_peak_memory,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
)
from sparsevlm_core import SparseVLMPruner


# ──────────────────────────────────────────────────────────────────────────
# 模型設定：eager attention 只在真的要跑 SparseVLM 時才切（整個
# language_model 共用一個 config 物件，改一次就全部生效，才能在任何一層
# 拿到 attn_weights）。baseline 跑之前要切回原本的 attn_implementation
# （通常是 FlashAttn2 未裝時的 SDPA），否則量出來的 peak memory 是「baseline
# 也被迫用 eager」的數字，跟你原本 internvl_svm.py 用的 attention 後端不同，
# 比較起來不公平——這裡確保兩邊的 peak memory 是在各自「正常會用」的
# attention 實作下量出來的，SparseVLM 因為機制本身需要 eager 才拿得到
# attn_weights，這個代價算在它自己頭上，不會被錯誤地也算進 baseline。
# ──────────────────────────────────────────────────────────────────────────
def _set_attn_implementation(model, impl: str):
    model.language_model.config._attn_implementation = impl
    model.language_model.model.config._attn_implementation = impl


def _get_eos_token_id(tokenizer, model) -> int:
    """跟 build_prompt() 用同一招拿 model 自帶的 conversation template，取其
    sep token 當 eos——跟官方 InternVLChatModel.chat()/generate() 設
    generation_config['eos_token_id'] 的方式一致，不用自己猜字串。"""
    model_module = importlib.import_module(type(model).__module__)
    get_conv_template = model_module.get_conv_template
    template = get_conv_template(model.template)
    return tokenizer.convert_tokens_to_ids(template.sep.strip())


# ──────────────────────────────────────────────────────────────────────────
# KV cache 記憶體：跟 SparseVLM 論文同一種量測方式（tensor 本身的大小，
# 不是整個 process 的 peak_allocated）
# ──────────────────────────────────────────────────────────────────────────
def estimate_kv_cache_mb(model, seq_lens_per_layer: list[int], batch: int = 1) -> float:
    """
    SparseVLM 論文報的「192/576 token 時 KV cache 從 302.4MB 降到
    100.8MB（約 67%）」，量的是 KV cache tensor 本身佔用的位元組數，公式是
    2(K+V) * batch * num_kv_heads * head_dim * seq_len * dtype_bytes，對
    每一層各自算再加總——不是 torch.cuda.max_memory_allocated() 量到的
    「整個 process 的 peak GPU 記憶體」。後者包含固定的模型權重（InternVL
    3.5-8B 約 17GB），同樣的壓縮比例反映到 peak_allocated 上會被這個大
    分母稀釋成很小的百分比，兩者不是同一把尺，不能直接拿來跟論文的數字
    對照——這個函式算的才是跟論文同一種量測方式的數字。

    seq_lens_per_layer：每一層目前 KV cache 的實際長度（token 數）。baseline
    沒有剪枝，所有層長度一樣；SparseVLM 剪枝後不同層長度不同（見
    generate_sparsevlm() 裡怎麼從 past_key_values.get_seq_length(layer_idx)
    逐層讀出真正的長度），所以這裡吃一個 per-layer 的 list，不是單一數字。
    """
    cfg = model.language_model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    dtype_bytes = torch.tensor([], dtype=model.dtype).element_size()

    total_tokens = sum(seq_lens_per_layer)
    total_bytes = 2 * batch * num_kv_heads * head_dim * total_tokens * dtype_bytes
    return total_bytes / (1024 ** 2)


def estimate_prompt_token_lengths(tokenizer, model, question: str, num_tiles: int):
    """回傳 (text_before_len, vis_len, text_after_len)，純 tokenizer 計算、
    不碰 GPU，用來算 baseline 的 prompt-only KV cache（baseline 是黑盒的
    model.batch_chat()，拿不到它內部真正的 past_key_values，改用這組公開
    可算的 token 數字，套用跟 SparseVLM 完全一樣的 estimate_kv_cache_mb()
    公式，兩邊才是同一種算法、可以直接對照）。"""
    prompt = build_prompt(tokenizer, model, question, num_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)
    text_before_len = tokenizer(text_before, return_tensors="pt").input_ids.shape[1]
    text_after_len = tokenizer(text_after, return_tensors="pt").input_ids.shape[1]
    vis_len = num_tiles * model.num_image_token
    return text_before_len, vis_len, text_after_len


# ──────────────────────────────────────────────────────────────────────────
# Prefill：手動逐層跑，在 pruning_loc 插入 SparseVLMPruner
# ──────────────────────────────────────────────────────────────────────────
def _segment_geometry(hidden_states: torch.Tensor, device):
    """SparseVLM 的『基礎版』做法：剪枝後 position_ids 直接從 0 重新編號
    （不是保留原始絕對位置），見 sparsevlm_core.py 開頭關於 V1 vs. V2_0
    的說明——這裡刻意跟 sparsevlm_core.py 的演算法選擇保持一致。"""
    L = hidden_states.shape[1]
    position_ids = torch.arange(L, device=device).unsqueeze(0)
    cache_position = torch.arange(L, device=device)
    return position_ids, cache_position


def sparsevlm_prefill(
    model, inputs_embeds: torch.Tensor, pruner: SparseVLMPruner,
    vis_start: int, vis_len: int, text_start: int,
):
    """回傳 (hidden_states [1, L_final, D], past_key_values, final_vis_len, stage_logs)。

    past_key_values（DynamicCache）天生允許各層長度不同（每層各自維護自己
    的 K/V list），這正是官方 repo 需要整個重寫 forward() 才能做到的事——
    這裡因為是自己手動逐層呼叫，直接拿到這個粒度的控制權，不需要額外
    工程。pruning_loc 那幾層剪枝『之前』先正常算完那一層的 attention（K/V
    已經以剪枝前的長度寫進 past_key_values[layer_idx]），剪枝只影響『這一層
    輸出』要不要繼續往下傳給下一層，不影響這一層自己已經 cache 好的 K/V。
    """
    lm_model = model.language_model.model  # Qwen3Model
    device = inputs_embeds.device
    past_key_values = DynamicCache(config=lm_model.config)

    hidden_states = inputs_embeds
    position_ids, cache_position = _segment_geometry(hidden_states, device)
    causal_mask = create_causal_mask(
        config=lm_model.config, input_embeds=hidden_states, attention_mask=None,
        cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
    )
    position_embeddings = lm_model.rotary_emb(hidden_states, position_ids)

    stage_logs = []
    for layer_idx, decoder_layer in enumerate(lm_model.layers):
        if layer_idx in pruner.pruning_loc:
            # 手動重現 Qwen3DecoderLayer.forward() 的 pre-norm block，但把「算
            # attn_out」跟「算打分要用的 attn_weights」分開算：
            #   (1) attn_out 用 F.scaled_dot_product_attention()——PyTorch 真正
            #       的融合 kernel（有裝 flash_attn 的話會走 FlashAttention2），
            #       不會材料化整個 [H, L, L] 的 attention matrix。
            #   (2) attn_weights 只手算「文字 token query」那幾列對「全部 key」
            #       的 attention，大小是 O(L_t * L)，L_t（問題文字 token 數）
            #       通常只有幾十個，比 O(L^2) 小上百倍——sparsevlm_prune_step()
            #       本來就只需要文字 token 的 query（見 extract_text_visual_
            #       relation / extract_full_question_relation），從來不需要
            #       視覺 token 自己的 attention row，所以這裡乾脆不算。
            # 讀官方原始碼的 LlamaDynamicvitFlashAttention2（return_logits 分支）
            # 發現同樣的設計精神——attn_out 走真正的 flash kernel、logits 另外
            # 手算一次，而不是整層切 eager；差別是官方版本對全部 query rows
            # 都手算 logits，這裡進一步縮小到只有文字 token 的 query rows。
            # 上一版把整層切成 eager（attn_out 也走慢速手算），在大圖（6400
            # 原始 token）上實測 peak memory 反而超過 baseline，根因就是那個
            # O(L^2) 的 attn_out 材料化，這裡直接避開。
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

            # 只對「文字 token query」（絕對位置 text_start 之後）算 attention
            # weights；這些 query 的因果遮罩很簡單：對自己位置之前（含視覺
            # token，天然全部在它們之前）全部可見，只有彼此之間才需要 causal。
            text_query = query_states[:, :, text_start:, :]                         # [B, H, L_t, d]
            text_logits = (text_query @ key_full.transpose(-2, -1)) * self_attn.scaling  # [B, H, L_t, L]
            L_t = text_logits.shape[-2]
            L_total = text_logits.shape[-1]
            row_pos = torch.arange(text_start, text_start + L_t, device=device).unsqueeze(1)
            col_pos = torch.arange(L_total, device=device).unsqueeze(0)
            text_logits = text_logits.masked_fill(col_pos > row_pos, float("-inf"))
            attn_weights = text_logits.softmax(dim=-1).to(query_states.dtype)       # [B, H, L_t, L]

            hidden_states = residual + attn_out
            residual = hidden_states
            hidden_states = decoder_layer.post_attention_layernorm(hidden_states)
            hidden_states = decoder_layer.mlp(hidden_states)
            hidden_states = residual + hidden_states

            stage_idx = pruner.pruning_loc.index(layer_idx)
            new_hidden, new_vis_len, debug_stats = pruner.step(
                stage_idx, hidden_states[0], attn_weights[0],
                vis_start, vis_len, text_start, row_offset=text_start,
            )
            hidden_states = new_hidden.unsqueeze(0)
            vis_len = new_vis_len
            text_start = vis_start + vis_len
            stage_logs.append({"layer": layer_idx, **debug_stats})

            # 長度變了：mask / position_ids / rotary 都要用新長度重算，之後的
            # 層一路沿用，直到下一次剪枝再重算一次。
            position_ids, cache_position = _segment_geometry(hidden_states, device)
            causal_mask = create_causal_mask(
                config=lm_model.config, input_embeds=hidden_states, attention_mask=None,
                cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
            )
            position_embeddings = lm_model.rotary_emb(hidden_states, position_ids)
        else:
            hidden_states = decoder_layer(
                hidden_states, attention_mask=causal_mask, position_ids=position_ids,
                past_key_values=past_key_values, use_cache=True,
                cache_position=cache_position, position_embeddings=position_embeddings,
            )

    hidden_states = lm_model.norm(hidden_states)
    return hidden_states, past_key_values, vis_len, stage_logs


# ──────────────────────────────────────────────────────────────────────────
# Decode：每一層各自的 KV cache 長度在 prefill 後可能不同（第一個 pruning
# layer 之前的層比較長，最後一個 pruning layer 之後的層比較短），所以不能
# 用一般 model.generate()（它假設全模型單一 cache_position）；改成逐層問
# past_key_values.get_seq_length(layer_idx) 各自要接在哪個位置。
# ──────────────────────────────────────────────────────────────────────────
def sparsevlm_decode_step(model, hidden_states_1tok: torch.Tensor, past_key_values, device):
    lm_model = model.language_model.model
    h = hidden_states_1tok
    for decoder_layer in lm_model.layers:
        layer_cache_len = past_key_values.get_seq_length(decoder_layer.self_attn.layer_idx)
        cache_position = torch.tensor([layer_cache_len], device=device)
        position_ids = cache_position.unsqueeze(0)
        position_embeddings = lm_model.rotary_emb(h, position_ids)
        # attention_mask=None：新 token 是這一層 cache 裡最新的一個位置，
        # 對「已經在 cache 裡的過去」跟「自己」全部可見，因果限制天然滿足，
        # 不需要額外的遮罩（跟 HF 官方 decode step 對單一新 token 的處理一致）。
        h = decoder_layer(
            h, attention_mask=None, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=True,
            cache_position=cache_position, position_embeddings=position_embeddings,
        )
    h = lm_model.norm(h)
    return model.language_model.lm_head(h)


# ──────────────────────────────────────────────────────────────────────────
# 頂層：build prompt → 灌 vision → sparsevlm_prefill → greedy decode
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_sparsevlm(
    model, tokenizer, pixel_values: torch.Tensor, question: str,
    budget: int, lam: float,
    vit_batch: int = 4, max_new_tokens: int = MAX_NEW_TOKENS,
    rank_energy_threshold: float = 0.99, merge_filter_ratio: float = 0.3,
    cluster_divisor: int = 10, knn_k: int = 5,
):
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
    text_start = vis_start + vis_len

    num_layers = model.language_model.config.num_hidden_layers
    pruner = SparseVLMPruner(
        num_layers=num_layers, target_budget=budget, lam=lam,
        rank_energy_threshold=rank_energy_threshold,
        merge_filter_ratio=merge_filter_ratio, cluster_divisor=cluster_divisor, knn_k=knn_k,
    )

    hidden_states, past_key_values, final_vis_len, stage_logs = sparsevlm_prefill(
        model, inputs_embeds, pruner, vis_start, vis_len, text_start,
    )

    # prompt-only 的 KV cache 大小（生成任何 token 之前）：直接從真正的
    # past_key_values 逐層讀出實際長度，不同層因為剪枝時機不同而不一樣
    # （跟論文同一種量測方式，見 estimate_kv_cache_mb() docstring）。
    seq_lens_per_layer = [past_key_values.get_seq_length(i) for i in range(num_layers)]
    kv_cache_mb = estimate_kv_cache_mb(model, seq_lens_per_layer)

    logits = model.language_model.lm_head(hidden_states[:, -1, :])
    next_id = logits.argmax(dim=-1)   # greedy，跟 baseline 的 do_sample=False 對齊

    eos_id = _get_eos_token_id(tokenizer, model)
    generated = [next_id.item()]
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        next_embed = embed_fn(next_id.view(1, 1))
        logits = sparsevlm_decode_step(model, next_embed, past_key_values, device)
        next_id = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_id.item())

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    stats = {
        "raw_vis_tokens": vis_len,
        "final_vis_tokens": final_vis_len,
        "budget": budget,
        "compression_ratio": round(vis_len / max(final_vis_len, 1), 3),
        "stage_logs": stage_logs,
        "kv_cache_mb": round(kv_cache_mb, 2),
    }
    return answer, stats


def budget_lam(raw_vis_tokens: int, budget: int, lam: float) -> float:
    """實驗流程的 budget 規則（不是 SparseVLM 演算法的一部分）：原圖 token 數 <= budget 時不剪枝（λ = 0），
    跟其他方法的 final = min(budget, raw) 一致。Eq.7 的剔除數 λ(L_v - rank(P)) 跟 budget 無關，固定 λ 下
    小圖會被剪到遠低於 budget（DocVQA b4096：不需壓縮的 43% 題目 91.0 → 75.9）。官方實驗的圖都是固定
    576 token、一定超過 budget，所以官方碼沒有這個情況。2026-10-14 使用者同意採用，論文附錄要註明。"""
    return 0.0 if raw_vis_tokens <= budget else lam


# ──────────────────────────────────────────────────────────────────────────
# 自動校準 λ：讓 Eq.7 的 3-stage 漸進剪枝『自然』壓到接近 budget，公平代表
# SparseVLM 本身的機制（不是靠 generate_sparsevlm() 最後那步保底硬裁切在
# 做大部分的壓縮工作）。
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def auto_tune_lam(
    model, tokenizer, pixel_values: torch.Tensor, question: str, budget: int,
    lam_lo: float = 0.005, lam_hi: float = 3.0, max_evals: int = 6,
    overshoot_tol: float = 0.5,
    vit_batch: int = 4, rank_energy_threshold: float = 0.99,
    merge_filter_ratio: float = 0.3, cluster_divisor: int = 10, knn_k: int = 5,
):
    """
    回傳 (lam, calib_log)。

    用 bounded binary search 找『最小的』λ，讓 3 個 pruning stage 跑完、
    「裁切前」的 organic vis_len 落在 [budget, budget*(1+overshoot_tol)]
    這個區間——刻意找最小的 λ 而不是隨便一個滿足區間的值，是因為 λ 越大
    代表 Eq.7 越激進地丟資訊，最小可行的 λ = 保留最多資訊、同時仍然滿足
    「clip 只是收尾、不是主力」這個公平性要求的那個值。

    Eq.7 的 rank(P) 是資料相依的（見 rank_based_stage_removal() docstring），
    不同 λ 會讓前面 stage 留下不同的 token 子集，連帶讓後面 stage 的 P 矩陣
    不同——沒有捷徑可以只跑一次 forward 就分析出對所有 λ 都成立的答案，
    所以這裡每個候選 λ 都是真的重新跑一次 prefill（max_new_tokens=1，不用
    真的 decode，省一點時間，但每次還是要吃 25 tile 的完整 ViT + 3 個
    pruning layer 的 attention，不是免費的——batch 跑很多圖片時，這個校準
    成本會乘上 max_evals 倍，見 main() 的 --auto_lam 說明）。

    organic vis_len 讀 stats['stage_logs'][-1]['vis_len_after']：這個欄位
    是 sparsevlm_prune_step() 內部算出來、clip 之前的值，SparseVLMPruner.
    step() 的 clip 邏輯不會覆寫它，所以可以直接拿來當『這個 λ 真正做到的
    壓縮結果』，不受最後那步硬裁切影響。
    """
    def _organic_len(lam: float) -> int:
        _, stats = generate_sparsevlm(
            model, tokenizer, pixel_values, question, budget=budget, lam=lam,
            vit_batch=vit_batch, max_new_tokens=1,
            rank_energy_threshold=rank_energy_threshold,
            merge_filter_ratio=merge_filter_ratio,
            cluster_divisor=cluster_divisor, knn_k=knn_k,
        )
        return stats["stage_logs"][-1]["vis_len_after"]

    threshold = budget * (1 + overshoot_tol)
    calib_log = []
    raw_len = pixel_values.shape[0] * model.num_image_token
    if raw_len <= threshold:
        # 原圖 token 本來就不超過門檻：任何 λ 都「達標」，這張圖對 λ 沒有資訊。
        # 回傳 lam_lo 但標記 skip，校準（run_sparsevlm.sh 取中位數）要排除這類圖，
        # 否則小圖一多，中位數會被拉到 lam_lo（之前 b4096/b5120 校準出 0.005 就是這樣）。
        return lam_lo, [{"skip": True, "raw_len": raw_len}]

    lo, hi = lam_lo, lam_hi
    len_lo = _organic_len(lo)
    calib_log.append({"lam": lo, "organic_len": len_lo})
    if len_lo <= threshold:
        # 連最小的 λ 都已經壓到門檻以下，不用再搜，直接用最小值（保留最多資訊）。
        return lo, calib_log

    len_hi = _organic_len(hi)
    calib_log.append({"lam": hi, "organic_len": len_hi})
    if len_hi > threshold:
        # 連最大的 λ 都壓不到門檻——lam_hi 開太小，把最大值當結果回傳，
        # 並如實記錄『沒收斂』，呼叫端可以自己決定要不要調大 lam_hi 重跑。
        calib_log.append({"converged": False, "reason": "lam_hi 太小，壓不到 budget"})
        return hi, calib_log

    for _ in range(max(0, max_evals - 2)):
        mid = (lo + hi) / 2
        len_mid = _organic_len(mid)
        calib_log.append({"lam": mid, "organic_len": len_mid})
        if len_mid <= threshold:
            hi = mid
        else:
            lo = mid

    calib_log.append({"converged": True})
    return hi, calib_log


# ──────────────────────────────────────────────────────────────────────────
# CLI：跟 internvl_svm.py 同款輸出格式，方便沿用既有評測腳本
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num", type=int, default=48, help="dynamic tile 上限（含縮圖），跟 internvl_stream.py 同語意")
    parser.add_argument("--vit_batch", type=int, default=4)
    parser.add_argument("--budget", type=int, default=512, help="target visual-token budget after all pruning stages")
    parser.add_argument("--lam", type=float, default=0.1, help="Eq.7 的 λ，沒有官方數值，先用 --probe_lam 掃一輪；"
                                                                  "有 --auto_lam 的話這個值只當校準搜尋失敗時的備援")
    parser.add_argument(
        "--auto_lam", action="store_true",
        help="每張圖、每個 budget 自動校準 λ（見 auto_tune_lam()），取代固定的 --lam："
             "bounded binary search 找最小的 λ，讓 Eq.7 的 3-stage 漸進剪枝『自然』"
             "（裁切前）壓到接近 budget，避免最後一步保底硬裁切變成主要的壓縮手段——"
             "這樣才是公平代表 SparseVLM 本身的機制，也才能跟你的 streaming 方法在"
             "『同一個最終 budget』下公平比較。代價：每張圖多花最多 "
             "--auto_lam_max_evals 次 prefill-only 探測（不含 decode，但仍要跑滿"
             "整段 ViT + 前 3 個 pruning layer），批次跑多張圖時這個成本會疊加。",
    )
    parser.add_argument("--auto_lam_lo", type=float, default=0.005)
    parser.add_argument("--auto_lam_hi", type=float, default=3.0)
    parser.add_argument("--auto_lam_max_evals", type=int, default=6)
    parser.add_argument("--auto_lam_overshoot_tol", type=float, default=0.5,
                         help="校準目標是讓裁切前的 organic vis_len 落在 "
                              "[budget, budget*(1+overshoot_tol)] 區間內")
    parser.add_argument("--rank_energy_threshold", type=float, default=0.99)
    parser.add_argument("--merge_filter_ratio", type=float, default=0.3)
    parser.add_argument("--cluster_divisor", type=int, default=10)
    parser.add_argument("--knn_k", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    parser.add_argument("--run_standard", action="store_true", help="跑未壓縮的 baseline（model.batch_chat）")
    parser.add_argument("--run_sparsevlm", action="store_true", help="跑 SparseVLM")
    parser.add_argument(
        "--probe_lam", type=str, default=None,
        help="逗號分隔的多個 lam 值，例如 '0.02,0.05,0.1,0.2'：只印每個 lam 剪完的"
             "最終 vis token 數，不跑 decode，用來在正式跑之前先挑一個讓 token 數"
             "落在你要的 budget 附近的 lam。",
    )

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--dataset", type=str, default="hrbench", choices=["docvqa", "infovqa", "hrbench", "mme_rw_lite", "vstar", "realworldqa"],
                        help="which HF dataset to load when --use_ds is set")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_sparsevlm.json")
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

    # ── 2. 載入圖像與問題（跟 internvl_stream.py 同一套 --use_ds/--dataset 選項，
    #        方便直接沿用既有的 evaluate_mmmu.py / evaluate_anls.py /
    #        evaluate_hrbench*.py，不用另外寫一套 dataset 對齊邏輯）──
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
        elif args.dataset in ("mme_rw_lite", "vstar", "realworldqa"):
            # 大圖、圖片延遲開檔：沿用 internvl_stream 的資料集註冊（process_common.load_mme_realworld / load_vstar）
            from internvl_stream import load_eval_dataset
            datasets = load_eval_dataset(args.dataset, args.num_images)
        else:
            return
    else:
        from process_common import load_local
        datasets = load_local(args.image, batch_size=1, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process")

    if args.probe_lam:
        d = datasets[0]
        pixel_values = load_image_tiles(get_image(d), input_size=448, max_num=max(1, args.max_num - 1))
        print(f"\n[probe_lam] tiles={pixel_values.shape[0]}, "
              f"raw_vis_tokens={pixel_values.shape[0] * model.num_image_token}")
        for lam in (float(x) for x in args.probe_lam.split(",")):
            _, stats = generate_sparsevlm(
                model, tokenizer, pixel_values, d["question"], budget=args.budget, lam=lam,
                vit_batch=args.vit_batch, max_new_tokens=1,   # decode 1 個 token 就好，只看 prefill 剪枝結果
                rank_energy_threshold=args.rank_energy_threshold,
                merge_filter_ratio=args.merge_filter_ratio,
                cluster_divisor=args.cluster_divisor, knn_k=args.knn_k,
            )
            print(f"  lam={lam:<8g} final_vis_tokens={stats['final_vis_tokens']:<6d} "
                  f"stage_logs={stats['stage_logs']}")
        return

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

    # ── 2b. HR-Bench / DocVQA / InfoVQA：把每題的 answer/options/... 存進 JSON 的
    #        meta，讓既有的 evaluate_*.py 直接讀 meta 算 GT，不用重載 HF dataset。──
    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
             "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
             "answers": d.get("answers", [])}
            for k, d in enumerate(datasets)
        ]

    tag = f"sparsevlm_budget={args.budget}_lam={args.lam}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
        return

    # ── 4. 逐張處理：SparseVLM 的剪枝決策依賴單張圖的完整 attention 矩陣，不同
    #        圖片剪完的序列長度天生不同，沒辦法像 internvl_stream.py 那樣同批
    #        padding 對齊，所以這裡固定一次一張（batch_size 恆為 1）──
    for i, d in enumerate(datasets):
        # 斷點續跑：這張圖兩邊都已經有結果就跳過（2026-10-14 補上：之前沒做完的 JSON 重跑會從第 1 張
        # 重新 append，答案跟 meta 錯位；只動續跑邏輯，不影響方法本身）
        need_ref = args.run_standard and len(output_results["references"]) <= i
        need_cand = args.run_sparsevlm and len(output_results["candidates"].get(tag, [])) <= i
        if not (need_ref or need_cand):
            continue
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(datasets)}] Processing image...")
        print(f"{'='*70}")

        pixel_values = load_image_tiles(get_image(d), input_size=448, max_num=max(1, args.max_num - 1))
        print(f"  tiles = {pixel_values.shape[0]}  raw_vis_tokens = {pixel_values.shape[0] * model.num_image_token}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        baseline_kv_mb = None
        sparsevlm_kv_mb = None

        try:
            if need_ref:
                _set_attn_implementation(model, baseline_attn_impl)   # 跟你原本 pipeline 用同一種 attention 後端

                # prompt-only KV cache（生成任何 token 之前）：baseline 是黑盒的
                # model.batch_chat()，拿不到它內部真正的 past_key_values，改用
                # 跟 SparseVLM 完全同一條公式（estimate_kv_cache_mb）、同一組
                # token 數字（build_prompt/tokenizer 直接算，不用真的跑一次
                # forward）反推——沒有剪枝，所有層長度都一樣。
                num_layers = model.language_model.config.num_hidden_layers
                tb_len, v_len, ta_len = estimate_prompt_token_lengths(
                    tokenizer, model, d["question"], pixel_values.shape[0])
                total_len = tb_len + v_len + ta_len
                baseline_kv_mb = estimate_kv_cache_mb(model, [total_len] * num_layers)

                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("sparsevlm_script_standard_generate"):
                    ref = generate_answer_standard(
                        model, tokenizer, [pixel_values], [d["question"]],
                        generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                    )
                ref = clean_answer(ref)
                output_results["references"].append(ref[0])
                print(f"  [KV cache, prompt-only] {baseline_kv_mb:.1f} MB "
                      f"({total_len} tokens x {num_layers} layers)")
                print(f"\n[Baseline Answer]\n{ref[0]}")

            if need_cand:
                # 不在這裡整段切 eager：sparsevlm_prefill() 內部只在真正呼叫
                # pruning_loc 那幾層 self_attn 的當下切，其餘層維持目前的
                # attn_implementation（此刻是 baseline_attn_impl）。
                output_results["candidates"].setdefault(tag, [])

                lam = args.lam
                if args.auto_lam:
                    lam, calib_log = auto_tune_lam(
                        model, tokenizer, pixel_values, d["question"], budget=args.budget,
                        lam_lo=args.auto_lam_lo, lam_hi=args.auto_lam_hi,
                        max_evals=args.auto_lam_max_evals, overshoot_tol=args.auto_lam_overshoot_tol,
                        vit_batch=args.vit_batch, rank_energy_threshold=args.rank_energy_threshold,
                        merge_filter_ratio=args.merge_filter_ratio,
                        cluster_divisor=args.cluster_divisor, knn_k=args.knn_k,
                    )
                    if calib_log and calib_log[0].get("skip"):
                        print(f"  [auto_lam] skip (raw tokens {calib_log[0]['raw_len']} <= threshold), lam={lam:.4g}")
                    else:
                        print(f"  [auto_lam] chosen lam={lam:.4g}  calib_log={calib_log}")
                raw_tokens = pixel_values.shape[0] * model.num_image_token
                if budget_lam(raw_tokens, args.budget, lam) != lam:
                    print(f"  [budget] raw tokens {raw_tokens} <= budget {args.budget}：不剪枝（λ=0）")
                    lam = 0.0

                print(f"\n[SparseVLM] Running layer-wise progressive pruning (lam={lam:.4g}) ...")
                with measure_peak_memory("sparsevlm_script_sparsevlm_generate"):
                    answer, stats = generate_sparsevlm(
                        model, tokenizer, pixel_values, d["question"],
                        budget=args.budget, lam=lam, vit_batch=args.vit_batch,
                        max_new_tokens=args.max_new_tokens,
                        rank_energy_threshold=args.rank_energy_threshold,
                        merge_filter_ratio=args.merge_filter_ratio,
                        cluster_divisor=args.cluster_divisor, knn_k=args.knn_k,
                    )
                stats["lam"] = lam
                sparsevlm_kv_mb = stats["kv_cache_mb"]
                print(f"  {stats['raw_vis_tokens']} -> {stats['final_vis_tokens']} tokens "
                      f"(compression={stats['compression_ratio']}x)  stage_logs={stats['stage_logs']}")
                print(f"  [KV cache, prompt-only] {sparsevlm_kv_mb:.1f} MB")
                output_results["candidates"][tag].append(answer)
                print(f"\n[SparseVLM Answer]\n{answer}")

            if baseline_kv_mb is not None and sparsevlm_kv_mb is not None:
                reduction = 100 * (1 - sparsevlm_kv_mb / baseline_kv_mb)
                print(f"\n[KV cache, prompt-only] baseline={baseline_kv_mb:.1f} MB  "
                      f"sparsevlm={sparsevlm_kv_mb:.1f} MB  ({reduction:.1f}% 減少)")

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
