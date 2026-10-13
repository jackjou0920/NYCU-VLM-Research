# ══════════════════════════════════════════════════════════════════════════════
# SVMA（InternVL）—— 串流 tile 選擇 + 有界 KV，單檔版
#
# 多尺度：budget K 塊 tile = m 塊低解析度全圖覆蓋（fill_grid）+ K - m 塊原解析度細節 tile（--coverage_tiles m；
#   m = 1 時覆蓋 = 官方縮圖，就是原本的 SVMA）。
# 流程（每張圖）：text_before prefill → 覆蓋 tile ViT + prior（一次短 forward）→ 邊界打分選細節 tile
#   （只對 S 塊邊界 tile 跑 q_attn）→ 細節 tile 依空間順序、覆蓋 tile 在最後寫 KV → text_after + decode。
#   見 InternVLAdapter.coverage_prior / stream_image_to_kv_cascade。
#
# 用法（一次跑一個 dataset × mode × budget，結果存 results/runs/<dataset>/<tag>_<model>.json，可中斷續跑）：
#   python internvl_stream.py --dataset docvqa --budget 1024                       # SVMA
#   python internvl_stream.py --dataset hrbench_8k --mode uncompressed
#   python internvl_stream.py --dataset infovqa --mode lowres_fill --budget 2048 --num_images 500
#   批次跑多個 dataset / budget + 評分：run_svma.sh；彙整成表：summarize_results.py
#
# 外部相依：
#   internvl_preprocess.py  → build_model / load_image_tiles / load_image_tiles_fill / load_image_multiscale
#   process_common.py       → load_docvqa / load_infovqa / load_hrbench
#   svma_config.py          → default_coverage_tiles / default_qattn_tiles / split_slots（budget → m、S、名額分配）
#
# 舊的單階段路徑（info_density 打分 + TileStreamingMemoryBank）、oracle 選擇、abs_pos、
# torch.compile 版 q_attn 已移除（2026-10-11），需要時從 git 歷史找。
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import gc
import heapq
import time
import json
import argparse
import importlib
import contextlib

import torch
import torch.nn.functional as F

from internvl_preprocess import build_model, load_image_tiles, load_image_tiles_fill, load_image_multiscale
from svma_config import default_coverage_tiles, default_qattn_tiles, split_slots


# ══════════════════════════════════════════════════════════════════════════════
# 裝置挑選 / 記憶體量測
# ══════════════════════════════════════════════════════════════════════════════
def get_optimal_cuda_device(min_required_gb: float = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_device_idx = 0
    max_free_memory = 0

    # 巡檢所有可用的 GPU
    for i in range(torch.cuda.device_count()):
        # mem_get_info(i) 回傳 tuple: (free_memory_bytes, total_memory_bytes)
        free_bytes, _ = torch.cuda.mem_get_info(i)

        if free_bytes > max_free_memory:
            max_free_memory = free_bytes
            best_device_idx = i

    # 轉成 GB 進行檢查
    max_free_gb = max_free_memory / (1024**3)

    if min_required_gb > 0 and max_free_gb < min_required_gb:
        print(f"警告：顯存最多的 GPU (cuda:{best_device_idx}) 僅剩 {max_free_gb:.2f} GB，未達要求的 {min_required_gb} GB。")

    return torch.device(f"cuda:{best_device_idx}")


DEVICE = get_optimal_cuda_device(min_required_gb=20.0)


@contextlib.contextmanager
def measure_peak_memory(tag: str, record_timeline: bool = False):
    """跟 internvl_preprocess.py 完全相同的量測邏輯。"""
    if record_timeline:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    torch.cuda.synchronize(DEVICE)
    torch.cuda.reset_peak_memory_stats(DEVICE)
    t0 = time.time()

    try:
        yield
    finally:
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated(DEVICE) / 1e9
        peak_reserved = torch.cuda.max_memory_reserved(DEVICE) / 1e9

        print(f"\n[{tag}] time={elapsed:.2f} sec  "
              f"peak_allocated={peak_alloc:.3f} GB  "
              f"peak_reserved={peak_reserved:.3f} GB")

        if record_timeline:
            torch.cuda.memory._dump_snapshot(f"{tag}_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)


def estimate_kv_cache_mb(model, seq_lens_per_layer: list[int], batch: int = 1) -> float:
    """
    KV cache tensor 本身的大小（MB），不是 torch.cuda.max_memory_allocated()
    量到的「整個 process 的 peak GPU 記憶體」——後者包含固定的模型權重
    （InternVL3.5-8B 約 17GB），同樣的壓縮比例反映到 peak_allocated 上會被
    這個大分母稀釋成很小的百分比，看不出壓縮方法本身的效果。KV cache 壓縮/
    token pruning 文獻（SparseVLM、H2O、StreamingLLM、SnapKV 等）報記憶體
    節省時，用的都是這個口徑（e.g. SparseVLM 論文在 192/576 token 報
    "302.4MB -> 100.8MB，約 67% 減少"），這裡用同一條公式算，才能跟文獻
    數字放在同一張表比。

    公式：2(K+V) * batch * num_kv_heads * head_dim * seq_len * dtype_bytes，
    每一層各自算再加總。seq_lens_per_layer 吃一個 per-layer 的 list 而不是
    單一數字，是因為像 SparseVLM 那種在 LLM 內部漸進式剪枝的方法，不同層
    的 KV cache 長度不一樣；SVMA 在進 LLM 之前就把 token
    數固定了，所有層長度相同，呼叫端傳 [final_size] * num_layers 即可。
    """
    cfg = model.language_model.config
    num_kv_heads = cfg.num_key_value_heads
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    dtype_bytes = torch.tensor([], dtype=model.dtype).element_size()

    total_tokens = sum(seq_lens_per_layer)
    total_bytes = 2 * batch * num_kv_heads * head_dim * total_tokens * dtype_bytes
    return total_bytes / (1024 ** 2)


def estimate_prompt_token_lengths(tokenizer, text_before: str, text_after: str):
    """回傳 (text_before_len, text_after_len)，純 tokenizer 計算、不碰 GPU。
    跟 vision token 數（tile 數 * num_image_token，或壓縮後的 final_size）
    加總即可得到完整序列長度，餵給 estimate_kv_cache_mb()。"""
    tb_len = tokenizer(text_before, return_tensors="pt").input_ids.shape[1]
    ta_len = tokenizer(text_after, return_tensors="pt").input_ids.shape[1]
    return tb_len, ta_len


# ══════════════════════════════════════════════════════════════════════════════
# InternVL 專屬低階函式
# ══════════════════════════════════════════════════════════════════════════════
MAX_NEW_TOKENS = 500
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
RANDOM_SEED = 0

# ── HR-Bench split（baseline 腳本用的環境變數）──
HRBENCH_SPLIT = os.environ.get("HRBENCH_SPLIT", "hrbench_4k")   # hrbench_4k | hrbench_8k
# 用環境變數切 split（資料選擇，不是方法開關）：baseline 腳本 / bench_metrics 都從這裡 import，
# run_*.sh 也讀同一個環境變數決定輸出檔名，例 HRBENCH_SPLIT=hrbench_8k DATASET=hrbench ./run_fastv.sh
# （internvl_stream.py 自己用 --dataset hrbench_4k / hrbench_8k，不看這個）
assert HRBENCH_SPLIT in ("hrbench_4k", "hrbench_8k"), HRBENCH_SPLIT

# ── 評測 prompt（固定；baseline 腳本 import 這三個常數，run_*.sh 也讀它們當輸出檔名的一部分）──
HRBENCH_PROMPT = "letter"     # 選項進 prompt、只輸出字母（官方 CircularEval）
DOCVQA_PROMPT = "short"       # 官方 ANLS 協定（evaluate_anls.py）
INFOVQA_PROMPT = "short"      # 同 DocVQA（validation split，HF 上叫 "val"）

# ── thumbnail prior：thumbnail 跑一次 LLM 前段 forward，取 text_after 所有位置（問題 + 作答指示 +
#    assistant 開頭）對 256 個 thumbnail patch 的 attention → 16x16 粗 saliency → 每個 grid tile 一個 prior。
#    第 21..28 層（1-indexed、含頭尾）平均；forward 跑到第 28 層就停，attention 列手算、主 forward 維持 flash-attn。
#    InfoVQA 全量：跟舊版 (29, 36) + 只看最後一個位置打平（b1024 +0.76、b2048 -0.03，都不顯著），forward 較短。
THUMB_ATTN_LAYERS = (21, 28)

# ── 兩階段 tile 選擇：thumbnail prior 粗排 → 邊界打分（--qattn_tiles S；不給時依 budget 自動，
#    見 svma_config.py 的 QATTN_TILES_BY_BUDGET）──
#   prior 排名前 (K-1-a) 塊直接保留（只跑 ViT），接下來 S 塊跑 ViT + q_attn 取 a 塊（a = min(K-1, S//2)），
#   其餘整塊跳過（連 ViT 都不跑）；存活 tile + thumbnail 依空間順序寫 KV。S=0 = 只用 prior。
#   q_attn = 第 QATTN_LAYER 層 text_after 對 tile 的 attention，打分 context = [text_before][thumbnail][tile][text_after]；
#     [text_before][thumbnail] 的前 QATTN_LAYER 層 KV 直接取自 thumbnail prior 那次 forward（算一次、每塊候選重用）。
#   診斷（InfoVQA b1024 177 張，所有 tile 都當候選）：oracle 最佳 tile 被選進前 K-1 的比例
#     L12 30.5% / L20 36.2% / ctx+L12 34.5% / ctx+L20 44.1%（隨機約 12%）。tag 後綴 _qa20ctx。
QATTN_LAYER = 20

VIT_BATCH = 4               # ViT micro-batch（系統參數，影響 peak memory / TTFT，不影響選擇結果）
CHUNK_SIZE = 1024           # LLM chunked-prefill 大小（同上）


def clean_answer(answers):
    for i, answer in enumerate(answers):
        answer = answer.replace("<|im_end|>", "").strip()
        answers[i] = answer

    return answers


# ──────────────────────────────────────────────────────────────────────────────
# 標準 generate 參考路徑（用於正確性比較）
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer_standard(model, tokenizer, pixel_values_list, questions, generation_config=None):
    """走官方 model.batch_chat() 路徑。batch_chat 本身就是為不同圖片/不同 tile 數設計的，
    不需要 padding：把每張圖的 tiles 直接沿 batch 維度 cat 起來，用 num_patches_list
    告訴模型每張圖各自佔幾個 tile 即可。"""

    assert len(pixel_values_list) == len(questions)

    if generation_config is None:
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
def build_prompt(tokenizer, model, question: str, num_image_tiles: int) -> str:
    """改用模型自己的 conversation template，跟 model.chat()/batch_chat() 內部
    組 prompt 的方式逐行一致，確保 baseline 與 streaming 路徑吃到完全相同的
    system message + 角色格式，只有 image token 數量依 budget 換算不同。"""
    model_module = importlib.import_module(type(model).__module__)
    get_conv_template = model_module.get_conv_template  # 跟 chat() 內部用的是同一個函式

    template = get_conv_template(model.template)
    template.system_message = model.system_message

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * model.num_image_token * num_image_tiles
        + IMG_END_TOKEN
    )
    # 跟 generate_answer_standard 裡 `f'<image>\n{q}'` 的慣例一致：
    # 先放 placeholder，再用 replace 換成真正的 image token block，
    # 確保 image token 前後的文字排版跟官方路徑一模一樣。
    template.append_message(template.roles[0], f'<image>\n{question}')
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    query = query.replace('<image>', image_tokens, 1)

    return query


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
# 串流核心：單一 tile 的 ViT → pixel_shuffle → mlp1
# ──────────────────────────────────────────────────────────────────────────────
def encode_tile(model, tile_pixel_values: torch.Tensor, dtype) -> torch.Tensor:
    """
    輸入  : tile_pixel_values [B_tile, 3, 448, 448]  (B_tile 通常 = vit_micro_batch)
    輸出  : [B_tile, num_image_token, D_llm]  (已經過 pixel_shuffle + mlp1，可直接餵 LLM)

    這個函式刻意保持「無狀態、單一 tile-batch 進、單一 tile-batch 出」，
    是讓 ViT 編碼能跟 chunked LLM prefill 交錯執行的關鍵。
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
# LLM 前段 attention 探針：thumbnail prior 與 cascade 的 q_attn 共用
#   只跑到需要的那一層就停；attention 權重只手算需要的那幾列（不材料化 [H, L, L]），
#   真正的 layer forward 仍走原本的 kernel（flash-attn 相容，不必切 eager）。
# ──────────────────────────────────────────────────────────────────────────────
def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """同 transformers qwen3 的 rotate_half（放在這裡，讓 torch.compile 的範圍內不必 import）。"""
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """同 transformers qwen3 的 repeat_kv：[B, Hkv, S, D] → [B, Hkv*n_rep, S, D]。"""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def _head_mean_attn_rows(decoder_layer, hidden_states, position_embeddings, q_start: int) -> torch.Tensor:
    """這一層 attention 的第 q_start..L-1 列（head 平均），回傳 [B, nq, L]。"""

    sa = decoder_layer.self_attn
    normed = decoder_layer.input_layernorm(hidden_states)
    B, L, _ = normed.shape
    hd = sa.head_dim
    q = sa.q_norm(sa.q_proj(normed[:, q_start:]).view(B, L - q_start, -1, hd)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(normed).view(B, L, -1, hd)).transpose(1, 2)
    cos, sin = position_embeddings
    q = q * cos[:, q_start:].unsqueeze(1) + _rotate_half(q) * sin[:, q_start:].unsqueeze(1)
    k = k * cos.unsqueeze(1) + _rotate_half(k) * sin.unsqueeze(1)
    k = _repeat_kv(k, sa.num_key_value_groups)
    logits = (q.float() @ k.float().transpose(-2, -1)) * sa.scaling          # [B, H, nq, L]
    q_pos = torch.arange(q_start, L, device=logits.device).view(1, 1, -1, 1)
    k_pos = torch.arange(L, device=logits.device).view(1, 1, 1, -1)
    return logits.masked_fill(k_pos > q_pos, float("-inf")).softmax(dim=-1).mean(dim=1)


def _layer_prefix_kv(decoder_layer, hidden_states, position_embeddings, length: int):
    """這一層 attention 對前 length 個位置算出的 (K, V)，K 已套 k_norm + RoPE，跟 layer 內部
    算的一樣（未 repeat_kv）。回傳 ([B, Hkv, length, hd], [B, Hkv, length, hd])。"""

    sa = decoder_layer.self_attn
    x = decoder_layer.input_layernorm(hidden_states[:, :length])
    B = x.shape[0]
    hd = sa.head_dim
    k = sa.k_norm(sa.k_proj(x).view(B, length, -1, hd)).transpose(1, 2)
    v = sa.v_proj(x).view(B, length, -1, hd).transpose(1, 2)
    cos, sin = position_embeddings
    k = k * cos[:, :length].unsqueeze(1) + _rotate_half(k) * sin[:, :length].unsqueeze(1)
    return k, v


@torch.no_grad()
def prefix_attention_rows(model, h: torch.Tensor, q_start: int, layers,
                          kv_sink: list | None = None, kv_len: int = 0, kv_upto: int = 0) -> dict:
    """h: [B, L, D] inputs_embeds（position 從 0 起）。跑第 1..max(layers) 層，對每個
    l ∈ layers（1-indexed）回傳該層 attention 的 q_start.. 列（head 平均）{l: [B, nq, L]}。
    第 max(layers) 層只算 q/k，不跑它的 attention 輸出與 MLP；更後面的層完全不跑。

    kv_sink 不是 None 時：順便把第 1..kv_upto 層對「前 kv_len 個位置」的 (K, V) 依序 append 進
    kv_sink（給 qattn_scores_with_prefix 重用）。kv_upto 超過 max(layers) 時會多跑到 kv_upto 層。"""
    from transformers.masking_utils import create_causal_mask

    layers = set(int(l) for l in layers)
    lm_model = model.language_model.model
    stop = max(layers) if kv_sink is None else max(max(layers), int(kv_upto))
    stop = min(stop, len(lm_model.layers))
    B, L, _ = h.shape
    position_ids = torch.arange(L, device=h.device).unsqueeze(0).expand(B, -1)
    cache_position = torch.arange(L, device=h.device)
    mask = create_causal_mask(config=lm_model.config, input_embeds=h, attention_mask=None,
                              cache_position=cache_position, past_key_values=None,
                              position_ids=position_ids)
    pos_emb = lm_model.rotary_emb(h, position_ids)
    out = {}
    for li, decoder_layer in enumerate(lm_model.layers[:stop], start=1):
        if li in layers:
            out[li] = _head_mean_attn_rows(decoder_layer, h, pos_emb, q_start)
        if kv_sink is not None and li <= kv_upto:
            kv_sink.append(_layer_prefix_kv(decoder_layer, h, pos_emb, kv_len))
        if li < stop:
            h = decoder_layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=None,
                              use_cache=False, cache_position=cache_position, position_embeddings=pos_emb)
            if isinstance(h, tuple):
                h = h[0]
    return out


def _layer_forward_with_prefix(decoder_layer, h, position_embeddings, prefix_k, prefix_v):
    """手寫的 Qwen3 decoder layer forward：h 的每個位置除了 causal 看自己之前的位置，也看得到
    整段 prefix（prefix_k / prefix_v：[1, Hkv, P, hd]，已套 k_norm + RoPE）。回傳下一層的 h。"""

    sa = decoder_layer.self_attn
    B, T, _ = h.shape
    hd = sa.head_dim
    x = decoder_layer.input_layernorm(h)
    q = sa.q_norm(sa.q_proj(x).view(B, T, -1, hd)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
    v = sa.v_proj(x).view(B, T, -1, hd).transpose(1, 2)
    cos, sin = position_embeddings
    q = q * cos.unsqueeze(1) + _rotate_half(q) * sin.unsqueeze(1)
    k = k * cos.unsqueeze(1) + _rotate_half(k) * sin.unsqueeze(1)
    P = prefix_k.shape[2]
    k = _repeat_kv(torch.cat([prefix_k.expand(B, -1, -1, -1), k], dim=2), sa.num_key_value_groups)
    v = _repeat_kv(torch.cat([prefix_v.expand(B, -1, -1, -1), v], dim=2), sa.num_key_value_groups)
    allowed = torch.ones(T, P + T, dtype=torch.bool, device=h.device)
    allowed[:, P:] = torch.ones(T, T, dtype=torch.bool, device=h.device).tril()
    attn = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=sa.scaling)
    h = h + sa.o_proj(attn.transpose(1, 2).reshape(B, T, -1))
    return h + decoder_layer.mlp(decoder_layer.post_attention_layernorm(h))


def _head_mean_attn_rows_with_prefix(decoder_layer, h, position_embeddings, prefix_k, q_start: int):
    """同 _head_mean_attn_rows，但 key = [prefix ; h]。回傳 [B, nq, P + T]（head 平均）。"""

    sa = decoder_layer.self_attn
    B, T, _ = h.shape
    hd = sa.head_dim
    x = decoder_layer.input_layernorm(h)
    q = sa.q_norm(sa.q_proj(x[:, q_start:]).view(B, T - q_start, -1, hd)).transpose(1, 2)
    k = sa.k_norm(sa.k_proj(x).view(B, T, -1, hd)).transpose(1, 2)
    cos, sin = position_embeddings
    q = q * cos[:, q_start:].unsqueeze(1) + _rotate_half(q) * sin[:, q_start:].unsqueeze(1)
    k = k * cos.unsqueeze(1) + _rotate_half(k) * sin.unsqueeze(1)
    P = prefix_k.shape[2]
    k = _repeat_kv(torch.cat([prefix_k.expand(B, -1, -1, -1), k], dim=2), sa.num_key_value_groups)
    logits = (q.float() @ k.float().transpose(-2, -1)) * sa.scaling          # [B, H, nq, P+T]
    q_pos = torch.arange(P + q_start, P + T, device=h.device).view(1, 1, -1, 1)
    k_pos = torch.arange(P + T, device=h.device).view(1, 1, 1, -1)
    return logits.masked_fill(k_pos > q_pos, float("-inf")).softmax(dim=-1).mean(dim=1)


def _qattn_prefix_core(lm_model, layer: int, P: int, tile_tokens: torch.Tensor, emb_after: torch.Tensor,
                       prefix_k: tuple, prefix_v: tuple) -> torch.Tensor:
    """q_attn（含 thumbnail context）的純張量核心。
    回傳 [B, A]：text_after 每個位置（A 個）對 tile 256 token 的 attention 總和（head 平均）。"""
    B, n_tok, _ = tile_tokens.shape
    h = torch.cat([tile_tokens.to(emb_after.dtype), emb_after.expand(B, -1, -1)], dim=1)
    T = h.shape[1]
    position_ids = torch.arange(P, P + T, device=h.device).unsqueeze(0).expand(B, -1)
    pos_emb = lm_model.rotary_emb(h, position_ids)
    for li in range(layer - 1):
        h = _layer_forward_with_prefix(lm_model.layers[li], h, pos_emb, prefix_k[li], prefix_v[li])
    rows = _head_mean_attn_rows_with_prefix(lm_model.layers[layer - 1], h, pos_emb, prefix_k[layer - 1], n_tok)
    return rows[..., P: P + n_tok].sum(dim=-1)


@torch.no_grad()
def qattn_scores_with_prefix(model, tile_tokens: torch.Tensor, prefix_kv: dict, emb_after: torch.Tensor,
                             layer: int = QATTN_LAYER) -> torch.Tensor:
    """q_attn（含覆蓋 tile context）：context = [text_before][覆蓋 tile][tile][text_after]，
    [text_before][覆蓋 tile] 用 prefix_kv（coverage_prior 存的 {"len": P, "kv": [(K, V), ...]}，
    第 1..layer 層）不重算；只對 [tile][text_after] 跑前 layer-1 層 + 第 layer 層的 q/k。
    數學上等於整段直接 forward。回傳 [B]。"""
    lm_model = model.language_model.model
    P, kv = int(prefix_kv["len"]), prefix_kv["kv"]
    if len(kv) < layer:
        raise ValueError(f"prefix_kv 只有 {len(kv)} 層，q_attn 要用到第 {layer} 層")
    ks = tuple(k for k, _ in kv[:layer])
    vs = tuple(v for _, v in kv[:layer])
    return _qattn_prefix_core(lm_model, layer, P, tile_tokens, emb_after, ks, vs).mean(dim=-1).float()


# ══════════════════════════════════════════════════════════════════════════════
# InternVL adapter
# ══════════════════════════════════════════════════════════════════════════════
class InternVLAdapter:
    """保留 InternVL 架構上的核心優勢－tile 之間彼此獨立（無 cross-tile 依賴），
    可以逐 tile encode → add_tile，不需要等所有 tile 到齊才開始處理。"""

    def __init__(self, tokenizer, model, budget: int):
        # InternVL 的 prompt 里 <IMG_CONTEXT> 数量取决于 equiv_tiles = budget
        # 换算出的等效 tile 数，而这个数字在建 prompt 当下就要知道，
        # 所以在建构 adapter 时一併带入 model / budget。
        self.tokenizer = tokenizer
        self.model = model
        self.budget = budget

    def build_text_segments(self, questions):
        num_image_token = self.model.num_image_token  # num_image_token=256
        equiv_tiles = -(-self.budget // num_image_token)  # ceil division

        text_before_list, text_after_list = [], []
        for question in questions:
            prompt = build_prompt(self.tokenizer, self.model, question, equiv_tiles)
            tb, ta = split_prompt_at_vision(prompt, self.tokenizer)
            text_before_list.append(tb)
            text_after_list.append(ta)
        return text_before_list, text_after_list

    def get_tokenizer(self):
        return self.tokenizer

    def get_input_embeddings(self, model):
        return model.language_model.get_input_embeddings()

    def lm_prefill(self, model, **kwargs):
        # prefill 只需要 past_key_values，不需要 logits。直接呼叫底層 base model
        # 跳過 lm_head，省掉 (B, chunk_size, vocab) 這個算完馬上被丟掉的巨大暫時張量。
        lm = model.language_model
        base = getattr(lm, "model", None)
        with torch.no_grad():
            if base is not None:
                return base(**kwargs)
            # 後備路徑：拿不到 base model 時，至少只算最後 1 個位置的 logits
            return lm(**kwargs, logits_to_keep=1)

    def lm_decode_step(self, model, **kwargs):
        with torch.no_grad():
            out = model.language_model(**kwargs)
        return out.logits[:, -1, :], out.past_key_values

    @torch.no_grad()
    def coverage_prior(self, model, cover_tokens, cover_grid, detail_grid, text_before, text_after,
                       capture_kv_upto: int | None = None):
        """覆蓋 tile 跑一次 LLM 前段 forward：[text_before][m 塊覆蓋 tile][text_after]，取第
        THUMB_ATTN_LAYERS=(lo, hi) 層（1-indexed、含頭尾）text_after 所有位置對 m*256 個覆蓋 patch 的
        attention → 每個 patch 依它在全圖的位置，累積到它中心落在的細節 tile → 每塊細節 tile 一個 prior。
        m = 1（覆蓋 tile = 官方縮圖）時就是原本的 thumbnail prior。

        cover_tokens：[m, 256, D]，逐列排列、網格 cover_grid=(cols, rows)；detail_grid：細節網格 (cols, rows)。
        回傳 (prior, prefix_kv)：prior = {細節 tile index -> [0,1]}；capture_kv_upto=L 時 prefix_kv =
        [text_before][覆蓋 tile] 第 1..L 層的 {"len": P, "kv": [(K, V), ...]}，給 q_attn 重用，否則 None。

        成本：一次 forward，長度 ≈ len(text_before) + m*256 + len(text_after)，只跑到第 hi 層；
        跟細節 tile 總數無關；attention 只手算需要的列（不切 eager、不存 [H, L, L]）。
        """
        m, n_patch, _ = cover_tokens.shape
        side = int(round(n_patch ** 0.5))                                 # 16
        cc, cr = cover_grid
        cols, rows = detail_grid
        span = m * n_patch

        # 1) text_before + 覆蓋 tile + text_after 的 inputs_embeds
        emb_layer = self.get_input_embeddings(model)
        tb = self.tokenizer([text_before], return_tensors="pt").to(DEVICE)
        ta = self.tokenizer([text_after], return_tensors="pt").to(DEVICE)
        emb_b = emb_layer(tb.input_ids)
        emb_a = emb_layer(ta.input_ids)
        inp = torch.cat([emb_b, cover_tokens.reshape(1, span, -1), emb_a], dim=1)
        span_start = emb_b.shape[1]
        q_start = span_start + span

        # 2) 第 lo..hi 層、text_after 對覆蓋 patch 的 attention，平均 row/layer（head 已平均）
        n_layers = len(model.language_model.model.layers)
        lo, hi = THUMB_ATTN_LAYERS[0], min(THUMB_ATTN_LAYERS[1], n_layers)
        kv_sink = [] if capture_kv_upto else None
        per_layer = prefix_attention_rows(model, inp, q_start, range(lo, hi + 1),
                                          kv_sink=kv_sink, kv_len=q_start,
                                          kv_upto=int(capture_kv_upto or 0))
        prefix_kv = {"len": q_start, "kv": kv_sink} if kv_sink is not None else None
        sal = torch.stack([
            per_layer[l][0, :, span_start: q_start].mean(dim=0)                  # [m * n_patch]
            for l in range(lo, hi + 1)
        ], dim=0).mean(dim=0)
        del per_layer, inp, emb_b, emb_a
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        sal = sal.reshape(m, side, side).cpu()                                   # 每塊 [16, 16] row-major

        # 3) 每個覆蓋 patch 的中心（全圖座標）落在哪塊細節 tile → 累積該 tile 的 saliency
        acc = [0.0] * (cols * rows)
        cnt = [0] * (cols * rows)
        for t in range(m):
            tr, tc = divmod(t, cc)
            for pr in range(side):
                for pc in range(side):
                    gc = min(cols - 1, int((tc * side + pc + 0.5) / (cc * side) * cols))
                    gr = min(rows - 1, int((tr * side + pr + 0.5) / (cr * side) * rows))
                    k = gr * cols + gc
                    acc[k] += float(sal[t, pr, pc])
                    cnt[k] += 1
        prior = [acc[k] / cnt[k] if cnt[k] else 0.0 for k in range(cols * rows)]
        lo, hi = min(prior), max(prior)
        prior = [(p - lo) / (hi - lo + 1e-6) for p in prior]                    # 再正規化到 [0,1]
        return {k: prior[k] for k in range(cols * rows)}, prefix_kv

    @torch.no_grad()
    def _commit_tiles(self, model, selected, chunk_size, past_key_values, running_mask):
        """Phase 2：把已定案的 tile（呼叫端已依 tile_index 排序）chunked prefill 進 KV cache，
        position 接在 text_before 後面連續編號。回傳 (past_key_values, running_mask, tokens_injected, kv_flushes)。"""
        num_image_token = model.num_image_token
        tokens_injected = 0
        kv_flushes = 0

        def prefill_vision(vtok_2d, start_pos):
            """Write one [c, D] block of committed visual tokens into the KV cache,
            at positions start_pos .. start_pos+c-1."""
            nonlocal past_key_values, running_mask, tokens_injected, kv_flushes
            vchunk = vtok_2d.unsqueeze(0)  # [1, c, D]
            c = vchunk.shape[1]
            mchunk = torch.ones((1, c), dtype=torch.long, device=DEVICE)
            running_mask = torch.cat([running_mask, mchunk], dim=1)
            position_ids = torch.arange(
                start_pos, start_pos + c, dtype=torch.long, device=DEVICE,
            ).unsqueeze(0)
            out = self.lm_prefill(
                model,
                inputs_embeds=vchunk,
                attention_mask=running_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = out.past_key_values
            tokens_injected += c
            kv_flushes += 1
            del out, vchunk, mchunk, position_ids

        tiles_per_flush = max(1, chunk_size // num_image_token)
        for s in range(0, len(selected), tiles_per_flush):
            group = selected[s:s + tiles_per_flush]
            prefill_vision(torch.cat(group, dim=0), past_key_values.get_seq_length())
        return past_key_values, running_mask, tokens_injected, kv_flushes

    @torch.no_grad()
    def stream_image_to_kv_cascade(
        self,
        model,
        detail_px,
        cover_tokens,
        text_before,
        text_after,
        tile_prior,
        budget,
        qattn_tiles,
        past_key_values,
        running_mask,
        prefix_kv=None,
        dtype=torch.bfloat16,
    ):
        """多尺度 tile 選擇：m 塊覆蓋 tile 一定保留，細節 tile 用 prior 粗排 → 邊界打分選 K - m 塊。
        仍是 streaming、tile 級選擇、不加模型。m = 1 時就是原本的 SVMA（覆蓋 tile = 縮圖）。

        K = budget // 256，m = cover_tokens.shape[0]，細節名額 slots = K - m。S = qattn_tiles：
          a = min(slots, S // 2)，b = S - a（svma_config.split_slots）
          1. prior 排名前 (slots-a) 塊：直接保留，只跑 ViT、不打分
          2. 接下來 S 塊（邊界附近）：跑 ViT + q_attn，取分數最高的 a 塊
          3. 其餘細節 tile：直接丟，連 ViT 都不跑
        q_attn 只跑 S 塊，不隨 budget 增加；S = 0 = 只用 prior 取前 slots 塊。

        串流：直接保留與邊界的 tile 依 tile_index（空間順序）逐 micro-batch 跑 ViT；邊界 tile 打完分
        進大小 a 的候選池，邊串流邊淘汰最低分（平手淘汰 tile_index 較大者）。q_attn 分數只看
        「這塊 tile + 問題」，是靜態的，所以串流淘汰的結果 = 邊界中的全域 top-a。
        寫 KV 的順序：選中的細節 tile 依空間順序在前，m 塊覆蓋 tile 在最後（同訓練時「網格 tile → 縮圖」）。
        常駐視覺 token 上限 = slots + VIT_BATCH 塊細節 tile + m 塊覆蓋 tile。

        q_attn 的打分 context = [text_before][覆蓋 tile][tile][text_after]；prefix_kv = coverage_prior
        存的前綴 KV，沒給就在這裡補算（多一個短 forward）。
        """
        num_image_token = model.num_image_token
        K = budget // num_image_token
        m = cover_tokens.shape[0]
        if not 1 <= m <= K:
            raise ValueError(f"coverage tiles m={m} 要在 1..K={K} 之間（budget={budget}）")
        slots = K - m
        n_detail = detail_px.shape[0]
        prior = tile_prior or {}
        ranked = sorted(range(n_detail), key=lambda i: (-float(prior.get(i, 0.0)), i))
        a, b = split_slots(slots, qattn_tiles)
        keep_direct = set(ranked[: slots - a])                # 直接保留：只跑 ViT，不打分
        window = ranked[slots - a: slots + b]                 # 邊界：跑 q_attn，取 a 塊
        cand = sorted(keep_direct | set(window))
        need_score = len(window) > a

        if need_score:
            emb = self.get_input_embeddings(model)
            emb_a = emb(self.tokenizer([text_after], return_tensors="pt").input_ids.to(DEVICE))
            if prefix_kv is None or len(prefix_kv["kv"]) < QATTN_LAYER:
                emb_b = emb(self.tokenizer([text_before], return_tensors="pt").input_ids.to(DEVICE))
                pre = torch.cat([emb_b, cover_tokens.reshape(1, -1, cover_tokens.shape[-1]).to(emb_b.dtype)], dim=1)
                sink = []
                prefix_attention_rows(model, pre, pre.shape[1] - 1, {QATTN_LAYER},
                                      kv_sink=sink, kv_len=pre.shape[1], kv_upto=QATTN_LAYER)
                prefix_kv = {"len": pre.shape[1], "kv": sink}
                del pre, emb_b

        heap = []            # (score, -tile_index, tile_index, tokens)：堆頂 = 最該淘汰的
        direct = []          # 直接保留的 (tile_index, tokens)
        scores = {}
        max_resident = 0
        for s in range(0, len(cand), VIT_BATCH):
            idxs = cand[s:s + VIT_BATCH]
            chunk = detail_px[idxs].to(DEVICE, dtype=dtype)
            toks = encode_tile(model, chunk, dtype=dtype)                   # [b, n_tok, D]
            del chunk
            max_resident = max(max_resident, len(heap) + len(direct) + len(idxs) + m)
            js = [j for j, idx in enumerate(idxs) if idx not in keep_direct]   # 邊界 tile
            for j, idx in enumerate(idxs):
                if idx in keep_direct:
                    direct.append((idx, toks[j].clone()))
            if js:
                sub = toks[js] if len(js) < len(idxs) else toks
                if need_score:
                    sc = qattn_scores_with_prefix(model, sub, prefix_kv, emb_a).tolist()
                else:
                    sc = [float(prior.get(idxs[j], 0.0)) for j in js]
                for v, j in zip(sc, js):
                    idx = idxs[j]
                    scores[idx] = float(v)
                    heapq.heappush(heap, (float(v), -idx, idx, toks[j].clone()))
                    if len(heap) > a:
                        heapq.heappop(heap)
            del toks

        del prefix_kv
        items = sorted([(idx, tok) for _, _, idx, tok in heap] + direct, key=lambda x: x[0])
        del heap, direct
        sel_idx = [idx for idx, _ in items]
        selected = [tok for _, tok in items] + [cover_tokens[t] for t in range(m)]
        del items
        past_key_values, running_mask, tokens_injected, kv_flushes = self._commit_tiles(
            model, selected, CHUNK_SIZE, past_key_values, running_mask,
        )
        del selected

        stats = {
            "total_seen_tiles": n_detail + m,
            "coverage_tiles": m,
            "vit_encoded_tiles": len(cand) + m,
            "qattn_tiles": int(qattn_tiles),
            "boundary_a_b": [a, b],
            "direct_kept": sorted(keep_direct),
            "window": sorted(window),
            "candidate_scores": {int(k): v for k, v in scores.items()},
            "capacity_tiles": K,
            "total_dropped_tiles": n_detail - len(sel_idx),
            "committed_tile_indices": sel_idx,
            "max_resident_tiles": max_resident,
            "kv_flushes": kv_flushes,
            "final_size": tokens_injected,
            "final_tiles": tokens_injected // num_image_token,
        }
        assert tokens_injected <= budget, f"KV visual tokens {tokens_injected} > budget {budget}"
        return past_key_values, running_mask, stats

    def generate_baseline(self, model, **kwargs):
        pixel_values_list = kwargs["pixel_values_list"]
        questions = kwargs["questions"]
        return generate_answer_standard(model, self.tokenizer, pixel_values_list, questions)

    def clean_answer(self, batch_or_questions, answers):
        return clean_answer(answers)


# ══════════════════════════════════════════════════════════════════════════════
# SVMA 主流程：text_before prefill → tile 選擇 + 寫 KV → text_after + decode
# ══════════════════════════════════════════════════════════════════════════════
def run_online_kv_with_memory_bank(
    model,
    adapter: InternVLAdapter,
    inputs: list[dict],
    questions: list[str],
    budget: int = 1024,
    qattn_tiles: int | None = None,
    max_new_tokens: int = MAX_NEW_TOKENS,
    dtype: torch.dtype = torch.bfloat16,
):
    """SVMA：每張圖 text_before prefill → 覆蓋 tile 的 ViT + prior → InternVLAdapter.stream_image_to_kv_cascade
    （邊界打分選細節 tile、細節 tile + 覆蓋 tile 依序寫 KV）→ text_after + greedy decode。

    inputs：每張圖一個 internvl_preprocess.load_image_multiscale 的 dict（cover / cover_grid / detail /
    detail_grid）；覆蓋 tile 數 m = cover 的塊數，m = 1 就是原本的 SVMA。
    qattn_tiles：S；None = 每張圖依 budget 與實際覆蓋塊數自動（svma_config.default_qattn_tiles）。
    budget 是最終 visual token 上限，InternVL3.5-8B budget=2048 = 最多 8 塊 256-token tile（含 m 塊覆蓋 tile）。
    一次只處理一張圖（batch 內逐張），全部 tile token 從不同時存在。
    回傳 (answers, all_stats, timing)。
    """
    B = len(inputs)
    assert B == len(questions)
    tokenizer = adapter.get_tokenizer()

    torch.cuda.synchronize(DEVICE)
    t_start = time.time()

    _mem_phases = {}

    def _mem_reset():
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)

    def _mem_mark(name):
        torch.cuda.synchronize(DEVICE)
        _mem_phases[name] = {
            "peak": torch.cuda.max_memory_allocated(DEVICE) / 1e9,
            "resting": torch.cuda.memory_allocated(DEVICE) / 1e9,
        }

    all_answers = []
    all_stats = []
    first_token_times = []
    vision_times = []
    kv_times = []

    for b in range(B):
        cover_px, detail_px = inputs[b]["cover"], inputs[b]["detail"]
        m = cover_px.shape[0]
        question = questions[b]
        S_b = default_qattn_tiles(budget, m) if qattn_tiles is None else qattn_tiles

        print("\n" + "=" * 80)
        print(f"[Image {b}] detail={inputs[b]['detail_grid']} cover={inputs[b]['cover_grid']} "
              f"budget={budget} S={S_b}")
        print("=" * 80)

        # --------------------------------------------------------
        # Step A: text-before prefill
        # --------------------------------------------------------
        _mem_reset()
        prompt = build_prompt(
            tokenizer,
            model,
            question,
            max(1, (budget + model.num_image_token - 1) // model.num_image_token),
        )
        text_before, text_after = split_prompt_at_vision(prompt, tokenizer)

        tok_before = tokenizer(
            [text_before],
            return_tensors="pt",
            padding=True,
            padding_side="right",
        )
        ids_before = tok_before.input_ids.to(DEVICE)
        mask_before = tok_before.attention_mask.to(DEVICE)
        embeds_before = adapter.get_input_embeddings(model)(ids_before)

        out = adapter.lm_prefill(
            model,
            inputs_embeds=embeds_before,
            attention_mask=mask_before,
            past_key_values=None,
            use_cache=True,
            return_dict=True,
        )
        past_key_values = out.past_key_values
        running_mask = mask_before

        del out, embeds_before, ids_before, tok_before
        _mem_mark("Prefill")

        # --------------------------------------------------------
        # Step B: true online Vision -> TileMemory -> KV
        # --------------------------------------------------------
        _mem_reset()
        torch.cuda.synchronize(DEVICE)
        t_vision_start = time.time()

        # 覆蓋 tile：ViT 編碼後常駐（最後寫進 KV），同時當 prior 的輸入
        cover_tokens = torch.cat([
            encode_tile(model, cover_px[i:i + VIT_BATCH].to(DEVICE, dtype=dtype), dtype=dtype)
            for i in range(0, m, VIT_BATCH)
        ], dim=0)                                                         # [m, 256, D]

        # prior：一次 [text_before][覆蓋 tile][text_after] 短 forward 算出每塊細節 tile 的 prior；
        # S > 0 時順便留下 [text_before][覆蓋 tile] 的前綴 KV 給 q_attn 重用。沒有細節名額 / 細節 tile 就跳過。
        tile_prior, prefix_kv = None, None
        if detail_px.shape[0] > 0 and budget // model.num_image_token > m:
            t_ta = time.time()
            tile_prior, prefix_kv = adapter.coverage_prior(
                model, cover_tokens, inputs[b]["cover_grid"], inputs[b]["detail_grid"],
                text_before, text_after,
                capture_kv_upto=QATTN_LAYER if S_b > 0 else None,
            )
            print(f"[Image {b}] coverage prior: m={m} +{time.time() - t_ta:.2f}s")

        past_key_values, running_mask, stats = adapter.stream_image_to_kv_cascade(
            model=model,
            detail_px=detail_px,
            cover_tokens=cover_tokens,
            text_before=text_before,
            text_after=text_after,
            tile_prior=tile_prior,
            budget=budget,
            qattn_tiles=S_b,
            past_key_values=past_key_values,
            running_mask=running_mask,
            prefix_kv=prefix_kv,
            dtype=dtype,
        )
        del cover_tokens, prefix_kv
        stats["cover_grid"] = list(inputs[b]["cover_grid"])
        stats["detail_grid"] = list(inputs[b]["detail_grid"])
        a_, b_ = stats["boundary_a_b"]
        print(f"[Image {b}] boundary S={S_b} (a={a_}, b={b_}): direct={stats['direct_kept']} "
              f"window={stats['window']} vit_encoded={stats['vit_encoded_tiles']}/{stats['total_seen_tiles']}")

        torch.cuda.synchronize(DEVICE)
        t_vision_end = time.time()
        vision_times.append(t_vision_end - t_vision_start)

        _mem_mark("Vision+OnlineKV")
        _mem_reset()

        print(
            f"[Image {b}] raw={stats['total_seen_tiles']} tiles -> "
            f"KV={stats['final_tiles']} tiles / {stats['final_size']} tokens; "
            f"dropped={stats['total_dropped_tiles']}; "
            f"max_resident={stats['max_resident_tiles']} tiles"
        )
        print(f"[Image {b}] selected_tile_indices={stats['committed_tile_indices']}")

        # KV cache（analytical，跟 SparseVLM 論文同一種量測方式，見
        # estimate_kv_cache_mb() docstring）：這個方法在進 LLM 前就把 token 數
        # 固定了，所有層長度一樣，直接用 stats['final_size'] 乘上層數即可，
        # 不用像 SparseVLM 那樣逐層查 past_key_values（每層長度不同）。
        tb_len, ta_len = estimate_prompt_token_lengths(tokenizer, text_before, text_after)
        total_len = tb_len + stats["final_size"] + ta_len
        num_layers = model.language_model.config.num_hidden_layers
        kv_cache_mb = estimate_kv_cache_mb(model, [total_len] * num_layers)
        stats["kv_cache_mb"] = round(kv_cache_mb, 2)
        print(f"[Image {b}] KV cache (prompt-only, analytical) = {kv_cache_mb:.1f} MB "
              f"({total_len} tokens x {num_layers} layers)")

        # --------------------------------------------------------
        # Step C: text-after + autoregressive decode
        # --------------------------------------------------------
        tok_after = tokenizer([text_after], return_tensors="pt", padding=True, padding_side="right")
        ids_after = tok_after.input_ids.to(DEVICE)
        mask_after = tok_after.attention_mask.to(DEVICE)

        # visual token 連續編號，text_after 與之後每個 decode token 接在 KV cache 長度後面
        next_pos = past_key_values.get_seq_length()
        running_mask = torch.cat([running_mask, mask_after], dim=1)
        position_ids = torch.arange(next_pos, next_pos + ids_after.shape[1], dtype=torch.long, device=DEVICE).unsqueeze(0)
        next_pos += ids_after.shape[1]

        torch.cuda.synchronize(DEVICE)
        t_decode_start = time.time()

        next_token_logits, past_key_values = adapter.lm_decode_step(
            model,
            input_ids=ids_after,
            attention_mask=running_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

        torch.cuda.synchronize(DEVICE)
        t_first_token = time.time()
        first_token_times.append(t_first_token - t_start)

        del ids_after, mask_after, position_ids, next_token_logits, tok_after

        eos_token_id = tokenizer.eos_token_id
        answer = tokenizer.decode(next_token[0], skip_special_tokens=False)
        finished = bool(
            eos_token_id is not None
            and next_token[0, 0].item() == eos_token_id
        )

        for step in range(max_new_tokens):
            if finished:
                break

            running_mask = torch.cat([running_mask, torch.ones((1, 1), dtype=torch.long, device=DEVICE)], dim=1)
            position_ids = torch.full((1, 1), next_pos, dtype=torch.long, device=DEVICE)
            next_pos += 1

            next_token_logits, past_key_values = adapter.lm_decode_step(
                model,
                input_ids=next_token,
                attention_mask=running_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token[0, 0].item()
            if not finished:
                answer += tokenizer.decode([token_id], skip_special_tokens=False)

            if eos_token_id is not None and token_id == eos_token_id:
                finished = True

            del next_token_logits, position_ids

        torch.cuda.synchronize(DEVICE)
        t_decode_end = time.time()
        kv_times.append(t_decode_start - t_vision_end)

        all_answers.append(answer)
        all_stats.append(stats)

        _mem_mark("Decode")

        del past_key_values, next_token, running_mask
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------
    # Peak/timing summary
    # ------------------------------------------------------------
    _overall_peak = max([v["peak"] for v in _mem_phases.values()] or [0.0])

    print("\n[Peak Memory Breakdown]")
    for name in ("Prefill", "Vision+OnlineKV", "Decode"):
        if name not in _mem_phases:
            continue
        p = _mem_phases[name]
        print(
            f"  {name:<18s} peak={p['peak']:6.2f} GB "
            f"resting_end={p['resting']:6.2f} GB "
            f"transient=+{p['peak'] - p['resting']:5.2f} GB"
        )
    print(f"  {'OVERALL':<18s} peak={_overall_peak:6.2f} GB")

    timing = {
        "TTFT": round(first_token_times[0] if first_token_times else 0.0, 3),
        "Vision+OnlineKV": round(sum(vision_times), 3),
        "First Token Forward": round(sum(kv_times), 3),
    }

    timing["_peak_mem_gb"] = {
        "Prefill": round(_mem_phases.get("Prefill", {}).get("peak", 0.0), 2),
        "Vision+OnlineKV": round(_mem_phases.get("Vision+OnlineKV", {}).get("peak", 0.0), 2),
        "Decode": round(_mem_phases.get("Decode", {}).get("peak", 0.0), 2),
        "OVERALL": round(_overall_peak, 2),
    }

    # KV cache（analytical，見 estimate_kv_cache_mb()）：batch 內每張圖各自
    # 存在 stats["kv_cache_mb"]（已經在 all_stats 裡），這裡另外彙總一個
    # batch 平均，跟 _peak_mem_gb 一樣掛在 timing 底下方便單一數字看整體。
    _kv_mb_vals = [s["kv_cache_mb"] for s in all_stats if "kv_cache_mb" in s]
    if _kv_mb_vals:
        timing["_kv_cache_mb"] = round(sum(_kv_mb_vals) / len(_kv_mb_vals), 2)

    return adapter.clean_answer(questions, all_answers), all_stats, timing


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def save_results_incremental(output_path: str, results: dict):
    """每處理完一張圖就整份重寫一次 JSON。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# 評測資料集
#   新增 dataset：在 DATASETS 加一筆 name -> (loader, metric)。loader(n) 回傳 process_common
#   格式的 list[dict]（image / question / id / answer 或 answers ...），n = None 代表全部；
#   metric 決定評分方式："anls"（evaluate_anls.py）或 "hrbench_circular"（evaluate_hrbench_circular.py）。
# ══════════════════════════════════════════════════════════════════════════════
def _load_docvqa(n):
    from process_common import load_docvqa
    return load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                       num_image=n, prompt_mode=DOCVQA_PROMPT)


def _load_infovqa(n):
    from process_common import load_infovqa
    return load_infovqa(dataset="Ahren09/InfoVQA", split="val", num_image=n, prompt_mode=INFOVQA_PROMPT)


def _hrbench_loader(split):
    def load(n):
        from process_common import load_hrbench
        return load_hrbench(dataset="DreamMr/HR-Bench", split=split, num_image=n, prompt_mode=HRBENCH_PROMPT)
    return load


DATASETS = {
    "docvqa":     (_load_docvqa, "anls"),
    "infovqa":    (_load_infovqa, "anls"),
    "hrbench_4k": (_hrbench_loader("hrbench_4k"), "hrbench_circular"),
    "hrbench_8k": (_hrbench_loader("hrbench_8k"), "hrbench_circular"),
}


def load_eval_dataset(name: str, num_images: int | None = None):
    """name 是 DATASETS 的 key；"hrbench" 是舊名稱，對應環境變數 HRBENCH_SPLIT（baseline 腳本還在用）。"""
    if name == "hrbench":
        name = HRBENCH_SPLIT
    if name not in DATASETS:
        raise ValueError(f"unknown dataset {name!r}; 可用：{sorted(DATASETS)}")
    n = num_images if num_images and num_images > 0 else None
    return DATASETS[name][0](n)


def build_meta(datasets) -> list[dict]:
    """每題的評分資訊存進 JSON 的 meta（evaluate_*.py 直接讀，不用重載 HF dataset、沒有對齊風險）：
    HR-Bench 用 answer / options / category，DocVQA / InfoVQA 用 answers（多個可接受答案，ANLS 取 max）。"""
    return [{"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
             "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
             "answers": d.get("answers", [])}
            for k, d in enumerate(datasets)]


# ══════════════════════════════════════════════════════════════════════════════
# 一次跑一個 (dataset, mode, budget)：一個 JSON 只放一個 tag
#   svma         ：多尺度 SVMA（m、S 都由 budget 決定，見 svma_config.py）
#   uncompressed ：官方 batch_chat，未壓縮（跟 budget 無關）
#   lowres_fill  ：Low-Res-Fill baseline，整張圖縮成 <= K 塊 tile、不加縮圖，再走 batch_chat
# ══════════════════════════════════════════════════════════════════════════════
MODES = ("svma", "uncompressed", "lowres_fill")


def run_tag(mode: str, budget: int) -> str:
    return mode if mode == "uncompressed" else f"{mode}_b{budget}"


def default_output_json(model_name: str, dataset: str, mode: str, budget: int,
                        out_root: str = "results/runs") -> str:
    """results/runs/<dataset>/<tag>_<model>.json"""
    return os.path.join(out_root, dataset, f"{run_tag(mode, budget)}_{os.path.basename(model_name)}.json")


def load_or_init_results(path: str, meta: list[dict], tag: str) -> tuple[dict, int]:
    """讀既有的 JSON 接著跑（中斷後續跑），回傳 (results, start)。
    同一個 JSON 的題目要跟這次一樣（逐筆對齊），已完成的部分 id 不同就停下來，避免混到別批題目。"""
    results = {"candidates": {}}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            results = json.load(f)
        old_ids = [m.get("id") for m in results.get("meta", [])]
        done = len(results.get("candidates", {}).get(tag, []))
        if [m.get("id") for m in meta[:done]] != old_ids[:done]:
            raise SystemExit(f"{path} 已完成的題目跟這次載入的不同（換了 dataset / num_images？），請換一個 --output_json")
    results["meta"] = meta
    start = min(len(results.setdefault("candidates", {}).setdefault(tag, [])), len(meta))
    for key in ("candidates", "ttft", "tile_selection", "kv_cache_mb"):     # 多出來的截掉，跟 meta 對齊
        if tag in results.get(key, {}):
            results[key][tag] = results[key][tag][:start]
    return results, start


def prepare_inputs(images, mode: str, budget: int, max_num: int):
    """把一批 PIL 影像轉成各 mode 的輸入。max_num = 未壓縮時的總 tile 上限（含縮圖），
    dynamic_preprocess 會再加 1 張縮圖，所以網格上限是 max_num - 1。"""
    max_num_grid = max(1, max_num - 1)
    if mode == "svma":
        m = default_coverage_tiles(budget)
        return [load_image_multiscale(img, input_size=448, max_num_grid=max_num_grid, coverage_tiles=m)
                for img in images]
    if mode == "lowres_fill":
        return [load_image_tiles_fill(img, input_size=448, max_tiles=budget // 256) for img in images]
    return [load_image_tiles(img, input_size=448, max_num=max_num_grid) for img in images]


def standard_kv_cache_mb(model, tokenizer, pixel_values, question) -> float:
    """batch_chat 路徑拿不到 past_key_values，用 prompt 長度 + tile 數反推 KV cache（跟 SVMA 同一條公式）。"""
    tb, ta = split_prompt_at_vision(build_prompt(tokenizer, model, question, pixel_values.shape[0]), tokenizer)
    tb_len, ta_len = estimate_prompt_token_lengths(tokenizer, tb, ta)
    total = tb_len + pixel_values.shape[0] * model.num_image_token + ta_len
    return estimate_kv_cache_mb(model, [total] * model.language_model.config.num_hidden_layers)


def run_batch(model, tokenizer, adapter, mode: str, inputs, questions, budget: int, dtype=torch.bfloat16):
    """回傳 (answers, extra)；extra 是每張圖一筆的附加紀錄 {key: list}，跟 answers 對齊。"""
    if mode == "svma":
        S = default_qattn_tiles(budget, default_coverage_tiles(budget))
        answers, all_stats, timing = run_online_kv_with_memory_bank(
            model, adapter, inputs, questions, budget=budget, qattn_tiles=S, dtype=dtype)
        # batch_size > 1 時同一批共用第一張圖的 TTFT；要逐張 TTFT 就用 --batch_size 1 或 bench_metrics.py
        return answers, {"ttft": [timing.get("TTFT")] * len(answers),
                         "tile_selection": [s.get("committed_tile_indices", []) for s in all_stats],
                         "kv_cache_mb": [s.get("kv_cache_mb") for s in all_stats]}
    answers = generate_answer_standard(model, tokenizer, inputs, questions)
    return answers, {"kv_cache_mb": [round(standard_kv_cache_mb(model, tokenizer, pv, q), 2)
                                     for pv, q in zip(inputs, questions)]}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="SVMA / baseline 評測：一次跑一個 dataset × mode × budget，結果存成一個 JSON（可中斷續跑）。"
    )
    p.add_argument("--dataset", required=True, choices=sorted(DATASETS))
    p.add_argument("--mode", default="svma", choices=MODES)
    p.add_argument("--budget", type=int, default=1024, help="每張圖的 vision token 上限（K = budget // 256 塊 tile）；uncompressed 不用")
    p.add_argument("--num_images", type=int, default=-1, help="只跑前 N 題（-1 = 全部）")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    p.add_argument("--max_num", type=int, default=48, help="未壓縮時的總 tile 上限（含縮圖），也是 SVMA 細節網格的上限")
    p.add_argument("--output_json", default=None, help="預設 results/runs/<dataset>/<tag>_<model>.json")
    args = p.parse_args(argv)
    if args.mode != "uncompressed" and args.budget < 256:
        p.error(f"--budget 至少要 256（1 塊 tile），收到 {args.budget}")
    return args


def main(argv=None):
    args = parse_args(argv)
    tag = run_tag(args.mode, args.budget)
    out = args.output_json or default_output_json(args.model_name, args.dataset, args.mode, args.budget)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    torch.manual_seed(RANDOM_SEED)
    t0 = time.time()

    datasets = load_eval_dataset(args.dataset, args.num_images)
    results, start = load_or_init_results(out, build_meta(datasets), tag)
    n = len(datasets)
    if args.mode == "svma":
        m = default_coverage_tiles(args.budget)
        print(f"[svma] budget={args.budget}  K={args.budget // 256}  m={m}  S={default_qattn_tiles(args.budget, m)}")
    print(f"[{tag}] {args.dataset}: {n} 題，從第 {start + 1} 題開始 → {out}")
    if start >= n:
        print("已全部完成，跳過")
        return out

    tokenizer, model = build_model(args.model_name, dtype=torch.bfloat16, device=DEVICE)
    adapter = InternVLAdapter(tokenizer, model, budget=args.budget)
    print(f"model loaded in {time.time() - t0:.1f}s, peak alloc {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.2f} GB")

    for i in range(start, n, args.batch_size):
        batch = datasets[i:i + args.batch_size]
        questions = [d["question"] for d in batch]
        inputs = prepare_inputs([d["image"] for d in batch], args.mode, args.budget, args.max_num)
        with measure_peak_memory(f"{tag} [{i + 1}~{i + len(batch)}/{n}]"):
            answers, extra = run_batch(model, tokenizer, adapter, args.mode, inputs, questions, args.budget)
        results["candidates"][tag] += answers
        for key, vals in extra.items():
            results.setdefault(key, {}).setdefault(tag, []).extend(vals)
        for q, a in zip(questions, answers):
            print(f"  Q: {q.splitlines()[0][:100]}\n  A: {a}")
        save_results_incremental(out, results)
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\nAll done in {time.time() - t0:.1f}s → {out}")
    return out


if __name__ == "__main__":
    main()
