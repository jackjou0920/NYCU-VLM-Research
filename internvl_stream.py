# ══════════════════════════════════════════════════════════════════════════════
# InternVL 串流 KV + Memory Bank —— 單檔獨立版
#
# 這支檔案把原本分散在六支檔案裡「只跟 InternVL 有關」的部分，原封不動合併進來：
#     internvl_svm.py       → main / save_results_incremental
#     internvl_adapter.py   → InternVLAdapter
#     internvl_core.py      → build_prompt / split_prompt_at_vision / encode_tile /
#                             generate_answer_standard / measure_baseline_ttft / clean_answer
#     stream_common.py      → run_online_kv_with_memory_bank / pad_and_stack_vision
#     stream_adapters.py    → get_optimal_cuda_device / DEVICE / mean_pool_question_embed /
#                             measure_peak_memory
#     stream_memory_bank.py → TileStreamingMemoryBank + 三個 score function
#
# 仍保留 import 的外部相依：
#   internvl_preprocess.py  → build_model / load_image_tiles / dynamic_preprocess
#   process_common.py       → load_local / load_hf_dataset / load_mmmu
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import re
import gc
import time
import json
import argparse
import importlib
import contextlib

import torch
import numpy as np
import torch.nn.functional as F

from internvl_preprocess import build_model, load_image_tiles


# ══════════════════════════════════════════════════════════════════════════════
# 裝置挑選 / 記憶體量測 / question embedding      （from stream_adapters.py）
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


def mean_pool_question_embed(get_input_embeddings_fn, tokenizer, questions, dtype):
    """tokenizer -> embedding -> mask 加权平均。
    两边原本都是包在 torch.no_grad() 里算的，这里保持一致。"""
    with torch.no_grad():
        q_tok = tokenizer(questions, return_tensors="pt", padding=True).to(DEVICE)
        q_embeds_all = get_input_embeddings_fn(q_tok.input_ids)
        q_mask = q_tok.attention_mask.unsqueeze(-1).float()
        pooled = (q_embeds_all * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-6)
    return pooled.to(dtype)


def strip_to_question(text: str) -> str:
    """從「已組好的選擇題 prompt」裡只取自然語言問句本身（給 info_density 的
    relevance 項用）。原本的 relevance 把整段 prompt（問句 + "A. .." 選項區塊 +
    "Answer with the option's letter ..." 指令）mean-pool 成一個向量，指令與選項
    字母把質心稀釋成幾乎跟每個 tile 都無關。這裡砍掉第一個 "A."/"A)" 選項行、
    以及 "Answer ..." 作答指示行之後的所有東西。純問句（open 模式）會原樣回傳。
    """
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^[A-Za-z][.)]\s", s):                       # 選項行 "A. ..." / "B) ..."
            break
        if re.match(r"(?i)^answer\b.*\b(option|options|letter|single word|phrase)\b", s):
            break
        out.append(ln)
    q = "\n".join(out).strip()
    return q or text.strip()


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


# ══════════════════════════════════════════════════════════════════════════════
# InternVL 專屬低階函式         （from internvl_core.py）
# ══════════════════════════════════════════════════════════════════════════════
MAX_NEW_TOKENS = 500
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
RANDOM_SEED = 0  # fixed so --score_fn random is reproducible run-to-run

# ── Tier-0 scorer 清理：以下全部是 info_density 的預設行為，沒有 CLI 旗標 ──
#   * novelty（多樣性）項永遠開
#   * pool 每個 tile 到貨都用最新全域統計量重算（原本的 --rescore）永遠開
#   * relevance 只 pool「問句本身」（strip 掉選項與作答指示）而非整段 prompt
SIGNAL_TRIM = 0.02     # D2：signal 項與全域 norm 統計量，永遠 winsorize 掉 norm 最高的 2% token
COVERAGE_FLOOR = 1     # A3：flush 的 top-K 永遠保留 1 個 farthest-point 名額，其餘按分數

# ── HR-Bench 設定：沒有 CLI 旗標，要換 split / prompt 就改這兩個常數 ──
HRBENCH_SPLIT = "hrbench_8k"   # hrbench_4k | hrbench_8k
HRBENCH_PROMPT = "open"      # letter（選項進 prompt、只輸出字母）| open（只給題目、自由生成）

# ── DocVQA 設定：沒有 CLI 旗標，要換 prompt 就改這個常數 ──
DOCVQA_PROMPT = "open"       # short（官方 ANLS 協定、比 evaluate_anls.py）
                              # | open（只給題目、自由生成，比 semantic similarity）

# ── thumbnail-attention tile prior（--thumb_attn 才啟用，預設關；跨方法比較用）──
#   thumbnail 跑一次 LLM forward → 取「答案位置對 256 個 thumbnail patch 的 attention」
#   → 16x16 粗 saliency → 每個 grid tile 一個 prior，再跟 info_density 分數線性混合：
#     final = (1 - THUMB_ATTN_W) * info_density + THUMB_ATTN_W * thumbnail_prior
#   peak memory 不動（thumbnail 本來就是一塊 tile）；TTFT 多一個「與 tile 數無關」的短 prefill。
#   tag 後綴 _thumbattn（權重非預設時再加 _w<x>）。
#   w=0.9 是方法收斂後的預設：HR-Bench b1024/b2048/b4096 掃描下每個 budget 都近最佳
#   （b4096 是唯一顯著贏 Tier-0 且超過 uncompressed 的設定），w=1.0（純 prior）在 K<=8
#   更好但 K>=16 略差；0.9 是單一值的折衷。
THUMB_ATTN_W = 0.9
THUMB_ATTN_LAYERS = 8   # 用最後這幾層 decoder 的 attention 平均


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


def measure_baseline_ttft(model, tokenizer, pixel_values_list, questions, device):
    """跟 generate_answer_standard 走完全相同的 batch_chat() 路徑，
    只把 max_new_tokens 砍到 1。"""
    torch.cuda.synchronize(device)
    t0 = time.time()

    generate_answer_standard(
        model, tokenizer, pixel_values_list, questions,
        generation_config=dict(max_new_tokens=1, do_sample=False),
    )

    torch.cuda.synchronize(device)
    return time.time() - t0


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


# ══════════════════════════════════════════════════════════════════════════════
# Importance scoring + Tile Streaming Memory Bank    （from stream_memory_bank.py）
# ══════════════════════════════════════════════════════════════════════════════
# ──────────────────────────────────────────────────────────────────────────────
# Importance scoring：統一介面 (tokens, question_embed=None, norm_stats=None) -> [N]
#
# norm_stats: 可選的 dict {"mean": scalar tensor, "std": scalar tensor}
#   - 若提供，signal 項會用「跨 tile 的全域統計量」做 z-score 正規化，
#     讓不同 tile 各自算出來的分數在同一個尺度上可比較。
#   - 若不提供（None），fallback 回舊版「單一呼叫內的相對排名」。
# ──────────────────────────────────────────────────────────────────────────────
def _winsorize_high(v: torch.Tensor, frac: float) -> torch.Tensor:
    """把 v 裡超過 (1-frac) 分位數的值壓到該分位數（D2）。

    ViT 已知會在少數背景 patch 產生 norm 異常大的 register/artifact token，
    它們會（a）灌爆 signal 項讓一個 tile 只因為有 artifact 就分高，
    （b）灌爆 GlobalNormStats 的 std。winsorize 掉頭部就能把這兩個效應壓下來。
    """
    if frac is None or frac <= 0.0 or v.numel() < 8:
        return v
    cap = torch.quantile(v.float(), 1.0 - float(frac))
    return v.clamp(max=cap.to(v.dtype))


def _infer_tile_grid(num_grid_tiles: int, orig_w: int, orig_h: int) -> tuple[int, int]:
    """重建 dynamic_preprocess 選的 (cols, rows)。

    dynamic_preprocess 的 best_ratio 一定是 num_grid_tiles 的某組因數分解（blocks =
    cols*rows == 非縮圖 tile 數），且是「aspect ratio 最接近原圖」的那組。這裡直接對
    num_grid_tiles 做因數分解、挑 |W/H - c/r| 最小的 (c, r)，等價於原本的選法。
    """
    n = int(num_grid_tiles)
    if n <= 1:
        return (1, 1)
    ar = (orig_w / orig_h) if orig_h else 1.0
    best, best_d = (n, 1), float("inf")
    for c in range(1, n + 1):
        if n % c:
            continue
        r = n // c
        d = abs(ar - c / r)
        if d < best_d:
            best_d, best = d, (c, r)
    return best


def score_l2_norm(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    # **_ 吞掉 norm_stats 等只有 info_density 用得到的 kwarg，讓 finalize()/merge
    # 的重算路徑對三個 score_fn 都能用同一個呼叫介面（l2_norm / random 忽略之）。
    return tokens.float().norm(dim=-1)


def score_random(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    return torch.rand(tokens.shape[0], device=tokens.device)


def score_information_density(
    tokens, question_embed,
    alpha=0.3, beta=0.3, gamma=0.4,
    knn_k=3,
    norm_stats: dict | None = None,
    signal_trim: float = SIGNAL_TRIM,
    # norm_stats：若提供 {"mean":..., "std":...}（跨「所有已收集 tile」的全域統計量），
    #   signal 改用全域 z-score → sigmoid 正規化；若為 None（例如 add_tile() 內第一顆
    #   tile）fallback 回「單一 tile 內部 max 正規化」。
    # signal_trim：D2，signal 項先 winsorize 掉 norm 最高的這個比例的 token（ViT
    #   register/artifact token norm 異常大）。預設 SIGNAL_TRIM，沒有 CLI 旗標。
):
    """
    Question-aware Information Density Score
    Score = alpha*S + beta*N + gamma*R，三項各自正規化到 [0,1] 後再加權。
    novelty（多樣性）項永遠參與計算，不再有開關。

    tokens:         [N, D]  (已經過 mlp1，跟 LLM embedding space 同維度)
    question_embed: [D]     (只 pool「問句本身」的 mean-pooled LLM embedding)
    """
    x = tokens.float()
    N = x.shape[0]

    # ---- Signal Strength ----
    raw_norm = _winsorize_high(x.norm(dim=-1), signal_trim)   # D2

    if norm_stats is not None:
        z = (raw_norm - norm_stats["mean"]) / norm_stats["std"]
        signal = torch.sigmoid(z)
    elif N > 1:
        signal = raw_norm / (raw_norm.max() + 1e-9)
    else:
        signal = torch.zeros(N, device=x.device)

    # ---- Novelty：top-k 最相似的平均相似度（1-NN 對雜訊太敏感），永遠開 ----
    x_norm = F.normalize(x, dim=-1)
    sim = x_norm @ x_norm.T
    sim.fill_diagonal_(-1.0)
    k = min(knn_k, max(N - 1, 1))
    topk_sim = sim.topk(k, dim=-1).values.mean(dim=-1) if N > 1 else torch.zeros(N, device=x.device)
    novelty = ((1 - topk_sim) / 2).clamp(0.0, 1.0)

    # ---- Question Relevance：centroid cosine，question_embed 已是「只 pool 問句」的 [D] ----
    if question_embed is None or isinstance(question_embed, str):
        relevance = torch.zeros(N, device=x.device)
        gamma_eff = 0.0
    else:
        qn = F.normalize(question_embed.float().reshape(-1), dim=-1)
        relevance = ((x_norm @ qn) + 1) / 2                 # [N]
        gamma_eff = gamma

    total = alpha + beta + gamma_eff
    a, b, g = alpha / total, beta / total, gamma_eff / total
    score = a * signal + b * novelty + g * relevance
    return score


SCORE_FUNCS = {
    "l2_norm": score_l2_norm,
    "info_density": score_information_density,
    "random": score_random,
}


class GlobalNormStats:
    """Running Welford mean/std of the per-token L2 norm, shared by both memory
    bank implementations so a tile's ``signal`` score stays comparable across
    tiles even though each tile is scored independently as it streams in."""

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    @torch.no_grad()
    def update(self, tile_tokens):
        v = _winsorize_high(tile_tokens.float().norm(dim=-1), SIGNAL_TRIM)   # D2
        n_b = v.numel()
        if n_b == 0:
            return
        mean_b = float(v.mean().item())
        M2_b = float(((v - v.mean()) ** 2).sum().item())
        if self._n == 0:
            self._n, self._mean, self._M2 = n_b, mean_b, M2_b
            return
        n_a, mean_a, M2_a = self._n, self._mean, self._M2
        n = n_a + n_b
        delta = mean_b - mean_a
        self._mean = mean_a + delta * n_b / n
        self._M2 = M2_a + M2_b + delta * delta * n_a * n_b / n
        self._n = n

    def as_dict(self, device=None):
        if self._n < 2:
            return None
        std = (self._M2 / self._n) ** 0.5
        return {
            "mean": torch.tensor(self._mean, device=device),
            "std": torch.tensor(max(std, 1e-6), device=device),
        }


@torch.no_grad()
def score_tile_tokens(tile_tokens, score_fn_name, question_embed, tile_agg, norm_stats):
    """Score a tile's tokens with ``score_fn_name`` and aggregate to one scalar.
    Shared by TileStreamingMemoryBank and SpatialBinTileMemoryBank."""
    if score_fn_name == "l2_norm":
        scores = score_l2_norm(tile_tokens, question_embed)
    elif score_fn_name == "random":
        scores = score_random(tile_tokens, question_embed)
    elif score_fn_name == "info_density":
        scores = score_information_density(
            tile_tokens, question_embed,
            alpha=0.3, beta=0.3, gamma=0.4,
            norm_stats=norm_stats,
        )
    else:
        raise ValueError(f"Unknown score_fn={score_fn_name}")
    scores = scores.float().reshape(-1)
    if scores.numel() == 0:
        return 0.0
    if tile_agg == "mean":
        return float(scores.mean().item())
    if tile_agg == "max":
        return float(scores.max().item())
    if tile_agg == "topk_mean":
        k = max(1, int(round(0.1 * scores.numel())))
        return float(scores.topk(k).values.mean().item())
    if tile_agg == "mean_std":
        return float((scores.mean() + scores.std(unbiased=False)).item())
    if tile_agg == "quantile":
        return float(scores.quantile(0.9).item())
    raise ValueError(f"Unknown tile_agg={tile_agg}")


def score_spread_summary(all_tile_scores):
    """min/max/mean/std of every per-tile aggregate score seen for one image.

    Used to sanity-check whether a score_fn actually discriminates between
    tiles: if std is tiny relative to the mean, ranking by this score is close
    to arbitrary and a scorer "losing" to random tile selection is expected,
    not surprising.
    """
    if not all_tile_scores:
        return {"score_min": None, "score_max": None, "score_mean": None, "score_std": None}
    t = torch.tensor(all_tile_scores, dtype=torch.float32)
    return {
        "score_min": float(t.min().item()),
        "score_max": float(t.max().item()),
        "score_mean": float(t.mean().item()),
        "score_std": float(t.std(unbiased=False).item()),
    }


def _contiguous_bins(indices, n_bins):
    """Split a sorted list of indices into n_bins contiguous, near-equal chunks."""
    n = len(indices)
    n_bins = min(n_bins, n)
    if n_bins <= 0:
        return []
    base, extra = divmod(n, n_bins)
    bins, pos = [], 0
    for b in range(n_bins):
        size = base + (1 if b < extra else 0)
        bins.append(indices[pos:pos + size])
        pos += size
    return bins


class TileEntry:
    """A complete InternVL visual tile. Selection is tile-level only."""
    __slots__ = ("tokens", "tile_index", "score", "arrival")
    def __init__(self, tokens, tile_index, score, arrival):
        self.tokens = tokens
        self.tile_index = tile_index
        self.score = float(score)
        self.arrival = int(arrival)


class TileStreamingMemoryBank:
    """Bounded candidate pool + global top-K tile selection (lazy commit).

    Two decisions are deliberately separated:

      * eviction  - runs on *every* incoming tile. The pool is pruned back to
                    K + D entries by dropping the lowest-scoring non-protected
                    tile (which may be the tile that just arrived). The pool is
                    therefore bounded for the whole stream.
      * commit    - happens only in ``flush()``, once the whole image has been
                    seen. ``flush()`` picks the final top-K among the survivors.
                    Nothing is written to the LLM KV cache before that, so an
                    early low-value tile can never be baked in irreversibly.

    K = budget // num_image_token   hard cap on committed tiles.
    D = delay_tiles                 extra candidate slots. Because the pool is
                                    distilled continuously, D=2 already tracks
                                    the global top-K closely; a larger D only
                                    buys robustness to score noise at the cost
                                    of D * num_image_token * dim memory.

    ``score_fn='info_density'`` signal term is z-scored against a running global
    mean/std of the per-token L2 norm (Welford), so tile scores are comparable
    across tiles even though each tile is scored independently on arrival.
    """

    def __init__(self, capacity, num_image_token=256, device=None, dtype=None,
                 score_fn="info_density", question_embed=None, tile_agg="max",
                 delay_tiles=2, protected_indices=(), tile_prior=None, thumb_attn_w=0.0):
        if capacity < num_image_token:
            raise ValueError(f"budget={capacity} < num_image_token={num_image_token}")
        if delay_tiles < 0:
            raise ValueError("delay_tiles must be >= 0")
        self.num_image_token = num_image_token
        self.K = capacity // num_image_token
        self.D = int(delay_tiles)
        self.pool_capacity = self.K + self.D
        self.device, self.dtype = device, dtype
        self.score_fn_name = score_fn
        self.question_embed = question_embed
        self.tile_agg = tile_agg
        # pool 每個 tile 到貨都用最新全域統計量重算（原本的 --rescore），永遠開。
        self.rescore = True
        # thumbnail-attention prior：{tile_index -> prior in [0,1]}；None = 不啟用（線性混合權重見 thumb_attn_w）。
        self.tile_prior = tile_prior
        self.thumb_attn_w = float(thumb_attn_w) if tile_prior is not None else 0.0
        self.protected_indices = set(int(i) for i in protected_indices)
        self.pool = []
        self._norm = GlobalNormStats()
        self.total_seen = 0
        self.total_evicted = 0
        self.total_committed = 0
        self.committed_indices = []
        self.all_scores = []  # every tile's aggregate score, incl. later-evicted ones
        self.max_resident_tiles = 0
        self.size_history = []

    @torch.no_grad()
    def _score_tile(self, tile_tokens):
        return score_tile_tokens(
            tile_tokens, self.score_fn_name, self.question_embed,
            self.tile_agg, self._norm.as_dict(self.device),
        )

    def _blend_prior(self, base_score, tile_index):
        """thumbnail-attention prior：final = (1-w)*base + w*thumbnail_prior。w=0 或沒有 prior 時原樣回傳。"""
        if self.tile_prior is None or self.thumb_attn_w <= 0.0:
            return base_score
        p = float(self.tile_prior.get(int(tile_index), 0.0))
        return (1.0 - self.thumb_attn_w) * float(base_score) + self.thumb_attn_w * p

    def _protected(self, tile_index):
        return tile_index in self.protected_indices

    @torch.no_grad()
    def _rescore_pool(self):
        for e in self.pool:
            e.score = self._blend_prior(self._score_tile(e.tokens), e.tile_index)

    @torch.no_grad()
    def _evict_to_capacity(self):
        overflow = len(self.pool) - self.pool_capacity
        if overflow <= 0:
            return
        cand = [e for e in self.pool if not self._protected(e.tile_index)]
        # lowest score first; tie -> drop the spatially-later tile so early
        # spatial coverage is preserved.
        cand.sort(key=lambda e: (e.score, -e.tile_index))
        for e in cand[:overflow]:
            self.pool.remove(e)
            self.total_evicted += 1
            del e.tokens

    @torch.no_grad()
    def add_tile(self, tile_tokens, tile_index):
        if tile_tokens.ndim != 2 or tile_tokens.shape[0] != self.num_image_token:
            raise ValueError(
                f"Expected [{self.num_image_token}, D], got {tuple(tile_tokens.shape)}"
            )
        self._norm.update(tile_tokens)
        score = self._blend_prior(self._score_tile(tile_tokens), tile_index)
        self.all_scores.append(score)
        self.pool.append(
            TileEntry(tile_tokens.detach().clone(), tile_index, score, self.total_seen)
        )
        self.total_seen += 1
        if self.rescore:
            self._rescore_pool()
        self._evict_to_capacity()
        self.max_resident_tiles = max(self.max_resident_tiles, len(self.pool))
        self.size_history.append(len(self.pool) * self.num_image_token)
        return []  # lazy commit: nothing enters the KV cache mid-stream

    @torch.no_grad()
    def flush(self):
        """End of image: pick the final top-K, returned in spatial order."""
        if self.rescore:
            self._rescore_pool()
        protected = sorted(
            (e for e in self.pool if self._protected(e.tile_index)),
            key=lambda e: e.tile_index,
        )
        rest = [e for e in self.pool if not self._protected(e.tile_index)]
        kept = protected[: self.K]
        keep_n = self.K - len(kept)
        if keep_n > 0:
            rest.sort(key=lambda e: e.score, reverse=True)
            cf = min(COVERAGE_FLOOR, keep_n)
            kept = kept + rest[: keep_n - cf]            # A3：keep_n - cf 個純看分數
            if cf > 0:
                # 剩下 cf 個名額給 farthest-point：每次挑「離已選 tile-index 最遠」的候選，
                # 平手時 pending 仍是分數排序 -> 取分數高的。tile-index 是 raster order，
                # 只是 2D 距離的粗代理，但夠便宜、也跟 spatial_bin 的分箱一致。
                pool_idx = {e.tile_index for e in kept}
                pending = rest[keep_n - cf:]
                for _ in range(cf):
                    if not pending:
                        break
                    pick = max(
                        pending,
                        key=lambda e: min((abs(e.tile_index - j) for j in pool_idx), default=0),
                    )
                    kept.append(pick)
                    pending.remove(pick)
                    pool_idx.add(pick.tile_index)
        kept.sort(key=lambda e: e.tile_index)
        self.total_committed = len(kept)
        self.committed_indices = [e.tile_index for e in kept]
        kept_ids = {id(e) for e in kept}
        for e in self.pool:
            if id(e) not in kept_ids:
                del e.tokens
        out = [e.tokens for e in kept]
        self.pool = []
        return out

    def stats(self):
        dropped = max(0, self.total_seen - self.total_committed)
        out = {
            "total_seen_tiles": self.total_seen,
            "total_seen_tokens": self.total_seen * self.num_image_token,
            "final_tiles": self.total_committed,
            "final_size": self.total_committed * self.num_image_token,
            "capacity_tiles": self.K,
            "capacity_tokens": self.K * self.num_image_token,
            "delay_tiles": self.D,
            "pool_capacity_tiles": self.pool_capacity,
            "total_evicted_tiles": self.total_evicted,
            "total_dropped_tiles": dropped,
            "total_committed_tiles": self.total_committed,
            "committed_tile_indices": self.committed_indices,
            "rescored": self.rescore,
            "max_resident_tiles": self.max_resident_tiles,
            "compression_ratio": self.total_seen / max(self.total_committed, 1),
            "size_history": self.size_history,
        }
        out.update(score_spread_summary(self.all_scores))
        return out


class SpatialBinTileMemoryBank:
    """Coverage-guaranteed tile selection: partitions the tile grid into K
    contiguous spatial bins (in raster/tile-index order, matching how
    dynamic_preprocess lays tiles out) and keeps at most one winner per bin.
    The final selection therefore always spans the whole image, regardless of
    how the score distribution happens to cluster.

    This trades score-optimality for a coverage guarantee: TileStreamingMemoryBank's
    unconstrained top-K can end up choosing several tiles from the same crowded
    region and none from another, which is fatal on documents where the answer
    is confined to one small area. Here a tile can only be evicted by a
    higher-scoring tile from its OWN bin, never by an unrelated tile elsewhere.

    Because bin membership only depends on the image's tile count (known before
    any vision feature is computed) and bins never compete with each other for a
    slot, a bin's winner is final as soon as every tile belonging to that bin has
    streamed by -- it could be committed to the LLM KV cache immediately, unlike
    the delayed global top-K in TileStreamingMemoryBank.
    """

    def __init__(self, capacity, num_image_token=256, device=None, dtype=None,
                 score_fn="info_density", question_embed=None, tile_agg="max",
                 protected_indices=(), num_tiles=1, tile_prior=None, thumb_attn_w=0.0):
        if capacity < num_image_token:
            raise ValueError(f"budget={capacity} < num_image_token={num_image_token}")
        self.num_image_token = num_image_token
        self.K = capacity // num_image_token
        self.D = 0  # no delay window: a bin's winner is decided, not deferred
        self.device, self.dtype = device, dtype
        self.score_fn_name = score_fn
        self.question_embed = question_embed
        self.tile_agg = tile_agg
        self.rescore = True   # champion 每次都用最新全域統計量重算，永遠開
        self.tile_prior = tile_prior          # thumbnail-attention prior，見 TileStreamingMemoryBank._blend_prior
        self.thumb_attn_w = float(thumb_attn_w) if tile_prior is not None else 0.0
        self.protected_indices = set(int(i) for i in protected_indices)

        grid_indices = [i for i in range(num_tiles) if i not in self.protected_indices]
        n_bins = max(0, self.K - len(self.protected_indices))
        self.bins = _contiguous_bins(grid_indices, n_bins)
        self.bin_of = {idx: b for b, members in enumerate(self.bins) for idx in members}

        self.champion = [None] * len(self.bins)
        self.protected_entries = {}
        self.pool_capacity = self.K
        self._norm = GlobalNormStats()
        self.total_seen = 0
        self.total_evicted = 0
        self.total_committed = 0
        self.committed_indices = []
        self.all_scores = []  # every tile's aggregate score, incl. later-evicted ones
        self.max_resident_tiles = 0
        self.size_history = []

    def _protected(self, tile_index):
        return tile_index in self.protected_indices

    def _blend_prior(self, base_score, tile_index):
        if self.tile_prior is None or self.thumb_attn_w <= 0.0:
            return base_score
        p = float(self.tile_prior.get(int(tile_index), 0.0))
        return (1.0 - self.thumb_attn_w) * float(base_score) + self.thumb_attn_w * p

    @property
    def pool(self):
        """Currently resident entries (logging only; mirrors TileStreamingMemoryBank.pool)."""
        return list(self.protected_entries.values()) + [c for c in self.champion if c is not None]

    @torch.no_grad()
    def add_tile(self, tile_tokens, tile_index):
        if tile_tokens.ndim != 2 or tile_tokens.shape[0] != self.num_image_token:
            raise ValueError(
                f"Expected [{self.num_image_token}, D], got {tuple(tile_tokens.shape)}"
            )
        self._norm.update(tile_tokens)
        score = self._blend_prior(score_tile_tokens(
            tile_tokens, self.score_fn_name, self.question_embed,
            self.tile_agg, self._norm.as_dict(self.device),
        ), tile_index)
        self.all_scores.append(score)
        entry = TileEntry(tile_tokens.detach().clone(), tile_index, score, self.total_seen)
        self.total_seen += 1

        if self._protected(tile_index):
            self.protected_entries[tile_index] = entry
        else:
            b = self.bin_of.get(tile_index)
            current = self.champion[b] if b is not None else None
            if current is not None:
                # keep the champion's score fresh against the latest global
                # norm stats before comparing it to the new challenger.
                current.score = self._blend_prior(score_tile_tokens(
                    current.tokens, self.score_fn_name, self.question_embed,
                    self.tile_agg, self._norm.as_dict(self.device),
                ), current.tile_index)
            if b is None:
                # K already fully spent on protected tiles -> no bin left for this one
                del entry.tokens
                self.total_evicted += 1
            elif current is None or entry.score > current.score:
                if current is not None:
                    del current.tokens
                    self.total_evicted += 1
                self.champion[b] = entry
            else:
                del entry.tokens
                self.total_evicted += 1

        resident = len(self.protected_entries) + sum(1 for c in self.champion if c is not None)
        self.max_resident_tiles = max(self.max_resident_tiles, resident)
        self.size_history.append(resident * self.num_image_token)
        return []  # lazy commit, same contract as TileStreamingMemoryBank

    @torch.no_grad()
    def flush(self):
        """End of image: one winner per bin + all protected tiles, in spatial order."""
        kept = list(self.protected_entries.values()) + [c for c in self.champion if c is not None]
        kept.sort(key=lambda e: e.tile_index)
        self.total_committed = len(kept)
        self.committed_indices = [e.tile_index for e in kept]
        out = [e.tokens for e in kept]
        self.protected_entries = {}
        self.champion = [None] * len(self.bins)
        return out

    def stats(self):
        dropped = max(0, self.total_seen - self.total_committed)
        out = {
            "total_seen_tiles": self.total_seen,
            "total_seen_tokens": self.total_seen * self.num_image_token,
            "final_tiles": self.total_committed,
            "final_size": self.total_committed * self.num_image_token,
            "capacity_tiles": self.K,
            "capacity_tokens": self.K * self.num_image_token,
            "delay_tiles": self.D,
            "pool_capacity_tiles": self.pool_capacity,
            "num_bins": len(self.bins),
            "total_evicted_tiles": self.total_evicted,
            "total_dropped_tiles": dropped,
            "total_committed_tiles": self.total_committed,
            "committed_tile_indices": self.committed_indices,
            "rescored": self.rescore,
            "max_resident_tiles": self.max_resident_tiles,
            "compression_ratio": self.total_seen / max(self.total_committed, 1),
            "size_history": self.size_history,
        }
        out.update(score_spread_summary(self.all_scores))
        return out


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
    def thumbnail_saliency_prior(self, model, pixel_values, text_before, text_after,
                                 cols, rows, dtype):
        """thumbnail 跑一次 LLM forward，取「答案位置對 256 個 thumbnail patch 的
        attention」→ 16x16 粗 saliency → 每個 grid tile 一個 prior。

        回傳 {tile_index -> prior in [0,1]}（thumbnail 自己那格設 1.0，反正是 protected）。
        拿不到 attention（backend 不吐）時回傳 None，呼叫端當 no-op（純 info_density）。

        成本：一次 forward，長度 ≈ len(text_before) + 256 + len(text_after)，跟總 tile
        數無關；不新增常駐張量，peak memory 不動。
        """
        num_tiles = pixel_values.shape[0]
        grid_tiles = cols * rows
        # 1) 只 encode thumbnail（dynamic_preprocess 的最後一塊）
        thumb_px = pixel_values[num_tiles - 1: num_tiles].to(DEVICE, dtype=dtype)
        thumb_tok = encode_tile(model, thumb_px, dtype=dtype)[0]          # [256, D]
        n_patch = thumb_tok.shape[0]                                      # = num_image_token = 256
        side = int(round(n_patch ** 0.5))                                 # 16

        # 2) text_before + thumbnail + text_after 的 inputs_embeds
        emb_layer = self.get_input_embeddings(model)
        tb = self.tokenizer([text_before], return_tensors="pt").to(DEVICE)
        ta = self.tokenizer([text_after], return_tensors="pt").to(DEVICE)
        emb_b = emb_layer(tb.input_ids)
        emb_a = emb_layer(ta.input_ids)
        inp = torch.cat([emb_b, thumb_tok.unsqueeze(0), emb_a], dim=1)
        att = torch.ones((1, inp.shape[1]), dtype=torch.long, device=DEVICE)
        thumb_start = emb_b.shape[1]

        # flash_attention_2 不吐 attention weights。只在這一次 forward 暫時切成 eager，
        # 之後還原 —— 主 streaming 路徑仍走 flash，TTFT / peak memory 的量測不受影響。
        lm = model.language_model
        base = getattr(lm, "model", None) or lm
        prev_impl = getattr(lm.config, "_attn_implementation", None)
        eager_impls = ("eager", "eager_paged", "flex_attention")
        switched = False
        if prev_impl not in eager_impls and hasattr(lm, "set_attn_implementation"):
            try:
                lm.set_attn_implementation("eager")
                switched = True
            except Exception as e:                                    # noqa: BLE001
                print(f"[thumb-attn] set_attn_implementation('eager') 失敗：{e}")
        try:
            out = base(inputs_embeds=inp, attention_mask=att, use_cache=False,
                       output_attentions=True, return_dict=True)
        finally:
            if switched:
                try:
                    lm.set_attn_implementation(prev_impl)
                except Exception as e:                                # noqa: BLE001
                    print(f"[thumb-attn] 還原 attn_implementation={prev_impl!r} 失敗：{e}")

        attns = getattr(out, "attentions", None)
        if not attns or attns[0] is None:
            print("[thumb-attn] 仍拿不到 attention，退化成 no-op（純 info_density）；"
                  "可改在 build_model 用 attn_implementation='eager' 或改寫成 attention hook")
            return None

        # 3) 最後 THUMB_ATTN_LAYERS 層、答案位置(-1) 對 thumbnail key span 的 attention，平均 head/layer
        layers = attns[-THUMB_ATTN_LAYERS:] if len(attns) >= THUMB_ATTN_LAYERS else attns
        sal = torch.stack([
            a[0, :, -1, thumb_start: thumb_start + n_patch].float().mean(dim=0)  # [n_patch]
            for a in layers
        ], dim=0).mean(dim=0)                                                    # [n_patch]
        sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
        sal = sal.reshape(side, side).cpu()                                      # [16, 16] row-major

        # 4) 每個 thumbnail patch 的中心落在哪個 grid tile → 累積該 tile 的 saliency
        acc = [0.0] * grid_tiles
        cnt = [0] * grid_tiles
        for pr in range(side):
            for pc in range(side):
                gc = min(cols - 1, int((pc + 0.5) / side * cols))
                gr = min(rows - 1, int((pr + 0.5) / side * rows))
                k = gr * cols + gc
                acc[k] += float(sal[pr, pc])
                cnt[k] += 1
        prior = [acc[k] / cnt[k] if cnt[k] else 0.0 for k in range(grid_tiles)]
        lo, hi = min(prior), max(prior)
        prior = [(p - lo) / (hi - lo + 1e-6) for p in prior]                    # 再正規化到 [0,1]

        d = {k: prior[k] for k in range(grid_tiles)}
        d[num_tiles - 1] = 1.0                                                  # thumbnail 本體
        return d

    def encode_and_bank(
        self,
        model,
        pixel_values_list,
        question_embeds,
        budget,
        score_fn,
        vit_batch,
        dtype,
        tile_agg="max",
        select="topk",
        delay_tiles=2,
    ):
        """V2 tile streaming WITHOUT global tile materialization.

        This function is intentionally selection-only. For true online KV,
        use ``stream_image_to_kv`` below, which commits selected tiles directly
        into the language-model KV cache. (legacy 診斷路徑，主流程未呼叫。)
        """
        if select not in ("topk", "spatial_bin"):
            raise ValueError(f"Unknown select={select!r}; expected 'topk' or 'spatial_bin'")

        B = len(pixel_values_list)
        num_image_token = model.num_image_token
        D_llm = model.mlp1[-1].out_features
        finalized, all_stats = [], []

        for b in range(B):
            pv = pixel_values_list[b]
            num_tiles = pv.shape[0]
            protected = {num_tiles - 1}  # whole-image thumbnail (last tile)
            if select == "topk":
                memory = TileStreamingMemoryBank(
                    capacity=budget,
                    num_image_token=num_image_token,
                    device=DEVICE,
                    dtype=dtype,
                    score_fn=score_fn,
                    question_embed=question_embeds[b],
                    tile_agg=tile_agg,
                    delay_tiles=delay_tiles,
                    protected_indices=protected,
                )
            else:
                memory = SpatialBinTileMemoryBank(
                    capacity=budget,
                    num_image_token=num_image_token,
                    device=DEVICE,
                    dtype=dtype,
                    score_fn=score_fn,
                    question_embed=question_embeds[b],
                    tile_agg=tile_agg,
                    protected_indices=protected,
                    num_tiles=num_tiles,
                )

            print(
                f"\n[Image {b}] raw_tiles={num_tiles} "
                f"raw_tokens={num_tiles * num_image_token} "
                f"budget={budget} (K={memory.K} + D={memory.D} pool tiles) "
                f"score={score_fn}"
            )

            # IMPORTANT: no torch.cat over all image tiles.
            for start_i in range(0, num_tiles, vit_batch):
                end_i = min(start_i + vit_batch, num_tiles)
                chunk = pv[start_i:end_i].to(DEVICE, dtype=dtype)

                tile_tokens = encode_tile(model, chunk, dtype=dtype)
                del chunk

                for j in range(tile_tokens.shape[0]):
                    memory.add_tile(
                        tile_tokens[j],
                        tile_index=start_i + j,
                    )

                del tile_tokens

                if torch.cuda.is_available():
                    mem = torch.cuda.memory_allocated(DEVICE) / 1e9
                    print(
                        f"  ├─> [V2 ViT+TileMemory] "
                        f"tiles {start_i}:{end_i}/{num_tiles} "
                        f"pool={len(memory.pool)}/{memory.pool_capacity} "
                        f"evicted={memory.total_evicted} "
                        f"alloc={mem:.2f} GB"
                    )

            # Selection-only fallback: flush selected tiles at the end.
            # This path is useful for debugging/ablation, while the main V2
            # execution path below performs incremental KV construction.
            committed = memory.flush()
            if committed:
                toks = torch.cat(committed, dim=0)
            else:
                toks = torch.empty(
                    0, D_llm, device=DEVICE, dtype=dtype
                )

            stat = memory.stats()
            stat["selection_only"] = True
            stat["final_size"] = toks.shape[0]
            stat["final_tiles"] = toks.shape[0] // num_image_token

            assert toks.shape[0] <= budget
            finalized.append(toks)
            all_stats.append(stat)

            print(
                f"  image {b}: raw={stat['total_seen_tiles']} tiles -> "
                f"final={stat['final_tiles']} tiles "
                f"({stat['final_size']} tokens), "
                f"evicted={stat['total_evicted_tiles']}, "
                f"dropped={stat['total_dropped_tiles']}, "
                f"max_resident={stat['max_resident_tiles']} tiles"
            )

        return finalized, all_stats

    @torch.no_grad()
    def stream_image_to_kv(
        self,
        model,
        pixel_values,
        question_embed,
        budget,
        score_fn="info_density",
        vit_batch=4,
        chunk_size=1024,
        dtype=torch.bfloat16,
        delay_tiles=2,
        tile_agg="max",
        select="topk",
        tile_prior=None,
        thumb_attn_w=0.0,
        past_key_values=None,
        running_mask=None,
    ):
        """True online Vision -> bounded candidate pool -> LLM KV construction.

        Phase 1 streams every tile through a bounded pool of at most
        ``K + delay_tiles`` entries (``K = budget // num_image_token``). Eviction
        runs on every tile, but no tile is committed yet.

        Phase 2 runs once the whole image has been seen: ``memory.flush()``
        selects the final top-K survivors and they are prefilled into the LLM KV
        cache in spatial (tile-index) order, in ``chunk_size`` slices. A tile is
        only ever written to KV after it has survived the full competition, so
        an early low-value tile is never baked in irreversibly.

        ``select='topk'`` uses TileStreamingMemoryBank (unconstrained global
        top-K). ``select='spatial_bin'`` uses SpatialBinTileMemoryBank, which
        guarantees the selected tiles span the whole image (one winner per
        spatial bin) at the cost of not always taking the globally highest
        scores.
        """
        num_image_token = model.num_image_token
        num_tiles = pixel_values.shape[0]

        if past_key_values is None:
            raise ValueError("past_key_values must be initialized by text_before prefill")
        if running_mask is None:
            raise ValueError("running_mask must be initialized by text_before prefill")
        if select not in ("topk", "spatial_bin"):
            raise ValueError(f"Unknown select={select!r}; expected 'topk' or 'spatial_bin'")

        # dynamic_preprocess appends the whole-image thumbnail as the LAST tile;
        # it carries global context so it always survives selection. (num_tiles
        # == 1 means no thumbnail was added and this harmlessly points at the
        # sole tile.)
        protected = {num_tiles - 1}
        if select == "topk":
            memory = TileStreamingMemoryBank(
                capacity=budget,
                num_image_token=num_image_token,
                device=DEVICE,
                dtype=dtype,
                score_fn=score_fn,
                question_embed=question_embed,
                tile_agg=tile_agg,
                delay_tiles=delay_tiles,
                protected_indices=protected,
                tile_prior=tile_prior,
                thumb_attn_w=thumb_attn_w,
            )
        else:
            memory = SpatialBinTileMemoryBank(
                capacity=budget,
                num_image_token=num_image_token,
                device=DEVICE,
                dtype=dtype,
                score_fn=score_fn,
                question_embed=question_embed,
                tile_agg=tile_agg,
                protected_indices=protected,
                num_tiles=num_tiles,
                tile_prior=tile_prior,
                thumb_attn_w=thumb_attn_w,
            )

        tokens_injected = 0
        kv_flushes = 0

        def prefill_vision(vtok_2d):
            """Write one [c, D] block of committed visual tokens into the KV cache."""
            nonlocal past_key_values, running_mask, tokens_injected, kv_flushes
            vchunk = vtok_2d.unsqueeze(0)  # [1, c, D]
            c = vchunk.shape[1]
            past_seq_len = past_key_values.get_seq_length()
            mchunk = torch.ones((1, c), dtype=torch.long, device=DEVICE)
            running_mask = torch.cat([running_mask, mchunk], dim=1)
            position_ids = torch.arange(
                past_seq_len, past_seq_len + c, dtype=torch.long, device=DEVICE,
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

        # ---- Phase 1: stream every tile through the bounded pool (no KV writes) ----
        for start_i in range(0, num_tiles, vit_batch):
            end_i = min(start_i + vit_batch, num_tiles)
            chunk = pixel_values[start_i:end_i].to(DEVICE, dtype=dtype)

            tile_tokens = encode_tile(model, chunk, dtype=dtype)
            del chunk

            for j in range(tile_tokens.shape[0]):
                memory.add_tile(tile_tokens[j], tile_index=start_i + j)

            del tile_tokens

            if torch.cuda.is_available():
                print(
                    f"  ├─> [stream] tiles {start_i}:{end_i}/{num_tiles} "
                    f"pool={len(memory.pool)}/{memory.pool_capacity} "
                    f"evicted={memory.total_evicted} "
                    f"alloc={torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB"
                )

        # ---- Phase 2: final top-K selection, then chunked KV prefill ----
        selected = memory.flush()  # list of [num_image_token, D], sorted by tile_index
        tiles_per_flush = max(1, chunk_size // num_image_token)
        for s in range(0, len(selected), tiles_per_flush):
            group = selected[s:s + tiles_per_flush]
            prefill_vision(torch.cat(group, dim=0))
        del selected

        stats = memory.stats()
        stats.update({
            "selection_only": False,
            "num_tiles": num_tiles,
            "tokens_injected": tokens_injected,
            "kv_flushes": kv_flushes,
            "final_size": tokens_injected,
            "final_tiles": tokens_injected // num_image_token,
        })

        assert tokens_injected <= budget, (
            f"KV visual tokens {tokens_injected} > budget {budget}"
        )
        return past_key_values, running_mask, stats

    @torch.no_grad()
    def stream_image_to_kv_oracle(
        self,
        model,
        pixel_values,
        text_before,
        text_after,
        gold_letter,
        budget,
        vit_batch=4,
        chunk_size=1024,
        dtype=torch.bfloat16,
        oracle_mode="per_tile",
        past_key_values=None,
        running_mask=None,
    ):
        """DIAGNOSTIC answer-aware tile selection -- it peeks at ``gold_letter``.

        Ranks tiles by how much each one raises P(gold answer letter) when it is
        appended, on top of the always-kept whole-image thumbnail, to the prompt,
        then commits the top-K into the KV cache in spatial order. This is NOT a
        deployable selector; it is the ceiling for "if tile selection were
        perfect", to be read against fixed_tiles / uncompressed and against the
        info_density online runs (if this does not clearly beat fixed_tiles, no
        scorer can, and the selection framework has no headroom on this dataset).

          oracle_mode='per_tile' : score every tile once with {thumbnail, tile};
                                   take the top (K-1). O(N) LM forwards.
          oracle_mode='greedy'   : forward-greedy -- at each step add the tile
                                   that most raises P(gold) given the tiles
                                   chosen so far. O(N*K) forwards, handles
                                   spatial redundancy between winners.

        The KV-commit tail is identical to ``stream_image_to_kv`` (chunked
        prefill of the selected tiles, spatial order), so downstream decode is
        unchanged.
        """
        if past_key_values is None or running_mask is None:
            raise ValueError("oracle path expects text_before to be prefilled already")
        if oracle_mode not in ("per_tile", "greedy"):
            raise ValueError(f"Unknown oracle_mode={oracle_mode!r}")

        num_image_token = model.num_image_token
        num_tiles = pixel_values.shape[0]
        K = max(1, budget // num_image_token)
        protected = num_tiles - 1  # whole-image thumbnail (last tile), always kept

        # ---- encode every tile once; keep all blocks (num_tiles is small, ~40) ----
        tile_blocks = [None] * num_tiles
        for s in range(0, num_tiles, vit_batch):
            e = min(s + vit_batch, num_tiles)
            chunk = pixel_values[s:e].to(DEVICE, dtype=dtype)
            tt = encode_tile(model, chunk, dtype=dtype)
            for j in range(tt.shape[0]):
                tile_blocks[s + j] = tt[j].detach().clone()
            del chunk, tt

        # ---- text embeds + gold-letter token id(s), computed once ----
        emb_layer = self.get_input_embeddings(model)
        tb = self.tokenizer([text_before], return_tensors="pt", padding=True, padding_side="right")
        ta = self.tokenizer([text_after], return_tensors="pt", padding=True, padding_side="right")
        emb_before = emb_layer(tb.input_ids.to(DEVICE))
        emb_after = emb_layer(ta.input_ids.to(DEVICE))
        m_before = tb.attention_mask.to(DEVICE)
        m_after = ta.attention_mask.to(DEVICE)

        gold_ids = []
        for form in (gold_letter, " " + gold_letter):
            ids = self.tokenizer(form, add_special_tokens=False).input_ids
            if ids:
                gold_ids.append(int(ids[0]))
        gold_ids = list(dict.fromkeys(gold_ids))
        if not gold_ids:
            raise ValueError(f"could not tokenize gold_letter={gold_letter!r}")

        def gold_logprob(idx_list):
            vis = torch.cat([tile_blocks[t].unsqueeze(0) for t in idx_list], dim=1)  # [1, c, D]
            inp = torch.cat([emb_before, vis, emb_after], dim=1)
            att = torch.cat(
                [m_before,
                 torch.ones((1, vis.shape[1]), dtype=m_before.dtype, device=DEVICE),
                 m_after],
                dim=1,
            )
            out = model.language_model(
                inputs_embeds=inp, attention_mask=att, use_cache=False, return_dict=True
            )
            lp = torch.log_softmax(out.logits[0, -1, :].float(), dim=-1)
            best = max(float(lp[g].item()) for g in gold_ids)
            del out, inp, att, vis, lp
            return best

        cand = [t for t in range(num_tiles) if t != protected]
        per_tile_lp = {}
        if oracle_mode == "greedy":
            chosen = [protected]
            while len(chosen) < K and cand:
                best_t, best_lp = None, -1e30
                for t in cand:
                    lp = gold_logprob(sorted(chosen + [t]))
                    per_tile_lp.setdefault(t, lp)  # first round == per-tile score
                    if lp > best_lp:
                        best_lp, best_t = lp, t
                chosen.append(best_t)
                cand.remove(best_t)
        else:  # per_tile
            for t in cand:
                per_tile_lp[t] = gold_logprob([protected, t])
            ranked = sorted(cand, key=lambda t: per_tile_lp[t], reverse=True)
            chosen = [protected] + ranked[: max(0, K - 1)]

        chosen = sorted(set(chosen))[:K]

        # ---- commit chosen tiles into the KV cache, spatial order, chunked ----
        tokens_injected, kv_flushes = 0, 0
        tiles_per_flush = max(1, chunk_size // num_image_token)
        for s in range(0, len(chosen), tiles_per_flush):
            grp = [tile_blocks[t] for t in chosen[s:s + tiles_per_flush]]
            vchunk = torch.cat(grp, dim=0).unsqueeze(0)  # [1, c, D]
            c = vchunk.shape[1]
            past_seq_len = past_key_values.get_seq_length()
            running_mask = torch.cat(
                [running_mask, torch.ones((1, c), dtype=torch.long, device=DEVICE)], dim=1
            )
            position_ids = torch.arange(
                past_seq_len, past_seq_len + c, dtype=torch.long, device=DEVICE
            ).unsqueeze(0)
            out = self.lm_prefill(
                model, inputs_embeds=vchunk, attention_mask=running_mask,
                position_ids=position_ids, past_key_values=past_key_values,
                use_cache=True, return_dict=True,
            )
            past_key_values = out.past_key_values
            tokens_injected += c
            kv_flushes += 1
            del out, vchunk, position_ids

        del tile_blocks

        lps = list(per_tile_lp.values())
        lps_t = torch.tensor(lps, dtype=torch.float32) if lps else torch.zeros(0)
        stats = {
            "selection_only": False,
            "oracle_mode": oracle_mode,
            "num_tiles": num_tiles,
            "total_seen_tiles": num_tiles,
            "total_seen_tokens": num_tiles * num_image_token,
            "final_tiles": tokens_injected // num_image_token,
            "final_size": tokens_injected,
            "tokens_injected": tokens_injected,
            "kv_flushes": kv_flushes,
            "capacity_tiles": K,
            "capacity_tokens": K * num_image_token,
            "delay_tiles": 0,
            "pool_capacity_tiles": num_tiles,
            "total_evicted_tiles": num_tiles - len(chosen),
            "total_dropped_tiles": num_tiles - len(chosen),
            "total_committed_tiles": len(chosen),
            "committed_tile_indices": chosen,
            "rescored": False,
            "max_resident_tiles": num_tiles,
            "compression_ratio": num_tiles / max(len(chosen), 1),
            "size_history": [],
            "score_min": float(lps_t.min()) if lps else None,
            "score_max": float(lps_t.max()) if lps else None,
            "score_mean": float(lps_t.mean()) if lps else None,
            "score_std": float(lps_t.std(unbiased=False)) if len(lps) > 1 else 0.0,
        }
        assert tokens_injected <= budget, (
            f"KV visual tokens {tokens_injected} > budget {budget}"
        )
        return past_key_values, running_mask, stats

    def generate_baseline(self, model, **kwargs):
        pixel_values_list = kwargs["pixel_values_list"]
        questions = kwargs["questions"]
        return generate_answer_standard(model, self.tokenizer, pixel_values_list, questions)

    def clean_answer(self, batch_or_questions, answers):
        return clean_answer(answers)


# ══════════════════════════════════════════════════════════════════════════════
# 共用骨架：online KV streaming 主流程       （from stream_common.py）
# ══════════════════════════════════════════════════════════════════════════════
def pad_and_stack_vision(finalized: list[torch.Tensor]):
    """把每張圖各自 <= budget 的 vision token 序列 padding 到 batch 內最大長度。"""
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
    adapter: InternVLAdapter,
    pixel_values_list,
    questions: list[str],
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,
    chunk_size: int = 1024,
    budget: int = 1024,
    score_fn: str = "info_density",
    tile_agg: str = "max",
    select: str = "topk",
    max_new_tokens: int = 500,
    delay_tiles: int = 2,
    gold_letters: list[str] | None = None,
    oracle_mode: str = "per_tile",
    thumb_attn: bool = False,
    thumb_attn_w: float = THUMB_ATTN_W,
    orig_sizes: list[tuple[int, int]] | None = None,
):
    """True online Vision -> bounded tile memory -> LLM KV pipeline.

    Unlike the previous implementation, this function:
      1. never concatenates all image tiles into ``flat_tiles``;
      2. keeps at most K + delay_tiles tiles resident at any time
         (K = budget // num_image_token); eviction runs on every tile;
      3. selects COMPLETE TILES only;
      4. defers the final top-K commit to end-of-image, so an early
         low-value tile is never written to KV irreversibly;
      5. prefills the selected tiles into the LLM KV cache in spatial
         order once selection is final;
      6. supports arbitrary input batch size by processing each image
         independently (the conservative correctness-first implementation).

    ``budget`` is the final visual-token budget. For InternVL3.5-8B,
    ``budget=2048`` means at most 8 complete 256-token tiles are committed.
    """
    B = len(pixel_values_list)
    assert B == len(questions)

    if select not in ("topk", "spatial_bin", "oracle"):
        raise ValueError(f"Unknown select={select!r}; expected 'topk', 'spatial_bin' or 'oracle'")
    if select == "oracle" and (gold_letters is None or len(gold_letters) != B):
        raise ValueError("select='oracle' needs gold_letters aligned with the image batch")
    if tile_agg != "max":
        print(f"[V2] tile_agg={tile_agg} (default is 'max')")

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

    # info_density 的 relevance 項只 pool「問句本身」（strip 掉選項區塊與作答指示），
    # 而非整段 prompt —— 指令與選項字母會把質心稀釋成跟每個 tile 都無關。
    question_embeds = mean_pool_question_embed(
        adapter.get_input_embeddings(model), tokenizer,
        [strip_to_question(q) for q in questions], dtype,
    )

    # ------------------------------------------------------------
    # Process each image independently.
    # This avoids cross-image padding and makes the O(K) invariant explicit.
    # ------------------------------------------------------------
    for b in range(B):
        pixel_values = pixel_values_list[b]
        question = questions[b]
        question_embed = question_embeds[b]

        print("\n" + "=" * 80)
        print(
            f"[Image {b}] tiles={pixel_values.shape[0]} "
            f"budget={budget} delay_tiles={delay_tiles} score={score_fn} tile_agg={tile_agg}"
        )
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

        # thumbnail-attention prior：一次 thumbnail forward 算出每個 grid tile 的 prior（--thumb_attn 才做）
        tile_prior = None
        if thumb_attn and select != "oracle":
            n_grid = pixel_values.shape[0] - 1
            ow, oh = (orig_sizes[b] if orig_sizes and b < len(orig_sizes) else (1, 1))
            cols, rows = _infer_tile_grid(n_grid, ow, oh)
            t_ta = time.time()
            tile_prior = adapter.thumbnail_saliency_prior(
                model, pixel_values, text_before, text_after, cols, rows, dtype,
            )
            print(f"[Image {b}] thumb-attn prior: grid={cols}x{rows} "
                  f"({'ok' if tile_prior is not None else 'no-op'}) "
                  f"+{time.time() - t_ta:.2f}s")

        if select == "oracle":
            past_key_values, running_mask, stats = adapter.stream_image_to_kv_oracle(
                model=model,
                pixel_values=pixel_values,
                text_before=text_before,
                text_after=text_after,
                gold_letter=gold_letters[b],
                budget=budget,
                vit_batch=vit_batch,
                chunk_size=chunk_size,
                dtype=dtype,
                oracle_mode=oracle_mode,
                past_key_values=past_key_values,
                running_mask=running_mask,
            )
        else:
            past_key_values, running_mask, stats = adapter.stream_image_to_kv(
                model=model,
                pixel_values=pixel_values,
                question_embed=question_embed,
                budget=budget,
                score_fn=score_fn,
                vit_batch=vit_batch,
                chunk_size=chunk_size,
                dtype=dtype,
                delay_tiles=delay_tiles,
                tile_agg=tile_agg,
                select=select,
                tile_prior=tile_prior,
                thumb_attn_w=thumb_attn_w,
                past_key_values=past_key_values,
                running_mask=running_mask,
            )

        torch.cuda.synchronize(DEVICE)
        t_vision_end = time.time()
        vision_times.append(t_vision_end - t_vision_start)

        _mem_mark("Vision+OnlineKV")
        _mem_reset()

        print(
            f"[Image {b}] raw={stats['total_seen_tiles']} tiles -> "
            f"KV={stats['final_tiles']} tiles / {stats['final_size']} tokens; "
            f"evicted={stats['total_evicted_tiles']}; "
            f"dropped={stats['total_dropped_tiles']}; "
            f"max_resident={stats['max_resident_tiles']} tiles"
        )
        print(f"[Image {b}] selected_tile_indices={stats.get('committed_tile_indices', [])}")
        print(
            f"[Image {b}] score spread: "
            f"min={stats.get('score_min')} max={stats.get('score_max')} "
            f"mean={stats.get('score_mean')} std={stats.get('score_std')} "
            f"(tiny std relative to mean => ranking is close to arbitrary)"
        )

        # --------------------------------------------------------
        # Step C: text-after + autoregressive decode
        # --------------------------------------------------------
        tok_after = tokenizer([text_after], return_tensors="pt", padding=True, padding_side="right")
        ids_after = tok_after.input_ids.to(DEVICE)
        mask_after = tok_after.attention_mask.to(DEVICE)

        past_seq_len = past_key_values.get_seq_length()
        running_mask = torch.cat([running_mask, mask_after], dim=1)
        position_ids = torch.arange(past_seq_len, past_seq_len + ids_after.shape[1], dtype=torch.long, device=DEVICE).unsqueeze(0)

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

            past_seq_len = past_key_values.get_seq_length()
            running_mask = torch.cat([running_mask, torch.ones((1, 1), dtype=torch.long, device=DEVICE)], dim=1)
            position_ids = torch.full((1, 1), past_seq_len, dtype=torch.long, device=DEVICE)

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

    return adapter.clean_answer(questions, all_answers), all_stats, timing


# ══════════════════════════════════════════════════════════════════════════════
# Main        （from internvl_svm.py）
# ══════════════════════════════════════════════════════════════════════════════
def save_results_incremental(output_path: str, results: dict):
    """每處理完一張圖就整份重寫一次 JSON。"""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)


def dedup_hrbench_open(datasets):
    """open 模式：每題只保留一筆。

    HR-Bench 官方把每題的四個選項輪換成連續四筆（id 連號 0..3 / 4..7 / ...，
    circular eval 用 id // 4 當題號）。letter 模式下這四筆的 prompt 因選項順序不同
    而互異，但 open 模式的 prompt 只有題目本身、不帶任何選項，四筆會產生一模一樣的
    輸入。全部跑一遍除了浪費四倍算力，還會在「與 baseline 比對 output 相似度」時把
    同一題重複計四次、稀釋統計量。這裡每個題號只保留第一筆。

    id 不是乾淨整數時退回用 question 字串去重，順序維持原本的出現順序。
    """
    seen, kept = set(), []
    for d in datasets:
        raw = str(d.get("id", "")).strip()
        try:
            key = int(raw) // 4
        except ValueError:
            key = d.get("question", "")
        if key in seen:
            continue
        seen.add(key)
        kept.append(d)
    print(f"[dedup_hrbench_open] {len(datasets)} 筆 -> {len(kept)} 筆（每題一問）")
    return kept


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--max_num",     type=int, default=48,   help="max dynamic tiles")
    parser.add_argument("--vit_batch",   type=int, default=4,    help="ViT micro-batch size")
    parser.add_argument("--chunk_size",  type=int, default=1024, help="LLM chunked-prefill size")
    parser.add_argument("--batch_size",  type=int, default=1,    help="Image batch size")
    parser.add_argument("--budget",      type=int, default=1024, help="The maximum number of vision tokens per image")
    parser.add_argument("--score_fn",    type=str, default="info_density", choices=["l2_norm", "info_density", "random"])
    # 註：token-level merge（merge / merge_spatial）已從 V2 拿掉，selection 一律是
    #     tile-level 硬淘汰。tag 格式：budget=<B>_<score_fn>_delay<N>[_<tile_agg>][_<select>]
    #     [_thumbattn[_w<x>]]（舊格式的 "_evict_" 已移除，舊 Block A JSON 不再相容）。
    parser.add_argument(
        "--select", type=str, default="topk", choices=["topk", "spatial_bin", "oracle"],
        help="存活 tile 的選擇策略。topk=純看分數；"
             "spatial_bin=依空間順序分箱、每箱取最高分，存活 tile 保證鋪滿全圖；"
             "oracle=偷看 gold answer 的診斷用上界（不可部署）：按「單獨補這塊 tile"
             "能把正解字母的機率拉高多少」排序，取 top-K。需要 dataset 提供 answer。",
    )
    parser.add_argument(
        "--oracle_mode", type=str, default="per_tile", choices=["per_tile", "greedy"],
        help="--select oracle 專用：per_tile=每塊獨立打分取 top-(K-1)，O(N) 次 LM 前向；"
             "greedy=貪婪逐步加塊、每步挑最能拉高正解機率的，O(N*K) 次前向，會處理 tile 冗餘。",
    )
    parser.add_argument("--delay_tiles", type=int, default=0,
                        help="Extra candidate-pool slots beyond K=budget//num_image_token. "
                             "The pool holds K+delay_tiles tiles")
    parser.add_argument(
        "--tile_agg", type=str, default="max",
        choices=["mean", "max", "topk_mean", "mean_std", "quantile"],
        help="how a tile's per-token scores are aggregated into one tile score "
             "before eviction ranking. 'mean' = current baseline.",
    )
    # Tier-0 scorer 清理（novelty / rescore / relevance 只 pool 問句 / signal_trim /
    # coverage_floor）已是 info_density 的預設行為，沒有 CLI 旗標；見檔案頂端常數。
    parser.add_argument("--thumb_attn", action="store_true",
                        help="用 thumbnail 的一次 LLM forward 取 attention → 16x16 粗 saliency "
                             "→ 每個 grid tile 一個 prior，跟 info_density 分數線性混合。peak memory "
                             "不動，TTFT 多一個與 tile 數無關的短 prefill。tag 後綴 _thumbattn。")
    parser.add_argument("--thumb_attn_w", type=float, default=THUMB_ATTN_W,
                        help="thumbnail-attention prior 的混合權重 w："
                             "final = (1-w)*info_density + w*thumbnail_prior。預設 THUMB_ATTN_W。")
    parser.add_argument("--run_stream", action="store_true", help="run online KV pipeline")
    parser.add_argument("--run_standard", action="store_true",
                            help="run the official (uncompressed) HF generate() path to produce reference answers")

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--dataset", type=str, default="hrbench", choices=["mmmu", "docvqa", "hrbench"],
                        help="which HF dataset to load when --use_ds is set")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_internvl.json")
    args = parser.parse_args()

    # 固定 seed，只是為了讓 --score_fn random 每次跑都可重現，不需要當成可調參數。
    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)

    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    # ── 1. 載入模型 ──
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"downsample_ratio         : {model.downsample_ratio}")
    print(f"select_layer             : {model.select_layer}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")

    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    adapter = InternVLAdapter(tokenizer, model, budget=args.budget)

    # ── 2. 載入圖像與問題 ──
    if args.use_ds:
        from process_common import load_docvqa, load_mmmu, load_hrbench
        if args.dataset == "mmmu":
            # MMMU：選擇題，prompt 會帶選項；每筆會多存 answer/options/question_type 進 meta
            datasets = load_mmmu(split="validation", num_image=args.num_images)
        elif args.dataset == "docvqa":
            datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                    num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
        elif args.dataset == "hrbench":
            datasets = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT,
                                    num_image=args.num_images, prompt_mode=HRBENCH_PROMPT)
            if HRBENCH_PROMPT == "open":
                datasets = dedup_hrbench_open(datasets)
        else:  
            return
    else:
        from process_common import load_local
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

    # ── 2b. MMMU / HR-Bench / DocVQA：把每題的 answer/options/... 存進 JSON 的 meta ──
    #        HR-Bench 多存 answer_text（open 模式下 evaluate 拿它比對自由輸出）與 category。
    #        DocVQA 多存 answers（官方允許的多個可接受答案，ANLS 要取 max），讓
    #        evaluate_anls.py 直接讀 meta 算 GT，不用重載 HF dataset、沒有對齊風險。
    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
             "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
             "answers": d.get("answers", [])}
            for k, d in enumerate(datasets)
        ]

    tag = f"budget={args.budget}_{args.score_fn}_delay{args.delay_tiles}"
    if args.tile_agg != "max":
        # 只有非預設聚合方式才進 tag（預設 max 不進 tag）。
        tag += f"_{args.tile_agg}"
    if args.select != "topk":
        # 非預設選擇策略才進 tag，維持舊結果檔（select=topk）能被 resume 認得。
        tag += f"_{args.select}"
    if args.select == "oracle" and args.oracle_mode != "per_tile":
        # oracle 的 per_tile 是預設，不進 tag；greedy 才加後綴，兩者結果分開存。
        tag += f"_{args.oracle_mode}"
    if args.thumb_attn:
        tag += "_thumbattn"
        if abs(args.thumb_attn_w - THUMB_ATTN_W) > 1e-9:
            tag += f"_w{args.thumb_attn_w:g}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        print(f"\nAll done. Total time: {elapsed:.2f} s")
        return

    # ── 4. 逐張（或依 batch_size 分小批）處理 ──
    for i in range(0, len(datasets), args.batch_size):
        batch_datasets = datasets[i:i + args.batch_size]

        print(f"\n{'='*70}")
        print(f"[{i+1}~{i+len(batch_datasets)}/{len(datasets)}] Processing Batch...")
        print(f"{'='*70}")

        # 只在真正要處理這一小批時才做影像前處理（tile 化），處理完就可以釋放
        # dynamic_preprocess 在網格 tile 之外，永遠會「再多加 1 張」全圖縮圖
        # （internvl_preprocess.py:72-73），所以 --max_num 傳給它的是「網格上限」，
        # 實際吃到的 tile 數 = min(max_num, 網格) + 1（縮圖）。這裡先扣掉 1，讓
        # --max_num 在這支檔案裡的語意變成「總 tile 數上限（含縮圖）」，才能跟
        # budget/K = capacity // num_image_token 的算法一致 —— 否則例如 --max_num
        # 4 的「直接少切」baseline 實際會吃到 5 tile = 1280 tokens，比同樣號稱
        # budget=1024（K=4）的串流方法多 25% token，兩者不是同一個預算，比較會失真。
        max_num_grid = max(1, args.max_num - 1)
        pixel_values_list = [
            load_image_tiles(d["image"], input_size=448, max_num=max_num_grid) for d in batch_datasets
        ]
        questions = [d["question"] for d in batch_datasets]
        orig_sizes = [tuple(d["image"].size) for d in batch_datasets]  # (W, H)，給 thumbnail-attention prior 重建 tile 網格
        tile_counts = [pv.shape[0] for pv in pixel_values_list]  # 每張圖的 raw tile 數
        print(f"  tiles per image = {tile_counts}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("internvl_standard_generate"):
                    ref_answers = adapter.generate_baseline(model, pixel_values_list=pixel_values_list, questions=questions)
                    output_results["references"] += ref_answers

                print("\n[Baseline Answer]")
                for i, answer in enumerate(ref_answers):
                    print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")

            if args.run_stream:
                if tag not in output_results["candidates"]: output_results["candidates"][tag] = []

                gold_letters = None
                if args.select == "oracle":
                    gold_letters = [str(d.get("answer", "") or "").strip() for d in batch_datasets]
                    if not all(gold_letters):
                        raise ValueError(
                            "--select oracle 需要每筆都有 gold answer letter（dataset 要提供 'answer'）；"
                            f"這批有空值：{gold_letters}"
                        )

                print(f"\n[Online Stream] Running online kv stream with memory bank ...")
                with measure_peak_memory("internvl_online_kv_memory_bank"):
                    answers, all_stats, timing = run_online_kv_with_memory_bank(
                        model, adapter, pixel_values_list, questions,
                        dtype=dtype,
                        vit_batch=args.vit_batch,
                        chunk_size=args.chunk_size,
                        budget=args.budget,
                        score_fn=args.score_fn,
                        tile_agg=args.tile_agg,
                        select=args.select,
                        max_new_tokens=MAX_NEW_TOKENS,
                        delay_tiles=args.delay_tiles,
                        gold_letters=gold_letters,
                        oracle_mode=args.oracle_mode,
                        thumb_attn=args.thumb_attn,
                        thumb_attn_w=args.thumb_attn_w,
                        orig_sizes=orig_sizes,
                    )
                    output_results["candidates"][tag] += answers

                print("\n[Online KV Answer]")
                for i, answer in enumerate(answers):
                    print(f"\n{i} -> [{batch_datasets[i]['question']}]\n{answer}")

            del pixel_values_list, questions

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on batch: {e}")
            raise

        gc.collect()
        torch.cuda.empty_cache()

        if args.save:
            save_results_incremental(args.output_json, output_results)
            print(f"\n[Saved] {args.output_json} updated "
                  f"({len(output_results['references'])}/{len(datasets)} images done)")

    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"\nAll done. Total time: {elapsed:.2f} s")

    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
