# ══════════════════════════════════════════════════════════════════════════════
# LLaVA-OneVision 串流 KV + Memory Bank —— 單檔獨立版
#
# 這支檔案把原本分散在六支檔案裡「只跟 LLaVA-OneVision 有關」的部分，原封不動
# 合併進來：
#     llava_ov_svm.py       → main / save_results_incremental
#     llava_ov_adapter.py   → LlavaOVAdapter
#     llava_ov_core.py      → build_prompt / split_prompt_at_vision / encode_patch /
#                             num_image_tokens_per_patch / compute_packed_row_layout /
#                             generate_answer_standard / clean_answer
#     stream_common.py      → run_online_kv_with_memory_bank / pad_and_stack_vision
#     stream_adapters.py    → get_optimal_cuda_device / DEVICE / mean_pool_question_embed /
#                             measure_peak_memory
#     stream_memory_bank.py → TileStreamingMemoryBank + 三個 score function
#
# 函式內部的數學與執行順序完全沒有改，只做了三件事：
#   1. 跨檔 import 改成同檔引用。
#   2. 拿掉 StreamModelAdapter ABC —— 單一模型不需要多型，LlavaOVAdapter 直接是
#      普通 class（方法簽章與 docstring 保留）。
#   3. main() 裡 run_online_kv_with_memory_bank() 的回傳值從 2 個改成 3 個
#      (answers, all_stats, timing) —— 骨架本來就回傳 timing，舊的 llava_ov_svm.py
#      少接一個會 unpack 失敗；這裡順手接起來並比照 InternVL 印出各段秒數。
#
# 仍保留 import 的外部相依（本來就跟 InternVL 無關、且已各自獨立）：
#   exp/llava_ov_preprocess.py → build_model / load_image_patches
#   process_common.py          → load_local / load_hf_dataset
#
# 舊的 llava_ov_svm.py / llava_ov_adapter.py / llava_ov_core.py / stream_*.py
# 原地保留當備份，完全不受這支檔案影響。
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import os
import gc
import math
import time
import json
import argparse
import contextlib

import torch
import torch.nn.functional as F
from transformers.models.llava_onevision.modeling_llava_onevision import (
    unpad_image, get_anyres_image_grid_shape,
)

from llava_ov_preprocess import build_model, load_image_patches


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
# LLaVA-OneVision 專屬低階函式         （from llava_ov_core.py）
# ══════════════════════════════════════════════════════════════════════════════
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
# LLaVA-OneVision 是標準 HF 模型，沒有 trust_remote_code 自帶的 chat helper，
# 所以這裡直接走 processor(...) + model.generate(...) 這條「完全沒有做任何壓縮」
# 的官方路徑 —— 會跑滿完整的 pack_image_features()（unpad + 必要時的雙線性內插 +
# 逐 token-row image_newline），是跟 streaming memory bank 版本比較答案品質時
# 真正該拿來當 ground truth 的基準。
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
    dummy tensor 跑一次拿 shape。

    回傳 (num_rows, row_width_with_newline)。
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


# ══════════════════════════════════════════════════════════════════════════════
# Importance scoring + Tile Streaming Memory Bank    （from stream_memory_bank.py）
# ══════════════════════════════════════════════════════════════════════════════
def score_l2_norm(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    # **_ 吞掉 norm_stats 等只有 info_density 用得到的 kwarg。
    return tokens.float().norm(dim=-1)

def score_random(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    return torch.rand(tokens.shape[0], device=tokens.device)

def score_information_density(
    tokens, question_embed,
    alpha=0.3, beta=0.3, gamma=0.4,
    knn_k=3,
    use_novelty: bool = False,
    norm_stats: dict | None = None,
    # ↑ 若提供 {"mean":..., "std":...}（來自 finalize() 時對「所有已收集 tile」
    #   算出的全域統計量），signal 改用全域 z-score → sigmoid 正規化。
    #   若為 None，fallback 回原本「單一 tile 內部 max 正規化」的行為。
):
    """
    Question-aware Information Density Score
    Score = alpha*S + beta*N + gamma*R，三項各自正規化到 [0,1] 後再加權。

    tokens:         [N, D]  (已經過 projector，跟 LLM embedding space 同維度)
    question_embed: [D]     (question 文字的 mean-pooled LLM embedding，同一個 space)
    """
    x = tokens.float()
    N = x.shape[0]

    # ---- Signal Strength：用相對排名而非除以 max，避免單一離群值壓縮整批分數 ----
    raw_norm = x.norm(dim=-1)

    if norm_stats is not None:
        z = (raw_norm - norm_stats["mean"]) / norm_stats["std"]
        signal = torch.sigmoid(z)
    elif N > 1:
        signal = raw_norm / (raw_norm.max() + 1e-9)
    else:
        signal = torch.zeros(N, device=x.device)

    # ---- Novelty：top-k 最相似的平均相似度 ----
    x_norm = F.normalize(x, dim=-1)
    sim = x_norm @ x_norm.T
    sim.fill_diagonal_(-1.0)
    k = min(knn_k, max(N - 1, 1))
    topk_sim = sim.topk(k, dim=-1).values.mean(dim=-1) if N > 1 else torch.zeros(N, device=x.device)

    if use_novelty:
        novelty = ((1 - topk_sim) / 2).clamp(0.0, 1.0)
    else:
        novelty = torch.zeros(N, device=x.device)
        beta = 0.0

    # ---- Question Relevance ----
    if question_embed is None or (isinstance(question_embed, str)):
        relevance = torch.zeros(N, device=x.device)
        gamma_eff = 0.0
    else:
        q = F.normalize(question_embed.float().reshape(-1), dim=-1)
        relevance = ((x_norm @ q) + 1) / 2   # [-1,1] -> [0,1]
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


class TileStreamingMemoryBank:
    """
    以 Tile 為單位（InternVL: 256-token 的 448px tile；LLaVA-OV: 官方 pack 之後
    攤平的「一列」，長度 = curr_width + 1）。
    add_tile() 只負責累積 + 算分,真正的淘汰決策延後到 finalize(),
    並用「空間分箱 + 箱內取最高分」保證全圖覆蓋不留空洞。
    """
    def __init__(
        self,
        capacity: int,
        dim: int, device, dtype,
        score_fn: str = "l2_norm",
        mode: str = "evict",
        num_image_token: int = 256,
        question_embed=None,
        tile_agg: str = "mean",
        agg_topk_frac: float = 0.1,
        agg_quantile: float = 0.9,
        agg_std_lambda: float = 1.0,
        merge_weight: str = "score",
        merge_sim_thresh: float = -1.0,
        select: str = "topk",
        protected_tiles: int = 1,
        protect_position: str = "last",
        max_newline_tokens: int = 0,
    ):
        self.capacity_tiles = max(1, (capacity - max_newline_tokens) // num_image_token)
        self.num_image_token = num_image_token
        self.dim = dim
        self.device = device
        self.dtype = dtype
        self.protected_tiles = min(protected_tiles, self.capacity_tiles)
        self.protect_position = protect_position

        self.tiles = []        # list of [num_image_token, D]，依原始空間(raster)順序累積
        self.tile_scores = []  # list of float
        self.score_fn = SCORE_FUNCS[score_fn]
        self.mode = mode
        self.question_embed = question_embed
        self.tile_agg = tile_agg
        self.agg_topk_frac = agg_topk_frac
        self.agg_quantile = agg_quantile
        self.agg_std_lambda = agg_std_lambda
        self.merge_weight = merge_weight
        self.merge_sim_thresh = merge_sim_thresh
        self.select = select

        self.total_seen_tiles = 0
        self.total_dropped_tiles = 0
        self.size_history = []

    def _aggregate_tile_score(self, token_scores: torch.Tensor) -> float:
        """把一個 tile 內 [num_image_token] 個 token 分數壓成單一 tile 分數。"""
        s = token_scores.float().reshape(-1)
        n = s.numel()
        if n == 0:
            return 0.0
        mode = self.tile_agg
        if mode == "mean":
            return s.mean().item()
        if mode == "max":
            return s.max().item()
        if mode == "topk_mean":
            k = max(1, int(round(self.agg_topk_frac * n)))
            return s.topk(k).values.mean().item()
        if mode == "mean_std":
            return (s.mean() + self.agg_std_lambda * s.std(unbiased=False)).item()
        if mode == "quantile":
            return s.quantile(self.agg_quantile).item()
        raise ValueError(f"unknown tile_agg={mode!r}")

    def add_tile(self, tile_tokens: torch.Tensor):
        token_scores = self.score_fn(tile_tokens, self.question_embed)
        tile_score = self._aggregate_tile_score(token_scores)

        self.tiles.append(tile_tokens)
        self.tile_scores.append(tile_score)
        self.total_seen_tiles += 1
        self.size_history.append(min(len(self.tiles), self.capacity_tiles) * self.num_image_token)

    def _compute_global_norm_stats(self):
        """把目前 self.tiles 裡所有 token 的 raw L2 norm 攤平算一次全域 mean/std。"""
        all_norms = torch.cat([t.float().norm(dim=-1) for t in self.tiles])
        return {"mean": all_norms.mean(), "std": all_norms.std() + 1e-9}

    def _protected_indices(self, n):
        if self.protected_tiles <= 0:
            return set()
        if self.protect_position == "first":
            return set(range(self.protected_tiles))
        return set(range(n - self.protected_tiles, n))

    def _assemble_flat(self, keep_idx, image_newline):
        """純粹依 raster 順序把保留下來的 tile concat 起來。
        image_newline 不是 None 的話，在最後面補一個 newline token。"""
        pieces = [self.tiles[i] for i in keep_idx]
        if image_newline is not None:
            newline_tok = image_newline.reshape(1, -1).to(device=self.device, dtype=self.dtype)
            pieces.append(newline_tok)
        return torch.cat(pieces, dim=0)

    def _assemble_grid(self, keep_idx, grid_shape, image_newline):
        """
        Grid-aware 重組：假設 self.tiles[0] 是被保護的 base image（整圖縮圖），
        self.tiles[1:] 依 row-major 順序對應到 grid_shape。這一列只要還有任何一個
        patch 存活，就把存活的 patch 依原本 column 順序接起來，接完這一列才補一個 newline。
        """
        num_patch_height, num_patch_width = grid_shape
        keep_set = set(keep_idx)

        pieces = []
        if 0 in keep_set:
            pieces.append(self.tiles[0])   # base image，不接 newline，直接放最前面

        newline_tok = None
        if image_newline is not None:
            newline_tok = image_newline.reshape(1, -1).to(device=self.device, dtype=self.dtype)

        for r in range(num_patch_height):
            row_pieces = [
                self.tiles[1 + r * num_patch_width + c]
                for c in range(num_patch_width)
                if (1 + r * num_patch_width + c) in keep_set
            ]
            if row_pieces:
                pieces.extend(row_pieces)
                if newline_tok is not None:
                    pieces.append(newline_tok)

        if not pieces:
            return torch.empty(0, self.dim, device=self.device, dtype=self.dtype)
        return torch.cat(pieces, dim=0)

    def _fold_group(self, src_tiles, tgt_tiles, norm_stats, info):
        """把 src_tiles 的每個 token，依 cosine 折進 tgt_tiles 的 token 裡最相近的
        那個（加權平均），就地改寫 self.tiles[tgt]。"""
        if not src_tiles or not tgt_tiles:
            return
        tgt_tok = torch.cat([self.tiles[i] for i in tgt_tiles], dim=0).float()   # [T, D]
        src_tok = torch.cat([self.tiles[i] for i in src_tiles], dim=0).float()   # [S, D]

        sim = F.normalize(src_tok, dim=-1) @ F.normalize(tgt_tok, dim=-1).T      # [S, T]
        best_sim, tgt = sim.max(dim=-1)                                          # [S]

        if self.merge_sim_thresh > -1.0:
            m = best_sim >= self.merge_sim_thresh
        else:
            m = torch.ones_like(best_sim, dtype=torch.bool)
        info["tokens_hard_dropped"] += int((~m).sum())
        src_tok, tgt, best_sim = src_tok[m], tgt[m], best_sim[m]
        if src_tok.shape[0] == 0:
            return

        if self.merge_weight == "score":
            w_t = torch.cat([
                self.score_fn(self.tiles[i], self.question_embed, norm_stats=norm_stats).reshape(-1)
                for i in tgt_tiles
            ]).float().clamp(min=1e-6)
            w_s = torch.cat([
                self.score_fn(self.tiles[i], self.question_embed, norm_stats=norm_stats).reshape(-1)
                for i in src_tiles
            ]).float()[m].clamp(min=1e-6)
        else:
            w_t = torch.ones(tgt_tok.shape[0], device=tgt_tok.device)
            w_s = torch.ones(src_tok.shape[0], device=src_tok.device)

        # new_t = (w_t·t + Σ_{s→t} w_s·s) / (w_t + Σ_{s→t} w_s)
        num = tgt_tok * w_t.unsqueeze(-1)
        num.index_add_(0, tgt, src_tok * w_s.unsqueeze(-1))
        den = w_t.clone()
        den.index_add_(0, tgt, w_s)
        merged = (num / den.unsqueeze(-1)).to(self.dtype)

        off = 0
        for i in tgt_tiles:
            L = self.tiles[i].shape[0]
            self.tiles[i] = merged[off:off + L].clone()
            off += L

        info["tokens_merged"] += int(m.sum())
        info["_sim_sum"] += float(best_sim.sum())
        info["_sim_cnt"] += int(m.sum())

    def _merge_evicted_tokens(self, keep_idx, prot_idx, norm_stats):
        """把被淘汰 tile 的 token 折進存活 token（就地改寫 self.tiles[target]）。"""
        info = {"tokens_merged": 0, "tokens_hard_dropped": 0, "mean_merge_sim": 0.0,
                "_sim_sum": 0.0, "_sim_cnt": 0}
        keep_set = set(keep_idx)
        drop_idx = [i for i in range(len(self.tiles)) if i not in keep_set]
        if not drop_idx:
            return {k: info[k] for k in ("tokens_merged", "tokens_hard_dropped", "mean_merge_sim")}

        target_pool = [i for i in keep_idx if i not in set(prot_idx)] or list(keep_idx)

        if self.mode == "merge_spatial":
            assign: dict[int, list[int]] = {}
            for e in drop_idx:
                t = min(target_pool, key=lambda k: (abs(k - e), k))
                assign.setdefault(t, []).append(e)
            for t, evs in assign.items():
                self._fold_group(evs, [t], norm_stats, info)
        else:
            self._fold_group(drop_idx, target_pool, norm_stats, info)

        if info["_sim_cnt"] > 0:
            info["mean_merge_sim"] = info["_sim_sum"] / info["_sim_cnt"]
        return {k: info[k] for k in ("tokens_merged", "tokens_hard_dropped", "mean_merge_sim")}

    def finalize(self, grid_shape=None, image_newline=None):
        """
        grid_shape   : None → flat 模式。
        image_newline: None → 不插入任何 newline token。
        """
        n = len(self.tiles)
        if n == 0:
            return torch.empty(0, self.dim, device=self.device, dtype=self.dtype), {
                "final_size": 0, "total_seen": 0, "total_dropped": 0,
                "compression_ratio": 0.0, "size_history": self.size_history,
                "work_budget": 0, "grid_patches_kept": 0, "rows_kept": 0,
            }

        merge_stats = {"tokens_merged": 0, "tokens_hard_dropped": 0, "mean_merge_sim": 0.0}

        if n <= self.capacity_tiles or self.mode == "fifo":
            work_budget = None   # 沒有真的做 eviction，跟原本一樣
            keep_idx = (
                list(range(max(0, n - self.capacity_tiles), n))
                if self.mode == "fifo" and n > self.capacity_tiles else list(range(n))
            )
        else:
            norm_stats = self._compute_global_norm_stats()
            self.tile_scores = [
                self._aggregate_tile_score(
                    self.score_fn(self.tiles[i], self.question_embed, norm_stats=norm_stats)
                )
                for i in range(n)
            ]

            prot_idx = self._protected_indices(n)
            work_budget = self.capacity_tiles - len(prot_idx)
            work_idx = [i for i in range(n) if i not in prot_idx]

            if self.select != "spatial_bin":
                work_idx.sort(key=lambda i: self.tile_scores[i], reverse=True)
                selected = work_idx[:work_budget]
            else:
                bin_edges = torch.linspace(0, len(work_idx), steps=work_budget + 1).long().tolist()
                selected = []
                for k in range(work_budget):
                    lo, hi = bin_edges[k], bin_edges[k + 1]
                    if lo >= hi:
                        hi = min(lo + 1, len(work_idx))
                    bin_local = work_idx[lo:hi]
                    best_local = max(bin_local, key=lambda i: self.tile_scores[i])
                    selected.append(best_local)

            keep_idx = sorted(set(prot_idx) | set(selected))
            self.total_dropped_tiles += (n - len(keep_idx))

            if self.mode in ("merge", "merge_spatial"):
                merge_stats = self._merge_evicted_tokens(keep_idx, prot_idx, norm_stats)

        if grid_shape is not None:
            final_tokens = self._assemble_grid(keep_idx, grid_shape, image_newline)
            num_patch_height, num_patch_width = grid_shape
            keep_set = set(keep_idx)
            grid_patches_kept = sum(1 for i in keep_idx if i != 0)
            rows_kept = sum(
                1 for r in range(num_patch_height)
                if any((1 + r * num_patch_width + c) in keep_set for c in range(num_patch_width))
            )
            grid_positions_kept = sorted(
                ((i - 1) // num_patch_width, (i - 1) % num_patch_width)
                for i in keep_idx if i != 0
            )
        else:
            final_tokens = self._assemble_flat(keep_idx, image_newline)
            grid_patches_kept, rows_kept, grid_positions_kept = len(keep_idx), None, None

        stats = {
            "final_size": final_tokens.shape[0],
            "total_seen": self.total_seen_tiles * self.num_image_token,
            "total_dropped": self.total_dropped_tiles * self.num_image_token,
            "compression_ratio": self.total_seen_tiles / max(len(keep_idx), 1),
            "size_history": self.size_history,
            "work_budget": work_budget,
            "grid_patches_kept": grid_patches_kept,
            "rows_kept": rows_kept,
            "grid_positions_kept": grid_positions_kept,
            "tokens_merged": merge_stats["tokens_merged"],
            "tokens_hard_dropped": merge_stats["tokens_hard_dropped"],
            "mean_merge_sim": round(merge_stats["mean_merge_sim"], 4),
        }
        return final_tokens, stats


# ══════════════════════════════════════════════════════════════════════════════
# LLaVA-OneVision adapter   （from llava_ov_adapter.py，拿掉 StreamModelAdapter ABC）
# ══════════════════════════════════════════════════════════════════════════════
class LlavaOVAdapter:
    """把 llava_ov_svm.py 裡「已確認沒問題」的 Step 1~5 邏輯原封不動包起來。

    encode_and_bank() 就是原本 run_online_kv_with_memory_bank() 的 Step 3~5，逐行
    照搬（官方 pack_image_features → 只有超過 budget 才用 TileStreamingMemoryBank
    在「列」粒度上淘汰），沒有改寫任何數學或順序。
    """

    def __init__(self, processor):
        self.processor = processor
        self.tokenizer = processor.tokenizer

    # ── Step 1：prompt 切分 ──
    def build_text_segments(self, questions):
        text_before_list, text_after_list = [], []
        for question in questions:
            prompt = build_prompt(self.processor, question)
            tb, ta = split_prompt_at_vision(prompt)
            text_before_list.append(tb)
            text_after_list.append(ta)
        return text_before_list, text_after_list

    def get_tokenizer(self):
        return self.tokenizer

    def get_input_embeddings(self, model):
        return model.get_input_embeddings()

    def lm_prefill(self, model, **kwargs):
        with torch.no_grad():
            return model.model.language_model(**kwargs)

    def lm_decode_step(self, model, **kwargs):
        with torch.no_grad():
            out = model.model.language_model(**kwargs)
            logits = model.lm_head(out.last_hidden_state[:, -1, :])
        return logits, out.past_key_values

    # ── Step 3~5：per-patch SigLIP → 官方 pack_image_features → (可選) 列淘汰 ──
    def encode_and_bank(
        self, model, pixel_values_list, image_sizes_list,
        question_embeds, budget, score_fn, merge_mode, vit_batch, dtype,
        tile_agg: str = "mean",
        merge_weight: str = "score", select: str = "topk",
    ):
        B = len(pixel_values_list)
        num_patches_list = [pv.shape[0] for pv in pixel_values_list]
        num_image_token = num_image_tokens_per_patch(model)
        if budget < num_image_token:
            raise ValueError(
                f"budget={budget} 小於單一 patch 的 token 數 ({num_image_token})，"
                f"base image 本身就無法塞進這個 budget，這個實驗設定不可行。"
            )

        print(f"\n[Memory Bank] budget={budget}  score_fn={score_fn}  mode={merge_mode}  tile_agg={tile_agg}")
        print(f"  patches per image (before compression) : {num_patches_list}")
        print(f"  raw vision tokens per image             : "
              f"{[n * num_image_token for n in num_patches_list]}")

        # ── Step 3：per-patch SigLIP（micro-batch 串流，控制 ViT 階段的 peak memory）──
        flat_patches = torch.cat(pixel_values_list, dim=0)
        owner = []
        for b, n in enumerate(num_patches_list):
            owner.extend([b] * n)

        D_llm = model.config.text_config.hidden_size
        per_image_tokens = [[] for _ in range(B)]

        for i in range(0, flat_patches.shape[0], vit_batch):
            chunk = flat_patches[i:i + vit_batch].to(DEVICE, dtype=dtype)
            owner_chunk = owner[i:i + vit_batch]

            patch_tokens = encode_patch(model, chunk, dtype=dtype)
            del chunk
            torch.cuda.empty_cache()

            for j, b in enumerate(owner_chunk):
                per_image_tokens[b].append(patch_tokens[j])

            mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9
            print(f"  ├─> [ViT] flat patch {i}~{min(i + vit_batch, flat_patches.shape[0])}"
                  f"/{flat_patches.shape[0]} done, peak alloc={mem:.2f} GB")

        del flat_patches
        per_image_tokens = [torch.stack(toks, dim=0) for toks in per_image_tokens]

        # ── Step 4：官方 pack_image_features（unpad + 必要時內插 + image_newline）──
        image_newline = model.model.image_newline
        with torch.no_grad():
            packed_list, feature_lens = model.model.pack_image_features(
                per_image_tokens,
                torch.stack(image_sizes_list).to(DEVICE),
                image_newline=image_newline,
            )
        del per_image_tokens

        print("\n[Pack] official pack_image_features() output (unpad + interpolate + newline done):")
        for b in range(B):
            print(f"  image {b}: packed_len={packed_list[b].shape[0]}  (budget={budget})")

        # ── Step 5：只有在官方 pack 出來的長度還是超過 budget 時，才用
        #    TileStreamingMemoryBank 在「一列（含 newline）」的粒度上做進一步淘汰。──
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
                image_sizes_list[b], model.config.image_grid_pinpoints,
                model.config.vision_config.image_size,
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
                question_embed=question_embeds[b], tile_agg=tile_agg,
                merge_weight=merge_weight, select=select,
                protected_tiles=0,   # base image 已經在迴圈外處理，bank 只管網格列
            )
            for r in range(num_rows):
                bank.add_tile(rows[r])   # 評分含這一列的 newline token，權重很小可忽略

            final_grid_tokens, stats = bank.finalize()   # flat 模式：newline 已內建在每個 tile 裡，不用再插
            finalized.append(torch.cat([base_tokens, final_grid_tokens], dim=0))

            # bank 只管網格列，final_size/total_seen 預設不含 base，這裡補回去。
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
        return finalized, all_stats

    def generate_baseline(self, model, batch, **kwargs):
        return generate_answer_standard(model, self.processor, batch)

    def clean_answer(self, batch_or_questions, answers):
        # run_online_kv_with_memory_bank 只傳 questions（list[str]）進來，但
        # clean_answer() 需要 dataset["question"] 才能把 echo 回來的問題文字
        # 從答案裡 strip 掉。這裡包一層轉換，clean_answer 本身邏輯不變。
        pseudo_datasets = [{"question": q} for q in batch_or_questions]
        return clean_answer(pseudo_datasets, answers)


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
    adapter: LlavaOVAdapter,
    pixel_values_list,          # list[Tensor]，LLaVA-OV 語意 = patches
    questions: list[str],
    image_sizes_list=None,      # LLaVA-OV 需要
    dtype: torch.dtype = torch.bfloat16,
    vit_batch: int = 4,
    chunk_size: int = 1024,
    budget: int = 1024,
    score_fn: str = "info_density",
    merge_mode: str = "evict",
    tile_agg: str = "mean",
    merge_weight: str = "score",
    select: str = "topk",
    max_new_tokens: int = 300,
):
    B = len(pixel_values_list)
    assert B == len(questions)
    tokenizer = adapter.get_tokenizer()

    torch.cuda.synchronize(DEVICE)
    t_start = time.time()                      # ← TTFT 起點

    # ── Sub-phase peak-memory 量測（Prefill / Vision Encode / Injection / Decode）──
    _mem_phases: dict[str, dict[str, float]] = {}

    def _mem_reset():
        torch.cuda.synchronize(DEVICE)
        torch.cuda.reset_peak_memory_stats(DEVICE)

    def _mem_mark(_name: str):
        torch.cuda.synchronize(DEVICE)
        _mem_phases[_name] = {
            "peak": torch.cuda.max_memory_allocated(DEVICE) / 1e9,
            "resting": torch.cuda.memory_allocated(DEVICE) / 1e9,
        }

    _mem_reset()

    # ── Step A0：question embedding ──
    question_embeds = mean_pool_question_embed(
        adapter.get_input_embeddings(model), tokenizer, questions, dtype,
    )

    # ── Step A1：prompt 切分 ──
    text_before_list, text_after_list = adapter.build_text_segments(questions)

    # ── Step A2：text_before prefill ──
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

    _mem_mark("Prefill")         # Step A0~A2：question embed + text_before prefill
    _mem_reset()

    torch.cuda.synchronize(DEVICE)
    t_vision_start = time.time()                # ← encode_and_bank 開始

    # ── Step B：模型專屬的 encode + bank ──
    finalized, all_stats = adapter.encode_and_bank(
        model, pixel_values_list, image_sizes_list,
        question_embeds, budget, score_fn, merge_mode, vit_batch, dtype,
        tile_agg=tile_agg,
        merge_weight=merge_weight, select=select,
    )

    torch.cuda.synchronize(DEVICE)
    t_vision_end = time.time()                  # ← encode_and_bank 結束

    _mem_mark("Vision Encode")   # Step B：ViT encode + memory bank finalize
    _mem_reset()

    # ── Step C：padding + Online KV chunked injection ──
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

        print(f"  ├─> [LLM flush] {tokens_done}/{max_len} vision tokens injected, "
              f"alloc={torch.cuda.memory_allocated(DEVICE)/1e9:.2f} GB")

    del vision_tensor, vision_mask

    torch.cuda.synchronize(DEVICE)
    t_injection_end = time.time()                # ← for-loop 結束、del vision_tensor 之後

    _mem_mark("Injection")       # Step C：pad_and_stack + chunked KV injection
    _mem_reset()

    # ── Step D：text_after + decode ──
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

    torch.cuda.synchronize(DEVICE)
    t_first_token = time.time()                  # ← 第一個 token 算出來

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

    _mem_mark("Decode")          # Step D：text_after + 自回歸 decode loop

    torch.cuda.synchronize(DEVICE)

    _overall_peak = max(p["peak"] for p in _mem_phases.values())
    print("\n[Peak Memory Breakdown]  reset_peak_memory_stats() → max_memory_allocated() per phase")
    for _name in ("Prefill", "Vision Encode", "Injection", "Decode"):
        _p = _mem_phases[_name]
        print(f"  {_name:<14s} peak={_p['peak']:6.2f} GB   resting_end={_p['resting']:6.2f} GB   "
              f"transient=+{_p['peak'] - _p['resting']:5.2f} GB")
    print(f"  {'OVERALL':<14s} peak={_overall_peak:6.2f} GB   "
          f"(外層 measure_peak_memory 被逐段 reset 過，只會顯示 Decode 段的數字)")

    timing = {
        "Stream TTFT": round(t_first_token - t_start, 3),
        "Vision Encode": round(t_vision_end - t_vision_start, 3),
        "Injection": round(t_injection_end - t_vision_end, 3),
        "First Token Forward": round(t_first_token - t_injection_end, 3),
    }
    peak_mem_gb = {
        "Prefill": round(_mem_phases["Prefill"]["peak"], 2),
        "Vision Encode": round(_mem_phases["Vision Encode"]["peak"], 2),
        "Injection": round(_mem_phases["Injection"]["peak"], 2),
        "Decode": round(_mem_phases["Decode"]["peak"], 2),
        "OVERALL": round(_overall_peak, 2),
    }
    timing["_peak_mem_gb"] = peak_mem_gb

    return adapter.clean_answer(questions, answers), all_stats, timing


# ══════════════════════════════════════════════════════════════════════════════
# Main        （from llava_ov_svm.py）
# ══════════════════════════════════════════════════════════════════════════════
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
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    # ── 1. 載入模型 ──
    processor, model = build_model(args.model_name, dtype=dtype)
    print(f"num_image_token per patch : {num_image_tokens_per_patch(model)}")
    print(f"vision_feature_layer      : {model.config.vision_feature_layer}")
    print(f"vision_feature_select     : {model.config.vision_feature_select_strategy}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")

    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    adapter = LlavaOVAdapter(processor)

    # ── 2. 載入圖像與問題 ──
    if args.use_ds:
        from process_common import load_hf_dataset
        # datasets = load_hf_dataset(dataset="MMMU/MMMU", subject="Agriculture", split="test", num_image=args.num_images)
        # datasets = load_hf_dataset(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", num_image=args.num_images)
        datasets = load_hf_dataset(dataset="lmms-lab-encoder/MMVet", subject=None, num_image=args.num_images)
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

    tag = f"budget={args.budget}_{args.merge_mode}_{args.score_fn}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        elapsed = time.time() - t0
        print(f"\nAll done. Total time: {elapsed:.2f} s")
        return

    # ── 4. 逐批處理 ──
    for i in range(0, len(datasets), args.batch_size):
        batch_datasets = datasets[i:i + args.batch_size]

        print(f"\n{'='*70}")
        print(f"[{i+1}~{i+len(batch_datasets)}/{len(datasets)}] Processing Batch ...")
        print(f"{'='*70}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        try:
            if args.run_standard and len(output_results["references"]) < len(datasets):
                print(f"\n[Standard] Running official (uncompressed) generate() path ...")
                with measure_peak_memory("llava_ov_standard_generate"):
                    ref_answers = adapter.generate_baseline(model, batch_datasets)
                    output_results["references"] += ref_answers

            if args.run_stream:
                if tag not in output_results["candidates"]: output_results["candidates"][tag] = []

                print(f"\n[Online Stream] Running online kv stream with memory bank ...")
                questions = [dataset["question"] for dataset in batch_datasets]

                pixel_values_list, image_sizes_list = load_image_patches(processor, batch_datasets)
                print(f"  patches per image = {[pv.shape[0] for pv in pixel_values_list]}")

                with measure_peak_memory("llava_ov_online_kv_memory_bank"):
                    answers, all_stats, timing = run_online_kv_with_memory_bank(
                        model, adapter, pixel_values_list, questions,
                        image_sizes_list=image_sizes_list,
                        dtype=dtype,
                        vit_batch=args.vit_batch,
                        chunk_size=args.chunk_size,
                        budget=args.budget,
                        merge_mode=args.merge_mode,
                        score_fn=args.score_fn,
                        max_new_tokens=MAX_NEW_TOKENS,
                    )
                    output_results["candidates"][tag] += answers

                print("\n[Online KV Answer]")
                for item, tt in timing.items():
                    if item.startswith("_"):   # e.g. _peak_mem_gb（記憶體 dict，不是秒數）
                        continue
                    print(f"  {item}: {tt} sec")

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


    torch.cuda.synchronize(DEVICE)
    elapsed = time.time() - t0
    print(f"\nAll done. Total time: {elapsed:.2f} s")

    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
