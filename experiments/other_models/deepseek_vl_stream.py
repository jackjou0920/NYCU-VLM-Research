# ══════════════════════════════════════════════════════════════════════════════
# DeepSeek-VL2 串流 KV + Memory Bank —— 單檔獨立版
#
# 把 internvl_stream.py 裡 run_online_kv_with_memory_bank 的整套架構（bounded tile
# candidate pool -> 全域 top-K -> 延遲 commit -> chunked KV prefill）原封不動搬到
# DeepSeek-VL2 上。以下三個部分是「照抄」，因為它們本來就跟模型架構無關，純粹是
# 對 [N, D] token 張量做運算：
#     DEVICE 選擇 / mean_pool_question_embed / strip_to_question / measure_peak_memory
#     scoring：score_l2_norm / score_random / score_information_density / GlobalNormStats
#     memory bank：TileStreamingMemoryBank / SpatialBinTileMemoryBank
#
# 跟 InternVL 版本的三個關鍵差異（都在下面對應位置用註解標出）：
#
#   1. thumbnail(global view) tile 在 DeepSeek-VL2 是「第 0 顆」tile（InternVL 是
#      dynamic_preprocess 放在「最後一顆」）。protected_indices 因此改成 {0}。
#
#   2. 官方 prepare_inputs_embeds() 把同一「tile row」裡所有 local tile 的 patch
#      在空間上先拼成一張大 feature map，每個「大圖 row」才補一個 image_newline，
#      也就是說一顆 tile 的 h*w 個 token 在最終序列裡是被拆成 h 段、跟同 row 其他
#      tile 交錯排列的 —— 不是連續的一塊。這跟 InternVL 每顆 tile 各自 pixel_shuffle
#      後直接串接（tile 之間互不干擾）根本不同。
#      Tile-level streaming / eviction 仍然只在「評分」階段把每顆 tile 當獨立單元
#      （見 encode_tile 產生的 raw [hw, D] token，不含 newline，才不會被常數
#      newline 向量污染分數）；但 commit 進 KV 前，assemble_local_tiles() 會把
#      「同一個 tile-row 裡倖存下來的 local tile」重新依原本欄位順序左右拼接、
#      每個 row 只補一次 newline，被淘汰的 tile 直接跳過不留空格 —— 這樣budget
#      夠大、完全不用淘汰時，排版跟官方 prepare_inputs_embeds() 逐 bit相同；
#      只有真的有 tile 被淘汰、row 內欄位數變少時，才會跟「未壓縮」的官方排版
#      不同，而這是 tile 級壓縮本來就無法避免的代價。
#      （v1 曾經讓每顆 local tile 各自獨立補 newline、不跟同 row 的其他 tile
#      合併；這在 budget 大到全部 tile 都保留的情況下實測會讓模型幾乎必定在
#      第一步就輸出 EOS——分布跟官方訓練時看過的排版差太多，MoE 路由被完全帶偏。
#      assemble_local_tiles() 這個版本修掉了這個問題。）
#
#   3. 因為 (2)，「進 memory bank 參與評分/競爭」的 token 數（raw，未加 newline）
#      跟「最終真正寫進 KV 的 token 數」（final，含 newline）不一樣：newline 向量
#      是跟內容無關的常數 embedding，混進 scoring 只會稀釋 novelty / 污染 signal
#      norm 分布，所以 scoring 一定要用 raw token；但 budget 這個使用者關心的量
#      指的是「最終真正進 KV 的 token 數」，所以 K（可留幾顆 tile）要用 final size
#      換算，再回頭用 raw size 建 memory bank（bank 內部 K = capacity // num_image_token
#      要跟這裡算出的 K 對上）。見 DeepseekVL2Adapter.stream_image_to_kv 開頭的換算。
#
# 沒有搬過來的部分（scope 內明確排除，不是遺漏）：
#   - stream_image_to_kv_oracle（answer-aware 診斷用 oracle selection）
#   - thumbnail_saliency_prior / thumb_attn（thumbnail attention prior 混合）
#   - encode_and_bank（selection-only 診斷路徑，主流程不會呼叫）
#   - main() 的 HR-Bench / MMMU 資料集載入與 CLI（internvl_stream.py 那份依賴
#     process_common.py / internvl_preprocess.py，是 InternVL 實驗專屬的外部檔案，
#     跟這支檔案要展示的「DeepSeek-VL2 版 run_online_kv_with_memory_bank」無關）
#
# 架構事實依據：直接抓自 deepseek-ai/DeepSeek-VL2 官方 repo 原始碼（
#   deepseek_vl2/models/modeling_deepseek_vl_v2.py / siglip_vit.py /
#   processing_deepseek_vl_v2.py / modeling_deepseek.py）與
#   deepseek-ai/deepseek-vl2-small 的 config.json，而非憑印象猜測。
# ══════════════════════════════════════════════════════════════════════════════
from __future__ import annotations

import gc
import re
import time
import math
import contextlib

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM
from deepseek_vl2.utils.io import load_pil_images


# ══════════════════════════════════════════════════════════════════════════════
# 裝置挑選 / 記憶體量測 / question embedding      —— 照搬 internvl_stream.py，跟模型無關
# ══════════════════════════════════════════════════════════════════════════════
def get_optimal_cuda_device(min_required_gb: float = 0) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    best_device_idx = 0
    max_free_memory = 0
    for i in range(torch.cuda.device_count()):
        free_bytes, _ = torch.cuda.mem_get_info(i)
        if free_bytes > max_free_memory:
            max_free_memory = free_bytes
            best_device_idx = i

    max_free_gb = max_free_memory / (1024**3)
    if min_required_gb > 0 and max_free_gb < min_required_gb:
        print(f"警告：顯存最多的 GPU (cuda:{best_device_idx}) 僅剩 {max_free_gb:.2f} GB，未達要求的 {min_required_gb} GB。")

    return torch.device(f"cuda:{best_device_idx}")


DEVICE = get_optimal_cuda_device(min_required_gb=20.0)


def mean_pool_question_embed(get_input_embeddings_fn, tokenizer, questions, dtype):
    """tokenizer -> embedding -> mask 加权平均，跟 InternVL 版本完全相同。"""
    with torch.no_grad():
        q_tok = tokenizer(questions, return_tensors="pt", padding=True).to(DEVICE)
        q_embeds_all = get_input_embeddings_fn(q_tok.input_ids)
        q_mask = q_tok.attention_mask.unsqueeze(-1).float()
        pooled = (q_embeds_all * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-6)
    return pooled.to(dtype)


def strip_to_question(text: str) -> str:
    """只取問句本身，砍掉選項行與作答指示行，見 internvl_stream.py 同名函式的說明。"""
    out = []
    for ln in text.splitlines():
        s = ln.strip()
        if re.match(r"^[A-Za-z][.)]\s", s):
            break
        if re.match(r"(?i)^answer\b.*\b(option|options|letter|single word|phrase)\b", s):
            break
        out.append(ln)
    q = "\n".join(out).strip()
    return q or text.strip()


@contextlib.contextmanager
def measure_peak_memory(tag: str, record_timeline: bool = False):
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
# DeepSeek-VL2 專屬低階函式
# ══════════════════════════════════════════════════════════════════════════════
MAX_NEW_TOKENS = 500
SIGNAL_TRIM = 0.02     # 跟 internvl_stream.py 一致：signal 項與全域 norm 統計量，永遠 winsorize 掉 norm 最高的 2% token
COVERAGE_FLOOR = 1     # flush 的 top-K 永遠保留 1 個 farthest-point 名額，其餘按分數
# 跟官方 inference.py 用的值一樣；baseline（generate_answer_standard）跟串流路徑
# （run_online_kv_with_memory_bank）共用同一個預設值，兩邊比較才公平——貪婪解碼
# 在 logits 分佈很平的地方容易卡進重複迴圈，見這次除錯的討論。設 1.0 等於關掉。
DEFAULT_REPETITION_PENALTY = 1.1


def _kv_len(past_key_values) -> int:
    """DeepseekV2Model.forward() 吃/吐的是 legacy tuple-of-tuples 格式的 KV cache
    （不是 transformers 現代的 Cache 物件，見 modeling_deepseek.py 裡
    `DynamicCache.from_legacy_cache(...)` 進、`.to_legacy_cache()` 出），
    所以沒有 `.get_seq_length()` 可用，長度直接看第一層 key 的 seq 維度。"""
    if past_key_values is None:
        return 0
    return past_key_values[0][0].shape[2]


def _apply_repetition_penalty(logits: torch.Tensor, seen_ids: set[int], penalty: float) -> torch.Tensor:
    """跟 transformers 內建 RepetitionPenaltyLogitsProcessor 同一套公式：
    logit > 0 除以 penalty、logit <= 0 乘以 penalty，兩種情況都是把「已經出現過
    的 token」的機率往下壓。penalty <= 1.0（含 None）視為關閉，原樣回傳。"""
    if not penalty or penalty <= 1.0 or not seen_ids:
        return logits
    ids = torch.tensor(sorted(seen_ids), device=logits.device, dtype=torch.long)
    scores = logits.index_select(-1, ids)
    scores = torch.where(scores < 0, scores * penalty, scores / penalty)
    return logits.index_copy(-1, ids, scores)


def clean_answer(tokenizer, answers: list[str]) -> list[str]:
    """跟 internvl_stream.py 的 clean_answer 同樣目的（砍掉手動 decode 累積出來的結尾
    eos marker 文字），但用 tokenizer.eos_token 動態取代 InternVL 版本寫死的
    "<|im_end|>"（DeepSeek 的 conv template 停止字串是 "<｜end▁of▁sentence｜>"）。"""
    eos_text = tokenizer.eos_token or ""
    for i, answer in enumerate(answers):
        if eos_text:
            answer = answer.replace(eos_text, "")
        answers[i] = answer.strip()
    return answers


# ──────────────────────────────────────────────────────────────────────────────
# 標準 generate 參考路徑（用於正確性比較）——修正官方 README 範例常見的手誤：
# DeepseekVLV2ForCausalLM 的語言模型屬性是 `.language`，不是 `.language_model`。
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer_standard(model, processor, pil_images, questions, generation_config=None):
    """走官方 README「Simple Inference Example with Multiple Images」的
    prepare_inputs_embeds() + language.generate() 路徑，泛化成 batch。

    DeepSeek-VL2 的 processor 沒有 InternVL batch_chat() 那種「一次吃 batch」
    介面 —— __call__() 一次只吃「一個 conversation」（可以內含多張圖）。要真的
    batch 多個獨立樣本，得逐一樣本呼叫 process_one() 建樣本、再用
    processor.batchify() 併成一個 batch（batchify 內部做 left-padding，等價於
    batch_chat() 的角色；也是 processor.__call__() 內部自己的做法，這裡只是
    把它攤開成一次處理一整個 batch 而不是一個樣本）。

    pil_images[i] 可以是：
      * 一張 PIL.Image                —— 單圖樣本（向下相容舊呼叫）
      * 一個 list[PIL.Image]          —— 多圖樣本，對應 README 的
        interleaved `<image>` 用法（例如 "This is image_1: <image>\\n
        This is image_2: <image>\\n ..."）

    questions[i] 是這個樣本的 content 字串：多圖時必須自己在字串裡放好對應
    數量的 `<image>` placeholder（跟 README 範例一致）；單圖且完全沒放
    placeholder 時，才自動補一個 "<image>\\n" 前綴，維持舊呼叫方式不用改。
    """
    assert len(pil_images) == len(questions)
    if generation_config is None:
        # repetition_penalty 跟 run_online_kv_with_memory_bank 的手動 decode 迴圈用同一個
        # DEFAULT_REPETITION_PENALTY，兩條路徑比較才公平（HF generate() 原生支援這個 kwarg，
        # 語意跟我在串流路徑手刻的 _apply_repetition_penalty 完全一樣）。
        generation_config = dict(
            max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            repetition_penalty=DEFAULT_REPETITION_PENALTY,
        )

    tokenizer = processor.tokenizer
    samples = []
    for images, question in zip(pil_images, questions):
        if not isinstance(images, (list, tuple)):
            images = [images]
        images = list(images)

        content = question
        n_placeholders = content.count("<image>")
        if n_placeholders == 0:
            if len(images) != 1:
                raise ValueError(
                    f"question has no <image> placeholder but {len(images)} images were "
                    f"given; a multi-image sample must embed one <image> per image in the "
                    f"question text, e.g. 'This is image_1: <image>\\nThis is image_2: <image>\\n...'"
                )
            content = f"<image>\n{content}"
        elif n_placeholders != len(images):
            raise ValueError(
                f"question has {n_placeholders} <image> placeholders but {len(images)} "
                f"images were given"
            )

        conversation = [
            {"role": "<|User|>", "content": content, "images": images},
            {"role": "<|Assistant|>", "content": ""},
        ]
        samples.append(processor.process_one(conversations=conversation, images=images, system_prompt=""))

    batched = processor.batchify(samples).to(DEVICE, dtype=model.dtype)
    inputs_embeds = model.prepare_inputs_embeds(
        input_ids=batched.input_ids,
        images=batched.images,
        images_seq_mask=batched.images_seq_mask,
        images_spatial_crop=batched.images_spatial_crop,
    )
    outputs = model.language.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=batched.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=True,
        **generation_config,
    )
    return [tokenizer.decode(o.cpu().tolist(), skip_special_tokens=True) for o in outputs]


def measure_baseline_ttft(model, processor, pil_images, questions, device):
    """跟 generate_answer_standard 走完全相同的路徑，只把 max_new_tokens 砍到 1。"""
    torch.cuda.synchronize(device)
    t0 = time.time()
    generate_answer_standard(
        model, processor, pil_images, questions,
        generation_config=dict(max_new_tokens=1, do_sample=False),
    )
    torch.cuda.synchronize(device)
    return time.time() - t0


# ──────────────────────────────────────────────────────────────────────────────
# 串流核心：單一 tile 的 SigLIP -> downsample_mlp_gelu projector
# ──────────────────────────────────────────────────────────────────────────────
def encode_tile(model, tile_pixel_values: torch.Tensor, dtype) -> torch.Tensor:
    """
    輸入 : tile_pixel_values [B_tile, 3, 384, 384]
    輸出 : [B_tile, hw, D_llm]  (SigLIP 沒有 CLS token —— class_token=False，
           ignore_head=True，vision(x) 直接吐 patch token，不用像 InternVL
           那樣手動切掉第 0 個 —— 已經過 projector，跟 LLM embedding space
           同維度，但「尚未」補 row newline，見 format_tile_block)

    刻意保持「無狀態、單一 tile-batch 進、單一 tile-batch 出」，跟
    internvl_stream.py 的 encode_tile 對稱：ViT + projector 是這裡的
    vision_model + pixel_shuffle + mlp1。
    """
    with torch.no_grad():
        feats = model.vision(tile_pixel_values)      # [B_tile, vit_seq_len, C_vit]
        tile_tokens = model.projector(feats)          # [B_tile, hw, D_llm]
        del feats
    return tile_tokens.to(dtype)


@torch.no_grad()
def _append_row_newline(model, feat_hWD: torch.Tensor) -> torch.Tensor:
    """[h, W, D] -> [h*(W+1), D]：每一列後面補一個 model.image_newline
    （跟官方 modeling_deepseek_vl_v2.py 裡 `torch.cat([features, new_lines], dim=1)`
    的邏輯一致；W 可以是單一 tile 的 w，也可以是好幾顆 tile 併排後的 w*k）。"""
    h, W, D = feat_hWD.shape
    newline = model.image_newline.detach().to(device=feat_hWD.device, dtype=feat_hWD.dtype).view(1, 1, D).expand(h, 1, D)
    feat = torch.cat([feat_hWD, newline], dim=1)      # [h, W+1, D]
    return feat.reshape(-1, D).contiguous()


@torch.no_grad()
def format_tile_block(model, tile_tokens_2d: torch.Tensor) -> torch.Tensor:
    """單顆 tile 的 [hw, D] -> [h*(w+1), D]。只用在 global(thumbnail) view ——
    官方格式化裡 global view 本來就是「自己獨立」補一次 newline，不會跟其他
    tile 合併，所以這裡逐 tile 獨立處理跟官方完全一致。local tile 不要用這個
    函式，要用下面的 assemble_local_tiles（因為官方 local tile 是跨 tile 合併
    之後才補 newline，見該函式的說明）。"""
    hw, D = tile_tokens_2d.shape
    h = w = int(round(hw ** 0.5))
    assert h * w == hw, f"expected a square token grid, got hw={hw}"
    return _append_row_newline(model, tile_tokens_2d.view(h, w, D))


@torch.no_grad()
def assemble_local_tiles(model, entries: list[tuple[int, torch.Tensor]], num_width_tiles: int) -> torch.Tensor:
    """把「倖存的 local tile」依官方排版規則重新合併，取代 v1 那個「每顆 tile
    各自獨立補 newline」的權宜設計（那個設計會讓模型在幾乎沒壓縮的情況下也生不
    出正常答案，實測會讓第一個生成 token 直接是 EOS —— 見這次除錯的 [DEBUG]
    log：`first-token top5` 最高分就是 `<｜end▁of▁sentence｜>`）。

    做法完全對應 modeling_deepseek_vl_v2.py 的
    `rearrange('(th tw)(h w)d -> (th h)(tw w)d')`：同一個 tile-row 裡所有「倖存」
    的 tile 依原本的欄位順序左右拼成一張大 feature map，那一整個 tile-row 只補
    一次 newline；被淘汰的 tile 直接跳過、不留空格（欄位會往左靠攏）。

    因此：當這個 row 裡的 tile 全部倖存（例如 budget 大到完全不用淘汰）時，
    這裡產生的排版跟官方 prepare_inputs_embeds() 逐 bit 相同；只有在真的有 tile
    被淘汰、同一 row 欄位數變少時，才會跟官方（沒有壓縮過的）排版不同——這是
    tile-level 壓縮本來就無法避免的代價，不是額外的排版誤差。

    entries: [(tile_index, raw_tokens[hw, D]), ...]，tile_index 從 1 起算
             （0 是 global view，不會、也不應該出現在這裡，呼叫端要先濾掉）。
    回傳 None 表示沒有任何 local tile 倖存（例如 budget 小到只夠留 global view）。
    """
    if not entries:
        return None
    hw, D = entries[0][1].shape
    h = w = int(round(hw ** 0.5))
    assert h * w == hw, f"expected a square token grid, got hw={hw}"

    rows: dict[int, list[tuple[int, torch.Tensor]]] = {}
    for tile_index, tokens in entries:
        local_idx = tile_index - 1
        row, col = divmod(local_idx, num_width_tiles)
        rows.setdefault(row, []).append((col, tokens.view(h, w, D)))

    row_blocks = []
    for row in sorted(rows.keys()):
        cols_sorted = [feat for _, feat in sorted(rows[row], key=lambda x: x[0])]
        row_feat = torch.cat(cols_sorted, dim=1)              # [h, w*k, D]，k = 這個 row 倖存的 tile 數
        row_blocks.append(_append_row_newline(model, row_feat))  # [h*(w*k+1), D]

    return torch.cat(row_blocks, dim=0)


# ══════════════════════════════════════════════════════════════════════════════
# Importance scoring + Tile Streaming Memory Bank
# —— 跟 internvl_stream.py 逐行相同：純 [N, D] token 張量運算，不碰任何模型特定 API。
# ══════════════════════════════════════════════════════════════════════════════
def _winsorize_high(v: torch.Tensor, frac: float) -> torch.Tensor:
    if frac is None or frac <= 0.0 or v.numel() < 8:
        return v
    cap = torch.quantile(v.float(), 1.0 - float(frac))
    return v.clamp(max=cap.to(v.dtype))


def score_l2_norm(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    return tokens.float().norm(dim=-1)


def score_random(tokens: torch.Tensor, question_embed=None, **_) -> torch.Tensor:
    return torch.rand(tokens.shape[0], device=tokens.device)


def score_information_density(
    tokens, question_embed,
    alpha=0.3, beta=0.3, gamma=0.4,
    knn_k=3,
    norm_stats: dict | None = None,
    signal_trim: float = SIGNAL_TRIM,
):
    """Question-aware Information Density Score，跟 InternVL 版本公式完全相同：
    Score = alpha*S + beta*N + gamma*R，三項各自正規化到 [0,1] 後再加權。"""
    x = tokens.float()
    N = x.shape[0]

    raw_norm = _winsorize_high(x.norm(dim=-1), signal_trim)
    if norm_stats is not None:
        z = (raw_norm - norm_stats["mean"]) / norm_stats["std"]
        signal = torch.sigmoid(z)
    elif N > 1:
        signal = raw_norm / (raw_norm.max() + 1e-9)
    else:
        signal = torch.zeros(N, device=x.device)

    x_norm = F.normalize(x, dim=-1)
    sim = x_norm @ x_norm.T
    sim.fill_diagonal_(-1.0)
    k = min(knn_k, max(N - 1, 1))
    topk_sim = sim.topk(k, dim=-1).values.mean(dim=-1) if N > 1 else torch.zeros(N, device=x.device)
    novelty = ((1 - topk_sim) / 2).clamp(0.0, 1.0)

    if question_embed is None or isinstance(question_embed, str):
        relevance = torch.zeros(N, device=x.device)
        gamma_eff = 0.0
    else:
        qn = F.normalize(question_embed.float().reshape(-1), dim=-1)
        relevance = ((x_norm @ qn) + 1) / 2
        gamma_eff = gamma

    total = alpha + beta + gamma_eff
    a, b, g = alpha / total, beta / total, gamma_eff / total
    return a * signal + b * novelty + g * relevance


SCORE_FUNCS = {
    "l2_norm": score_l2_norm,
    "info_density": score_information_density,
    "random": score_random,
}


class GlobalNormStats:
    """Running Welford mean/std of the per-token L2 norm — 跟 internvl_stream.py 相同。"""

    def __init__(self):
        self._n = 0
        self._mean = 0.0
        self._M2 = 0.0

    @torch.no_grad()
    def update(self, tile_tokens):
        v = _winsorize_high(tile_tokens.float().norm(dim=-1), SIGNAL_TRIM)
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
    if score_fn_name == "l2_norm":
        scores = score_l2_norm(tile_tokens, question_embed)
    elif score_fn_name == "random":
        scores = score_random(tile_tokens, question_embed)
    elif score_fn_name == "info_density":
        scores = score_information_density(
            tile_tokens, question_embed, alpha=0.3, beta=0.3, gamma=0.4, norm_stats=norm_stats,
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
    __slots__ = ("tokens", "tile_index", "score", "arrival")
    def __init__(self, tokens, tile_index, score, arrival):
        self.tokens = tokens
        self.tile_index = tile_index
        self.score = float(score)
        self.arrival = int(arrival)


class TileStreamingMemoryBank:
    """Bounded candidate pool + global top-K tile selection (lazy commit)。
    邏輯跟 internvl_stream.py 的同名類別完全一致 —— 见該檔案的 docstring。"""

    def __init__(self, capacity, num_image_token, device=None, dtype=None,
                 score_fn="info_density", question_embed=None, tile_agg="max",
                 delay_tiles=2, protected_indices=()):
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
        self.rescore = True
        self.protected_indices = set(int(i) for i in protected_indices)
        self.pool = []
        self._norm = GlobalNormStats()
        self.total_seen = 0
        self.total_evicted = 0
        self.total_committed = 0
        self.committed_indices = []
        self.all_scores = []
        self.max_resident_tiles = 0
        self.size_history = []

    @torch.no_grad()
    def _score_tile(self, tile_tokens):
        return score_tile_tokens(
            tile_tokens, self.score_fn_name, self.question_embed,
            self.tile_agg, self._norm.as_dict(self.device),
        )

    def _protected(self, tile_index):
        return tile_index in self.protected_indices

    @torch.no_grad()
    def _rescore_pool(self):
        for e in self.pool:
            e.score = self._score_tile(e.tokens)

    @torch.no_grad()
    def _evict_to_capacity(self):
        overflow = len(self.pool) - self.pool_capacity
        if overflow <= 0:
            return
        cand = [e for e in self.pool if not self._protected(e.tile_index)]
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
        score = self._score_tile(tile_tokens)
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
        return []

    @torch.no_grad()
    def flush(self):
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
            kept = kept + rest[: keep_n - cf]
            if cf > 0:
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
    """Coverage-guaranteed tile selection —— 邏輯跟 internvl_stream.py 完全一致，
    唯一差別是 DeepSeek-VL2 的 tile 順序天生就是「processor 已經算好」的
    raster order（見 images_spatial_crop），不需要像 InternVL 那樣從 tile 數
    反推 (cols, rows)。"""

    def __init__(self, capacity, num_image_token, device=None, dtype=None,
                 score_fn="info_density", question_embed=None, tile_agg="max",
                 protected_indices=(), num_tiles=1):
        if capacity < num_image_token:
            raise ValueError(f"budget={capacity} < num_image_token={num_image_token}")
        self.num_image_token = num_image_token
        self.K = capacity // num_image_token
        self.D = 0
        self.device, self.dtype = device, dtype
        self.score_fn_name = score_fn
        self.question_embed = question_embed
        self.tile_agg = tile_agg
        self.rescore = True
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
        self.all_scores = []
        self.max_resident_tiles = 0
        self.size_history = []

    def _protected(self, tile_index):
        return tile_index in self.protected_indices

    @property
    def pool(self):
        return list(self.protected_entries.values()) + [c for c in self.champion if c is not None]

    @torch.no_grad()
    def add_tile(self, tile_tokens, tile_index):
        if tile_tokens.ndim != 2 or tile_tokens.shape[0] != self.num_image_token:
            raise ValueError(
                f"Expected [{self.num_image_token}, D], got {tuple(tile_tokens.shape)}"
            )
        self._norm.update(tile_tokens)
        score = score_tile_tokens(
            tile_tokens, self.score_fn_name, self.question_embed,
            self.tile_agg, self._norm.as_dict(self.device),
        )
        self.all_scores.append(score)
        entry = TileEntry(tile_tokens.detach().clone(), tile_index, score, self.total_seen)
        self.total_seen += 1

        if self._protected(tile_index):
            self.protected_entries[tile_index] = entry
        else:
            b = self.bin_of.get(tile_index)
            current = self.champion[b] if b is not None else None
            if current is not None:
                current.score = score_tile_tokens(
                    current.tokens, self.score_fn_name, self.question_embed,
                    self.tile_agg, self._norm.as_dict(self.device),
                )
            if b is None:
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
        return []

    @torch.no_grad()
    def flush(self):
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


# ══════════════════════════════════════════════════════════════════════════════
# DeepseekVL2Adapter
# ══════════════════════════════════════════════════════════════════════════════
class DeepseekVL2Adapter:
    """DeepSeek-VL2 版的 InternVLAdapter。跟 InternVL 最大的結構差異：

    DeepSeek-VL2 的 processor 把「動態切 tile」（select_best_resolution + crop）
    跟「組 chat template + tokenize」綁在同一次呼叫裡（tokenize_with_images），
    不像 InternVL 分成 load_image_tiles（外部檔案）+ build_prompt 兩支函式，
    所以這裡不需要、也沒辦法把兩者分開；budget 也不需要在建構 adapter 時就知道
    （不像 InternVL 的 num_image_token * equiv_tiles 需要用 budget 換算 placeholder
    數量）——因為 text_before/text_after 是直接照 images_seq_mask 的位置切
    input_ids，跟 placeholder 實際有幾個 token 無關（反正 streaming 路徑不會用到
    這些 placeholder token，只是拿它們定位圖片該插入的位置）。
    """

    def __init__(self, processor: DeepseekVLV2Processor, model: DeepseekVLV2ForCausalLM):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.model = model
        self.hidden_size = model.language.get_input_embeddings().weight.shape[-1]

        # 跟官方 tokenize_with_images() 算 placeholder 數量用的公式完全相同：
        #   h = w = ceil((image_size // patch_size) / downsample_ratio)
        grid = processor.image_size // processor.patch_size
        h = w = math.ceil(grid / processor.downsample_ratio)
        self.tile_side = h
        self.raw_tile_tokens = h * w            # 進 memory bank 評分/競爭用的大小（無 newline）
        self.final_tile_tokens = h * (w + 1)    # 真正寫進 KV 的大小（含每列一個 newline）

    def get_tokenizer(self):
        return self.tokenizer

    def get_input_embeddings(self, model):
        return model.language.get_input_embeddings()

    def lm_prefill(self, model, **kwargs):
        """prefill 只需要 past_key_values，不需要 logits，直接呼叫
        DeepseekV2Model（跳過 lm_head），對稱 InternVLAdapter.lm_prefill
        呼叫 `model.language_model.model` 的做法。"""
        base = model.language.model
        with torch.no_grad():
            return base(**kwargs)

    def lm_decode_step(self, model, **kwargs):
        with torch.no_grad():
            out = model.language(**kwargs)
        return out.logits[:, -1, :], out.past_key_values

    def prepare_sample(self, question: str, pil_image):
        """一次 processor 呼叫同時做完 dynamic tiling 與 prompt tokenize，取代
        InternVL 版本的 load_image_tiles + build_prompt。回傳：
            ids_before, ids_after : 1D LongTensor（<image> placeholder span 前後的文字 token）
            pixel_values          : [1+num_local_tiles, 3, 384, 384]，第 0 顆是 global view
            num_width_tiles       : local tile grid 的欄數（給 assemble_local_tiles 分 row 用）
        """
        conversation = [
            {"role": "<|User|>", "content": f"<image>\n{question}", "images": [pil_image]},
            {"role": "<|Assistant|>", "content": ""},
        ]
        prepare = self.processor(
            conversations=conversation, images=[pil_image],
            force_batchify=True, system_prompt="",
        )
        input_ids = prepare.input_ids[0]
        seq_mask = prepare.images_seq_mask[0]
        img_pos = seq_mask.nonzero(as_tuple=True)[0]
        if img_pos.numel() == 0:
            raise ValueError("processor output has no <image> placeholder span")
        i0, i1 = int(img_pos[0]), int(img_pos[-1]) + 1
        ids_before = input_ids[:i0].clone()
        ids_after = input_ids[i1:].clone()

        num_width_tiles, num_height_tiles = (int(x) for x in prepare.images_spatial_crop[0, 0])
        n_local = num_width_tiles * num_height_tiles
        pixel_values = prepare.images[0][: 1 + n_local].clone()
        assert pixel_values.shape[0] == 1 + n_local

        return ids_before, ids_after, pixel_values, num_width_tiles

    @torch.no_grad()
    def stream_image_to_kv(
        self,
        model,
        pixel_values,
        num_width_tiles,
        question_embed,
        budget,
        score_fn="info_density",
        vit_batch=4,
        chunk_size=1024,
        dtype=torch.bfloat16,
        delay_tiles=2,
        tile_agg="max",
        select="topk",
        past_key_values=None,
        running_mask=None,
    ):
        """True online Vision -> bounded candidate pool -> LLM KV construction。
        結構跟 InternVLAdapter.stream_image_to_kv 完全對應（Phase 1 串流評分、
        Phase 2 flush 後 chunked prefill），差異只在於：

          * budget 用「最終 tile 大小」(final_tile_tokens) 換算可留幾顆 tile
            (K_target)，再回頭用「原始 tile 大小」(raw_tile_tokens) 建 memory bank
            —— 見檔案開頭差異 3 的說明。K_target 是保守估計的上界：因為 commit
            時同一 row 倖存的 local tile 會共用一個 newline（見 assemble_local_tiles），
            實際灌進 KV 的 token 數只會比 K_target * final_tile_tokens 少，不會超過。
          * flush() 拿到的是 raw token；commit 前 global tile 用 format_tile_block()
            獨立補 newline，local tile 用 assemble_local_tiles() 依「倖存 tile 所在
            的 row」重新合併（跟官方排版一致，見該函式說明），中間插入 view_seperator
            （結構性 markup，不算進 budget，比照 InternVL 把 IMG_START/END 放在
            budget 外的做法）。
        """
        num_tiles = pixel_values.shape[0]
        if past_key_values is None:
            raise ValueError("past_key_values must be initialized by text_before prefill")
        if running_mask is None:
            raise ValueError("running_mask must be initialized by text_before prefill")
        if select not in ("topk", "spatial_bin"):
            raise ValueError(f"Unknown select={select!r}; expected 'topk' or 'spatial_bin'")

        raw_n, final_n = self.raw_tile_tokens, self.final_tile_tokens
        K_target = max(1, budget // final_n)
        bank_capacity = K_target * raw_n   # 讓 bank 內部 K = bank_capacity // raw_n 剛好等於 K_target

        # global(thumbnail) view 是 DeepSeek-VL2 tile 序列裡的「第 0 顆」（InternVL 是最後一顆）
        protected = {0}
        if select == "topk":
            memory = TileStreamingMemoryBank(
                capacity=bank_capacity, num_image_token=raw_n, device=DEVICE, dtype=dtype,
                score_fn=score_fn, question_embed=question_embed, tile_agg=tile_agg,
                delay_tiles=delay_tiles, protected_indices=protected,
            )
        else:
            memory = SpatialBinTileMemoryBank(
                capacity=bank_capacity, num_image_token=raw_n, device=DEVICE, dtype=dtype,
                score_fn=score_fn, question_embed=question_embed, tile_agg=tile_agg,
                protected_indices=protected, num_tiles=num_tiles,
            )

        tokens_injected = 0
        kv_flushes = 0

        def prefill_vision(vtok_2d):
            nonlocal past_key_values, running_mask, tokens_injected, kv_flushes
            vchunk = vtok_2d.unsqueeze(0)  # [1, c, D]
            c = vchunk.shape[1]
            past_seq_len = _kv_len(past_key_values)
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

        # ---- Phase 1: stream every tile through the bounded pool (raw content tokens, no KV writes) ----
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

        # ---- Phase 2: final top-K selection -> official-style row-merge decoration -> chunked KV prefill ----
        selected = memory.flush()  # raw [raw_n, D] blocks, sorted by tile_index; index 0 (global) always first
        committed_indices = memory.committed_indices          # same order as `selected`
        entries = list(zip(committed_indices, selected))
        del selected

        global_entry = next((t for idx, t in entries if idx == 0), None)
        local_entries = [(idx, t) for idx, t in entries if idx != 0]
        del entries

        parts = []
        if global_entry is not None:
            parts.append(format_tile_block(model, global_entry))
        if local_entries:
            sep = model.view_seperator.detach().to(device=DEVICE, dtype=dtype).view(1, -1)
            parts.append(sep)
            parts.append(assemble_local_tiles(model, local_entries, num_width_tiles))
        full_seq = torch.cat(parts, dim=0) if parts else torch.empty(0, self.hidden_size, device=DEVICE, dtype=dtype)
        del parts, global_entry, local_entries

        for s in range(0, full_seq.shape[0], chunk_size):
            prefill_vision(full_seq[s: s + chunk_size])
        del full_seq

        stats = memory.stats()
        stats.update({
            "selection_only": False,
            "num_tiles": num_tiles,
            "tokens_injected": tokens_injected,
            "kv_flushes": kv_flushes,
            "final_size": tokens_injected,
            "raw_tile_tokens": raw_n,
            "final_tile_tokens": final_n,
        })

        # +1：view_seperator 是結構性 markup，比照 InternVL 把 IMG_START/END 放在 budget 外的做法
        assert tokens_injected <= budget + 1, (
            f"KV visual tokens {tokens_injected} > budget {budget}+1"
        )
        return past_key_values, running_mask, stats

    def generate_baseline(self, model, **kwargs):
        pil_images = kwargs["pil_images"]
        questions = kwargs["questions"]
        return generate_answer_standard(model, self.processor, pil_images, questions)

    def clean_answer(self, batch_or_questions, answers):
        return clean_answer(self.tokenizer, answers)


# ══════════════════════════════════════════════════════════════════════════════
# 共用骨架：online KV streaming 主流程
# ══════════════════════════════════════════════════════════════════════════════
def run_online_kv_with_memory_bank(
    model,
    adapter: DeepseekVL2Adapter,
    pil_images: list,
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
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY,
):
    """True online Vision -> bounded tile memory -> LLM KV pipeline，DeepSeek-VL2 版。

    跟 internvl_stream.py 的 run_online_kv_with_memory_bank 結構逐段對應：
      Step A：text-before prefill
      Step B：真正的 online Vision -> TileMemory -> KV（stream_image_to_kv）
      Step C：text-after + 逐 token 手動 decode（legacy tuple KV cache，見 _kv_len）

    輸入是 `pil_images`（PIL.Image 列表）而非 InternVL 版本的 `pixel_values_list`
    ——因為 DeepSeek-VL2 的 dynamic tiling 是 processor 的一部分，沒有獨立於
    prompt-building 之外的「純 tiling」步驟可以在外部先做掉。

    ``budget`` 是最終寫進 KV 的視覺 token 上限（單位：DeepSeek-VL2 每顆 tile 含
    row newline 後的大小，見 DeepseekVL2Adapter.final_tile_tokens）。

    ``repetition_penalty``：手動逐 token decode（見下方 for 迴圈）是純
    ``torch.argmax`` 貪婪解碼，不像 HF ``generate()`` 內建
    ``RepetitionPenaltyLogitsProcessor``，沒有東西擋著模型陷入「the frame is
    in the frame is in the frame...」這種重複迴圈（貪婪解碼在 logits 分佈很平的
    地方本來就容易卡進重複 attractor，baseline 用同一張圖也已經有輕微的重複/
    語法錯亂跡象，只是它剛好沒有真的卡死；streaming 路徑因為是分段 chunked
    prefill、數值上跟一次到位的官方路徑不會 bit-for-bit 相同，容易把本來就在
    邊界上的模型推進迴圈）。這裡用跟 DeepSeek-VL2 官方 inference.py 一樣的
    repetition_penalty 公式（對「已經生成過的 token id」的 logit 做懲罰：
    正值除以 penalty、負值乘以 penalty），設 1.0 等於關掉、還原成純貪婪解碼。
    """
    B = len(pil_images)
    assert B == len(questions)

    if select not in ("topk", "spatial_bin"):
        raise ValueError(
            f"Unknown select={select!r}; expected 'topk' or 'spatial_bin' "
            f"('oracle' 沒有搬過來，見檔案開頭說明)"
        )
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

    question_embeds = mean_pool_question_embed(
        adapter.get_input_embeddings(model), tokenizer,
        [strip_to_question(q) for q in questions], dtype,
    )

    # ------------------------------------------------------------
    # Process each image independently, same rationale as internvl_stream.py:
    # avoids cross-image padding and keeps the O(K) invariant explicit.
    # ------------------------------------------------------------
    for b in range(B):
        pil_image = pil_images[b]
        question = questions[b]
        question_embed = question_embeds[b]

        ids_before, ids_after, pixel_values, num_width_tiles = adapter.prepare_sample(question, pil_image)

        print("\n" + "=" * 80)
        print(
            f"[Image {b}] tiles={pixel_values.shape[0]} "
            f"budget={budget} delay_tiles={delay_tiles} score={score_fn} tile_agg={tile_agg}"
        )
        print("=" * 80)
        # DEBUG（診斷用，之後可以刪）：確認 <image> placeholder 前後切出來的文字是否合理
        print(f"[DEBUG] ids_before ({ids_before.shape[0]} tok): "
              f"{tokenizer.decode(ids_before, skip_special_tokens=False)!r}")
        print(f"[DEBUG] ids_after  ({ids_after.shape[0]} tok): "
              f"{tokenizer.decode(ids_after, skip_special_tokens=False)!r}")

        # --------------------------------------------------------
        # Step A: text-before prefill
        # --------------------------------------------------------
        _mem_reset()
        ids_before = ids_before.unsqueeze(0).to(DEVICE)
        mask_before = torch.ones_like(ids_before)
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

        del out, embeds_before, ids_before
        _mem_mark("Prefill")
        print(f"[DEBUG] kv_len after text_before prefill = {_kv_len(past_key_values)} "
              f"(running_mask len = {running_mask.shape[1]})")

        # --------------------------------------------------------
        # Step B: true online Vision -> TileMemory -> KV
        # --------------------------------------------------------
        _mem_reset()
        torch.cuda.synchronize(DEVICE)
        t_vision_start = time.time()

        past_key_values, running_mask, stats = adapter.stream_image_to_kv(
            model=model,
            pixel_values=pixel_values.to(DEVICE, dtype=dtype),
            num_width_tiles=num_width_tiles,
            question_embed=question_embed,
            budget=budget,
            score_fn=score_fn,
            vit_batch=vit_batch,
            chunk_size=chunk_size,
            dtype=dtype,
            delay_tiles=delay_tiles,
            tile_agg=tile_agg,
            select=select,
            past_key_values=past_key_values,
            running_mask=running_mask,
        )

        torch.cuda.synchronize(DEVICE)
        t_vision_end = time.time()
        vision_times.append(t_vision_end - t_vision_start)

        _mem_mark("Vision+OnlineKV")
        _mem_reset()
        print(f"[DEBUG] kv_len after vision prefill = {_kv_len(past_key_values)} "
              f"(running_mask len = {running_mask.shape[1]})")

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
        ids_after = ids_after.unsqueeze(0).to(DEVICE)
        mask_after = torch.ones_like(ids_after)

        past_seq_len = _kv_len(past_key_values)
        running_mask = torch.cat([running_mask, mask_after], dim=1)
        position_ids = torch.arange(
            past_seq_len, past_seq_len + ids_after.shape[1], dtype=torch.long, device=DEVICE,
        ).unsqueeze(0)

        torch.cuda.synchronize(DEVICE)
        t_decode_start = time.time()

        seen_ids = set(ids_after[0].tolist())   # repetition_penalty 的懲罰對象，跟 HF generate() 預設一樣涵蓋 prompt tail + 之後生成的 token

        next_token_logits, past_key_values = adapter.lm_decode_step(
            model,
            input_ids=ids_after,
            attention_mask=running_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True,
        )
        next_token_logits = _apply_repetition_penalty(next_token_logits, seen_ids, repetition_penalty)
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        seen_ids.add(int(next_token[0, 0].item()))

        # DEBUG（診斷用，之後可以刪）：看第一個 decode step 到底最想生什麼
        top5 = torch.topk(torch.log_softmax(next_token_logits.float(), dim=-1)[0], k=5)
        print("[DEBUG] first-token top5:", [
            (tokenizer.decode([tid]), round(float(lp), 3))
            for tid, lp in zip(top5.indices.tolist(), top5.values.tolist())
        ])

        torch.cuda.synchronize(DEVICE)
        t_first_token = time.time()
        first_token_times.append(t_first_token - t_start)

        del ids_after, mask_after, position_ids, next_token_logits

        eos_token_id = tokenizer.eos_token_id
        answer = tokenizer.decode(next_token[0], skip_special_tokens=False)
        finished = bool(
            eos_token_id is not None
            and next_token[0, 0].item() == eos_token_id
        )
        print(f"[DEBUG] first token = {tokenizer.decode(next_token[0])!r}, finished={finished}")

        for step in range(max_new_tokens):
            if finished:
                break

            past_seq_len = _kv_len(past_key_values)
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
            next_token_logits = _apply_repetition_penalty(next_token_logits, seen_ids, repetition_penalty)
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            token_id = next_token[0, 0].item()
            seen_ids.add(token_id)
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
# Demo：跟官方 README 一樣的單圖 visual grounding 例子，baseline vs. 串流 KV 各跑一次
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    model_path = "deepseek-ai/deepseek-vl2-small"
    vl_chat_processor: DeepseekVLV2Processor = DeepseekVLV2Processor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: DeepseekVLV2ForCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).to(DEVICE).eval()

    question = "What is shown in this image in extreme detail? Answer in detail."
    conversation = [
        {
            "role": "<|User|>",
            "content": f"<image>\n{question}",
            "images": ["./img_datasets/img02.jpg"],
        },
        {"role": "<|Assistant|>", "content": ""},
    ]
    pil_images = load_pil_images(conversation)

    with measure_peak_memory("baseline"):
        baseline_answers = generate_answer_standard(vl_gpt, vl_chat_processor, pil_images, [question])
    print("\n[baseline] answer:", baseline_answers[0])

    adapter = DeepseekVL2Adapter(vl_chat_processor, vl_gpt)
    with measure_peak_memory("online_kv_memory_bank"):
        stream_answers, stream_stats, timing = run_online_kv_with_memory_bank(
            vl_gpt, adapter, pil_images, [question],
            budget=20000, score_fn="info_density", select="topk",
        )
    print("\n[online_kv_memory_bank] answer:", stream_answers[0])
    print("[online_kv_memory_bank] timing:", timing)
