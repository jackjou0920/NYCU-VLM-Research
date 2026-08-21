"""
共用的模型 adapter 介面。
 
設計原則：streaming_common.py 裡的 run_online_kv_with_memory_bank() 只跑「跟
模型無關」的骨架（text_before prefill → chunked vision injection → text_after
+ decode），任何需要碰 model 內部結構的地方，一律透過這裡定義的 adapter 方法，
不直接在骨架裡 if is_internvl / is_llava_ov。
 
刻意 **不** 統一的部分：encode_and_bank()。這是兩個模型架構本質不同的地方：
    - InternVL：tile 之間彼此獨立（無 cross-tile 依賴），可以逐 tile
      encode → add_tile，不需要等其他 tile 到齊 → 保留「完全 by-tile 線上串流」
      的架構可能性（tile 一來就能決定要不要留）。
    - LLaVA-OneVision：unpad + 雙線性內插是整張圖一起做的（跨 patch 依賴），
      必須先蒐集全部 patch，呼叫官方 model.model.pack_image_features() 拿到
      in-distribution 表示，才能在「列」的粒度上進一步用 bank 做淘汰。
若把這一步塞進同一個函式，等於強迫 InternVL 也變成「等全部到齊才處理」，
會直接抹掉你要保留的架構優勢，所以這裡讓兩邊各自實作，只共用
TileStreamingMemoryBank 本身（eviction 演算法）。
"""
from __future__ import annotations
import abc
import time
import torch
import contextlib


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


def mean_pool_question_embed(get_input_embeddings_fn, tokenizer, questions, dtype):
    """两个模型目前用的算法完全一样：tokenizer -> embedding -> mask 加权平均。
    抽成共用函式，adapter 只需要把自己的 get_input_embeddings 传进来。
    两边原本都是包在 torch.no_grad() 里算的，这里保持一致。"""
    with torch.no_grad():
        q_tok = tokenizer(questions, return_tensors="pt", padding=True).to(DEVICE)
        q_embeds_all = get_input_embeddings_fn(q_tok.input_ids)
        q_mask = q_tok.attention_mask.unsqueeze(-1).float()
        pooled = (q_embeds_all * q_mask).sum(dim=1) / q_mask.sum(dim=1).clamp(min=1e-6)
    return pooled.to(dtype)
 

DEVICE = get_optimal_cuda_device(min_required_gb=30.0)


@contextlib.contextmanager
def measure_peak_memory(tag: str, record_timeline: bool = False):
    """跟 internvl_preprocess.py 完全相同的量測邏輯，直接複製過來避免額外的跨檔相依。"""
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


class StreamModelAdapter(abc.ABC):
    """每個模型實作這個介面。streaming_common.py 只透過這幾個方法跟模型互動。"""
 
    # ---- prompt / tokenizer ----
    @abc.abstractmethod
    def build_text_segments(self, questions: list[str]) -> tuple[list[str], list[str]]:
        """回傳 (text_before_list, text_after_list)，vision span 已切開。"""
 
    @abc.abstractmethod
    def get_tokenizer(self):
        """回傳可直接 tokenizer(list_of_str, padding=True) 的物件
        （InternVL: tokenizer 本身；LLaVA-OV: processor.tokenizer）。"""
 
    @abc.abstractmethod
    def get_input_embeddings(self, model):
        """回傳 callable(input_ids) -> embeds。
        InternVL: model.language_model.get_input_embeddings()
        LLaVA-OV: model.get_input_embeddings()"""
 
    # ---- language model forward（chunked prefill + decode 共用）----
    @abc.abstractmethod
    def lm_prefill(self, model, **kwargs):
        """負責 text_before / vision chunk 的 prefill，只需要 past_key_values，
        不需要 logits。回傳 out（要有 .past_key_values）。"""
 
    @abc.abstractmethod
    def lm_decode_step(self, model, **kwargs) -> tuple[torch.Tensor, object]:
        """跑一次 decode forward，回傳 (next_token_logits, new_past_key_values)。
        InternVL 的 language_model 自帶 lm_head，直接吐 logits；
        LLaVA-OV 的 model.model.language_model 是 base model，要自己接
        model.lm_head(out.last_hidden_state[:, -1, :])。這裡把這個差異吃掉，
        讓 decode loop 兩邊共用同一份程式碼。"""
 
    # ---- 核心分歧點：tile/patch 編碼 + 進 memory bank ----
    @abc.abstractmethod
    def encode_and_bank(
        self, model, pixel_values_list, image_sizes_list,
        question_embeds, budget, score_fn, merge_mode, vit_batch, dtype,
    ) -> tuple[list[torch.Tensor], list[dict]]:
        """回傳 (finalized_per_image, stats_per_image)，每個 finalized[i] 的
        token 數 <= budget。實作方式見本檔案頂部說明，兩個模型完全不同。"""
 
    # ---- baseline（未壓縮）----
    @abc.abstractmethod
    def generate_baseline(self, model, batch, **kwargs) -> list[str]:
        """InternVL: model.batch_chat(...)；LLaVA-OV: processor+model.generate()。"""
 
    # ---- 雜項 ----
    @abc.abstractmethod
    def clean_answer(self, batch, answers: list[str]) -> list[str]:
        """LLaVA-OV 目前有做 clean_answer 後處理，InternVL 沿用同一介面但可
        直接 no-op 回傳，保持共用 decode loop 呼叫端一致。"""
 