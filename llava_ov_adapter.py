"""
LlavaOVAdapter：把 llava_ov_svm.py 裡「已確認沒問題」的 Step 1~5 邏輯原封不動
包成 StreamModelAdapter 介面，供 streaming_common.py 共用骨架呼叫。
 
encode_and_bank() 就是原本 run_online_kv_with_memory_bank() 的 Step 3~5，逐行
照搬（官方 pack_image_features → 只有超過 budget 才用 TileStreamingMemoryBank
在「列」粒度上淘汰），沒有改寫任何數學或順序。
 
注意：這裡 import 的是 llava_ov_core（葉節點模組），不是 llava_ov_svm——
llava_ov_svm.py 本身會 import LlavaOVAdapter，若這裡改成
`from llava_ov_svm import ...` 會形成 llava_ov_svm ⇄ llava_ov_adapter 的
循環 import（跟 InternVL 那邊踩過的坑一樣）。
"""
from __future__ import annotations

import torch
from transformers.models.llava_onevision.modeling_llava_onevision import get_anyres_image_grid_shape
from stream_adapters import DEVICE, StreamModelAdapter
from stream_memory_bank import TileStreamingMemoryBank
from llava_ov_core import (
    build_prompt, split_prompt_at_vision, encode_patch,
    num_image_tokens_per_patch, compute_packed_row_layout,
    generate_answer_standard, clean_answer,
)
 
 
class LlavaOVAdapter(StreamModelAdapter):
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
    ):
        B = len(pixel_values_list)
        num_patches_list = [pv.shape[0] for pv in pixel_values_list]
        num_image_token = num_image_tokens_per_patch(model)
        if budget < num_image_token:
            raise ValueError(
                f"budget={budget} 小於單一 patch 的 token 數 ({num_image_token})，"
                f"base image 本身就無法塞進這個 budget，這個實驗設定不可行。"
            )
 
        print(f"\n[Memory Bank] budget={budget}  score_fn={score_fn}  mode={merge_mode}")
        print(f"  patches per image (before compression) : {num_patches_list}")
        print(f"  raw vision tokens per image             : "
              f"{[n * num_image_token for n in num_patches_list]}")
 
        # ── Step 3：per-patch SigLIP（micro-batch 串流，控制 ViT 階段的 peak memory）
        #    這一步跟 budget 大小、要不要 eviction 無關，永遠都要跑 ──
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
 
        # ── Step 4：官方 pack_image_features（unpad + 必要時內插 + image_newline）
        #    保證結果 in-distribution，不管單張圖的 crop 數多寡 ──
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
                question_embed=question_embeds[b],
                protected_tiles=0,   # base image 已經在迴圈外處理，bank 只管網格列
            )
            for r in range(num_rows):
                bank.add_tile(rows[r])   # 評分含這一列的 newline token，權重很小可忽略
 
            final_grid_tokens, stats = bank.finalize()   # flat 模式：newline 已內建在每個 tile 裡，不用再插
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
        return finalized, all_stats
 
    def generate_baseline(self, model, batch, **kwargs):
        return generate_answer_standard(model, self.processor, batch)
 
    def clean_answer(self, batch_or_questions, answers):
        # streaming_common.py 只傳 questions（list[str]）進來，但 llava_ov_core
        # 的 clean_answer() 需要 dataset["question"] 才能把 echo 回來的問題文字
        # 從答案裡 strip 掉。這裡包一層轉換，clean_answer 本身邏輯不變。
        pseudo_datasets = [{"question": q} for q in batch_or_questions]
        return clean_answer(pseudo_datasets, answers)