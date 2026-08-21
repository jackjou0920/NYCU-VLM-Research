"""
InternVLAdapter：保留 InternVL 架構上的核心優勢－tile 之間彼此獨立（無
cross-tile 依賴，不像 LLaVA-OV 的 unpad/interpolate 需要整張圖一起算），
可以逐 tile encode → add_tile，不需要等所有 tile 到齊才開始處理。

跟 llava_ov_adapter.py 的分工原則一樣：這裡只是把 internvl_svm.py 裡
原本 encode_tile + 逐 tile add_tile 的循環包成 encode_and_bank()，序列/數學
不變。唯一「向 llava_ov 對齊」的地方是補上跟 llava_ov 一樣的 budget 合法性
檢查（原本 InternVL 版本沒有這個 guard）。

【架構備註】目前 TileStreamingMemoryBank.finalize() 設計上需要等所有 tile
都 add 完才能做全域 top-K/分箱決策（這是之前修 bug 時刻意做的設計，避免
「來一個丟一個」的貪婪淘汰在資訊不足時做出次優決策）。所以這裡的
encode_and_bank 目前還是「先收集完所有 tile 的 embedding 才 finalize」，
但因為 InternVL tile 之間沒有跨 tile 依賴，未來若要真正做到「tile 一來就
決定要不要送進 LLM chunked prefill”，只需要在這個函數內把
Step(ViT+bank) 與 streaming_common 的 chunked injection 交錯執行即可－
這是 LLaVA-OV 架構上做不到的（它的 pack_image_features 本身就是全圖依賴），
所以這個可擴展性差異被完整地保留在這一個函數裡，不影響共用骨架。
"""
from __future__ import annotations

import torch
from stream_adapters import DEVICE, StreamModelAdapter
from stream_memory_bank import TileStreamingMemoryBank
from internvl_core import (
    build_prompt, split_prompt_at_vision, encode_tile,
    generate_answer_standard, clean_answer
)


class InternVLAdapter(StreamModelAdapter):
    def __init__(self, tokenizer, model, budget: int):
        # InternVL 的 prompt 里 <IMG_CONTEXT> 数量取决于 equiv_tiles = budget
        # 换算出的等效 tile 数（跟 internvl_svm.py 原本 inline 逻辑一致），
        # 而这个数字在建 prompt 当下就要知道（不像 llava_ov 要等 pack 完才知道），
        # 所以在建构 adapter 时一併带入 model / budget。
        self.tokenizer = tokenizer
        self.model = model
        self.budget = budget
 
    def build_text_segments(self, questions):
        num_image_token = self.model.num_image_token  # num_image_token=256
        equiv_tiles = -(-self.budget // num_image_token)  # ceil division，跟原本一致
 
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
        with torch.no_grad():
            return model.language_model(**kwargs)
 
    def lm_decode_step(self, model, **kwargs):
        with torch.no_grad():
            out = model.language_model(**kwargs)
        return out.logits[:, -1, :], out.past_key_values
 
    def encode_and_bank(
        self, model, pixel_values_list, image_sizes_list,
        question_embeds, budget, score_fn, merge_mode, vit_batch, dtype,
    ):
        B = len(pixel_values_list)
        num_tiles_list = [pv.shape[0] for pv in pixel_values_list]
        num_image_token = model.num_image_token
        if budget < num_image_token:
            raise ValueError(
                f"budget={budget} 小於單一 tile 的 token 數 ({num_image_token})，"
                f"protected tile 本身就無法塞進這個 budget。"
            )

        print(f"\n[Memory Bank] budget={budget}  score_fn={score_fn}  mode={merge_mode}")
        print(f"  tiles per image (before compression) : {num_tiles_list}")
        print(f"  raw vision tokens per image             : "
                f"{[n * num_image_token for n in num_tiles_list]}")

        # 攤平：把所有圖片的真實 tile 串成一條 flat tensor，並記錄每個 tile 屬於哪張圖
        flat_tiles = torch.cat(pixel_values_list, dim=0)  # [sum(N_tiles_b), 3, 448, 448]
        owner = []                                        # owner[k] = 第 k 個 flat tile 屬於哪個 batch index
        for b, n in enumerate(num_tiles_list):
            owner.extend([b] * n)
 
        D_llm = model.mlp1[-1].out_features
        banks = [
            TileStreamingMemoryBank(
                capacity=budget, dim=D_llm, device=DEVICE, dtype=dtype,
                score_fn=score_fn, mode=merge_mode, num_image_token=num_image_token,
                question_embed=question_embeds[b],
            ) for b in range(B)
        ]
 
        # ── 核心：tile 之間彼此獨立，逐 tile encode -> add_tile。 
        # 這個循環完全沒有依賴「其他 image 或其他 tile 是否處理完」，
        # 是 InternVL 相對 LLaVA-OV 保留的關鍵架構彈性。 ──
        for i in range(0, flat_tiles.shape[0], vit_batch):
            chunk = flat_tiles[i:i + vit_batch].to(DEVICE, dtype=dtype)
            owner_chunk = owner[i:i + vit_batch]
 
            tile_tokens = encode_tile(model, chunk, dtype=dtype)
            del chunk
            torch.cuda.empty_cache()
 
            for j, b in enumerate(owner_chunk):
                banks[b].add_tile(tile_tokens[j])

            mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9
            print(f"  ├─> [ViT+Bank] flat tile {i}~{min(i+vit_batch, flat_tiles.shape[0])}"
                  f"/{flat_tiles.shape[0]} done, peak alloc={mem:.2f} GB")
 
        del flat_tiles

        print()
        finalized, all_stats = [], []
        for b in range(B):
            toks, stats = banks[b].finalize()
            finalized.append(toks)
            all_stats.append(stats)
            print(f"  image {b}: raw={stats['total_seen']:5d} -> final={stats['final_size']:5d} "
                  f"(budget={budget}, compression={stats['compression_ratio']:.2f}x, "
                  f"dropped/merged={stats['total_dropped']})")
            assert stats["final_size"] <= budget
 
        return finalized, all_stats
 
    def generate_baseline(self, model, **kwargs):
        pixel_values_list = kwargs["pixel_values_list"]
        questions = kwargs["questions"]
        return generate_answer_standard(model, self.tokenizer, pixel_values_list, questions)
 
    def clean_answer(self, batch_or_questions, answers):
        return clean_answer(answers)
