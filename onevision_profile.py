import re
import time
import argparse
import torch
import transformers
import torch.cuda.nvtx as nvtx
from datasets import load_dataset
from accelerate import Accelerator
from transformers.image_utils import load_image
from transformers import AutoProcessor, LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150

print(f"TRANSFORMERS PATH = {transformers.__file__}")
print(f"DEVICE: {DEVICE}")


def build_model(model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16):
    device_map = Accelerator().device
    # processor = AutoProcessor.from_pretrained(model_name)
    processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    print(f"processor type: {type(processor)}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        # attn_implementation="eager"  # 強制關閉 FlashAttention, attn_implementation="flash_attention_2"
    ).to(DEVICE)

    model.eval()
    return processor, model


def clean_answer(messages, answer):
    answer = answer.replace(messages[0]["role"], "").strip()
    for item in messages[0]["content"]:
        if item["type"] == "text":
            answer = answer.replace(item["text"], "").strip()
    
    answer = answer.replace("assistant", "").strip()
    return answer


def generate_answer(model, processor, inputs, messages):
    """
    Returns hidden states from OneVision full pipeline
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image in extreme detail?"},
            ],
        },
    ]
    """
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    answer = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]
    return clean_answer(messages, answer)


def run_online_vision_kv_pipeline(model, processor, inputs, messages, prompt, dtype=torch.float16, chunk_size=1024, vit_batch=5):
    # ==========================================
    # 加入 torch.no_grad() 徹底釋放前向傳播的記憶體
    # ==========================================
    with torch.no_grad():
        # ==========================================
        # Step 1: 載入與前處理 (取得 25 個 Tiles)
        # ==========================================
        print("\n[Step 1] Image Preprocessing ...")

        pixel_values = inputs.pixel_values.to(DEVICE, dtype=dtype)  # [1, 25, 3, 384, 384]
        image_sizes = inputs.image_sizes
        print(f"-> pixel_values shape = {pixel_values.shape}, image_sizes: {image_sizes}")
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

        B, N, C, H, W = pixel_values.shape
    
        # ==========================================
        # Step 2: ViT 微批次串流 (粉碎 Activation 峰值)
        # ==========================================
        print("\n[Step 2] ViT Micro-batching ...")
        flat_pixels = pixel_values.view(B * N, C, H, W) # [25, 3, 384, 384]

        vit_feature_list = []
        # 每次只送入 vit_micro_batch (例如 5) 個 Tile，避免 OOM
        for i in range(0, B * N, vit_batch):
            chunk = flat_pixels[i : i + vit_batch]
            
            # Call Vision Tower (SigLIP)
            vision_outputs = model.model.vision_tower(chunk, output_hidden_states=True)
            # 依據 LLaVA 設定提取倒數第二層特徵
            selected_features = vision_outputs.hidden_states[model.config.vision_feature_layer]

            # 若是 SigLIP，通常不需要像 CLIP 那樣丟棄 CLS token，這由底層邏輯處理
            # 根據 vision_feature_select_strategy 切割
            if model.config.vision_feature_select_strategy == "default":
                selected_features = selected_features[:, 1:]

            # vit_feature_list.append(selected_features.detach().clone())
            vit_feature_list.append(selected_features)
            del vision_outputs, selected_features, chunk  # ← 釋放所有層的激活
            
            # 視情況可以保留 empty_cache 來對抗碎片化，但已經不會存計算圖了
            torch.cuda.empty_cache()
            
            print(f"Chunk {i}, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
        # 在記憶體中重新組合為約 1.8 萬個 Token
        vit_features = torch.cat(vit_feature_list, dim=0) # [25, 729, ViT_Dim]
        print(f"-> vit_features shape = {vit_features.shape}")

        # 通過 Multi-Modal Projector 映射至 LLM 維度
        image_features = model.model.multi_modal_projector(vit_features) # [25, 729, 3584]
        # 加回 Batch 維度 -> [1, 25, 729, 3584]
        image_features = image_features.unsqueeze(0)
        print(f"-> Completed ViT output: image_features shape = {image_features.shape}")

        # ==========================================
        # Step 3: 特徵空間重組與壓縮 (pack_image_features)
        # ==========================================
        print("\n[Step 3] Execute pack_image_features (spatial reconstruction and bilinear interpolation) ...")
        # 呼叫 HF 模型內建的 pack_image_features 函數
        # 這一步會進行：Reshape(大畫布) -> Unpad(去邊界) -> Interpolate(下採樣) -> 加入 \n
        packed_output, feature_lens = model.model.pack_image_features(
            image_features,
            image_sizes,
            image_newline=model.model.image_newline
        )
        packed_image_features = packed_output[0].unsqueeze(0)
        total_vision_tokens = packed_image_features.shape[1]
        del packed_output
    
        print(f"-> Compression complete, final packed_image_features = {packed_image_features.shape} (about {total_vision_tokens} tokens)")
    
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # ==========================================
    # Step 4: LLM Chunked KV 增量建構 (真正零峰值 Prefill)
    # ==========================================
    print(f"\n[Step 4] Start LLM Chunked Prefill (Chunk Size: {chunk_size}) ...")

    # 這裡的 prompt 就是你在前面定義的那個 chat_template 字串
    text_before, text_after = prompt.split("<image>")
    past_key_values = None
    # print(f"text_before: {repr(text_before)}")
    # print(f"text_after: {repr(text_after)}")

    # 4a. 先注入 <image> 之前的文字 (即 "<|im_start|>user ")
    if text_before:
        inputs_before = processor.tokenizer(text_before, return_tensors="pt").to(DEVICE)
        embeds_before = model.model.language_model.get_input_embeddings()(inputs_before.input_ids)
        attention_mask = torch.ones((1, embeds_before.shape[1]), dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=embeds_before,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        past_key_values = outputs.past_key_values

        mem_alloc = torch.cuda.memory_allocated() / 1e9
        print(f"  ├─> Image-front text injection complete, memory alloc: {mem_alloc:.2f} GB")


    # 4b. 接著把那 7362 個 vision token分批餵進去
    for i in range(0, total_vision_tokens, chunk_size):
        chunk_end = min(i + chunk_size, total_vision_tokens)
        vision_chunk = packed_image_features[:, i:chunk_end, :]
        chunk_len = chunk_end - i

        # 取得當前 KV 快取累積的真實長度
        past_seq_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        
        # 1. 精準構建 Attention Mask
        attention_mask = torch.ones((1, past_seq_len + chunk_len), dtype=torch.long, device=DEVICE)
        # 2. 關鍵修正：精準構建當前 Chunk 的真實 Position IDs，避免 FA2 動態記憶體分配失控，其位置應該緊接著過去已處理的總長度
        position_ids = torch.arange(
            past_seq_len, past_seq_len + chunk_len, dtype=torch.long, device=DEVICE
        ).unsqueeze(0)

        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=vision_chunk,
                attention_mask=attention_mask,
                position_ids=position_ids,  # 顯式注入正確的位置編碼
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

        past_key_values = outputs.past_key_values        
        del vision_chunk, outputs, attention_mask

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()  # 每 chunk 後重置，監控下一個 chunk
        mem_alloc = torch.cuda.memory_allocated() / 1e9
        mem_peak = torch.cuda.max_memory_allocated() / 1e9
        print(
            f"  ├─> Vision Chunk {i:04d} ~ {chunk_end:04d} / {total_vision_tokens} Injection complete", 
            f"memory alloc: {mem_alloc:.2f} GB, memory peak: {mem_peak:.2f} GB"
        )

    past_seq_len = past_key_values.get_seq_length()
    print("-> The visual key-value cache has been completed")
    print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    
    # ==========================================
    # Step 5: 文字 Prefill 與手動流式解碼 (Decode)
    # ==========================================
    print("\n[Step 5] Inject the second half of the text and initiate decoding ...")

    # Step 5 prefill: 把後半段的文字轉成 Token
    inputs_after = processor.tokenizer(text_after, return_tensors="pt").to(DEVICE)
    inputs_after_ids = inputs_after.input_ids
    text_len = inputs_after_ids.shape[1]
    # text_embeds = model.model.language_model.get_input_embeddings()(inputs_after_ids)
    # print(f"text_after token count: {inputs_after_ids.shape[1]}")
    # print(f"text_after tokens: {[processor.tokenizer.decode([id]) for id in inputs_after_ids[0]]}")

    # 計算總 Mask 長度 = 過去累積的 (文字1 + 視覺) + 當前新傳入的文字2
    attention_mask = torch.ones((1, past_seq_len + text_len), dtype=torch.long, device=DEVICE)
    # 位置編碼修正
    position_ids = torch.arange(past_seq_len, past_seq_len + text_len, dtype=torch.long, device=DEVICE).unsqueeze(0)

    with torch.no_grad():
        outputs = model(
            input_ids=inputs_after_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,  # 顯式注入
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
    
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    print("\n" + "="*50)
    token = processor.tokenizer.decode(next_token[0])
    answer = token
    print(token, end="", flush=True)

    # 進入自迴歸解碼迴圈 (Decode Loop)
    for step in range(MAX_NEW_TOKENS):
        # 1. 取得當前累積的總 KV 長度
        past_seq_len = past_key_values.get_seq_length()
        # 2. 構建當前 token 的 Attention Mask (過去長度 + 1)
        attention_mask = torch.ones((1, past_seq_len + 1), dtype=torch.long, device=DEVICE)

        # 3. 關鍵修正：顯式指定當前 token 在整串序列中的「精準絕對位置」
        # 因為每次只進來 1 個 token，所以它的位置剛好就是 past_seq_len
        position_ids = torch.tensor([[past_seq_len]], dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                position_ids=position_ids,  # 顯式注入位置編碼，穩住 FA2 記憶體分配器
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        token_id = next_token.item()
        # 遇到結束 Token 則停止
        if token_id == processor.tokenizer.eos_token_id:
            del outputs, next_token_logits, attention_mask, position_ids
            break
            
        token = processor.tokenizer.decode([token_id])
        answer += token
        print(token, end="", flush=True)

        # === 記憶體管理：徹底斷開 Python 引用計數，防止 Activation 殘留 ===
        del outputs, next_token_logits, attention_mask, position_ids
        
        # 每 50 步稍微清空一下 PyTorch 快取釋放碎片（可選，不影響效能）
        if step % 50 == 0:
            torch.cuda.empty_cache()
        
    print("\n" + "="*50)
    return clean_answer(messages, answer)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations to run before profiling region")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat profiled region (avg)")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size to test scaling")
    args = parser.parse_args()

    nvtx.range_push("Load_Model")
    # dtype = torch.float16
    dtype = torch.bfloat16
    processor, model = build_model(dtype=dtype)
    nvtx.range_pop()
    print(f"Model loaded, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Model class: {type(model)}")

    nvtx.range_push("Load_Image")
    image = load_image("4000x6000.jpg")
    nvtx.range_pop()

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image in extreme detail?"},
            ],
        },
    ]

    nvtx.range_push("Image_Processor")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    texts = [prompt] * args.batch_size
    images = [image] * args.batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    nvtx.range_pop()

    # # Warmup (helps remove first-time kernel compile/alloc noise)
    # print(f"Running {args.warmup} warmup iterations...")
    # nvtx.range_push("Warmup_Model")
    # with torch.no_grad():
    #     for i in range(args.warmup):
    #         _ = model.generate(**inputs, max_new_tokens=1)
    # torch.cuda.synchronize()
    # nvtx.range_pop()

    # Profile target region with NVTX ranges
    times = []
    for it in range(args.repeat):
        nvtx.range_push(f"INFER_{it}")

        torch.cuda.synchronize()
        t0 = time.time()

        _ = generate_answer(model, processor, inputs, messages)

        # _ = run_online_vision_kv_pipeline(model, processor, inputs, messages, prompt, dtype=dtype)

        nvtx.range_pop()

        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
        print(f"Run iter {it}: {times[-1]*1000:.2f} ms")

    print(f"Average profiled region time over {args.repeat} runs: {sum(times)/len(times)*1000:.2f} ms")


if __name__ == "__main__":
    main()
