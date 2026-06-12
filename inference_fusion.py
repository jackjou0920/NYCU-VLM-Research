import time
import argparse
import torch
import transformers
import functools
import torch.cuda.nvtx as nvtx
from accelerate import Accelerator
from transformers.image_utils import load_image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20

print(f"TRANSFORMERS PATH = {transformers.__file__}")
print(f"DEVICE: {DEVICE}")


def build_model(device, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16):
    device_map = Accelerator().device
    processor = AutoProcessor.from_pretrained(model_name)

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        # attn_implementation="eager"  # 強制關閉 FlashAttention
    ).to(device)

    model.eval()
    return processor, model


def register_nvtx_hooks(model):
    """
    動態註冊 PyTorch Hooks，將 NVTX marker 深入到模型的特定子模組。
    這能讓我們在 nsys/ncu 中精準捕捉到 Attention 和 Fusion 的 Kernel。
    """
    def wrap_with_nvtx(name, original_forward):
        @functools.wraps(original_forward)
        def wrapper(*args, **kwargs):
            # 1. 確保之前的任務都完了再開始標記
            torch.cuda.synchronize()
            nvtx.range_push(name)
            try:
                # 執行原本的 forward
                res = original_forward(*args, **kwargs)
                # 2. 關鍵！確保這個模組的 GPU Kernel 都跑完了再 Pop
                torch.cuda.synchronize() 
                return res
            finally:
                # 確保即使發生錯誤也會 pop 掉 NVTX
                nvtx.range_pop()
        return wrapper
    
    patched_modules = []
    core = model.model

    # 1. 標記 Vision Tower (處理圖像特徵)
    if hasattr(core, "vision_tower"):
        core.vision_tower.forward = wrap_with_nvtx("Vision_Tower", core.vision_tower.forward)
        patched_modules.append("Vision_Tower")

    # 2. 標記 Multi-modal Projector (視覺與語言的特徵對齊/Fusion 橋樑)
    if hasattr(core, "multi_modal_projector"):
        core.multi_modal_projector.forward = wrap_with_nvtx("MM_Projector", core.multi_modal_projector.forward)
        patched_modules.append("MM_Projector")

    # 3. 標記 Language Model 及內部的 Attention Layers
    if hasattr(core, "language_model"):
        lm = core.language_model
        lm.forward = wrap_with_nvtx("LLM_Forward", lm.forward)
        patched_modules.append("LLM_Forward")
        
        # 深入 Qwen2 (或其他 LLM 骨幹) 內部的 Attention 機制
        # 注意：不同模型的 layer 屬性路徑可能微調，LLaVA-OneVision 通常包在 .model.layers 下
        if hasattr(lm, "layers"):
            for i, layer in enumerate(lm.layers):
                if hasattr(layer, "self_attn"):
                    layer.self_attn.forward = wrap_with_nvtx(f"Attention_Layer_{i}", layer.self_attn.forward)
                if hasattr(layer, "mlp"):
                    layer.mlp.forward = wrap_with_nvtx(f"MLP_Layer_{i}", layer.mlp.forward)
            patched_modules.append("Attention_Layers")

    print(f"✅ Successfully patched NVTX wrappers to: {', '.join(patched_modules)}")


def run_chunked_prefill(model, inputs, chunk_size=1024):
    """
    核心實驗邏輯：手動將 Image 拆分成 Tile 進行串行 Prefill。
    精確處理單個 <image> token 映射到多個融合 Tiles 特徵的架構。
    """

    pixel_values = inputs["pixel_values"]  # Shape: [B, Num_Tiles, C, H, W]
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]
    
    batch_size, num_tiles, C, H, W = pixel_values.shape

    # ---- 階段 1：Tile-by-Tile 影像特徵提取 (降低 Vision Tower Peak Memory) ----
    tile_features_list = []
    for t in range(num_tiles):
        # 每次只讓 1 個 Tile 的 Activation 待在 VRAM，其餘 Tile 算完即釋放
        single_tile = pixel_values[:, t:t+1, :, :, :].flatten(0, 1) # [B, C, H, W]
        with torch.no_grad():
            vision_outputs = model.model.vision_tower(single_tile, output_hidden_states=True)
            selected_image_feature = vision_outputs.hidden_states[-2] # 依 LLaVA 常規取倒數第二層
            # 這是該 Tile 產生的實際視覺 Tokens (例如 576 tokens)
            tile_features = model.model.multi_modal_projector(selected_image_feature)
            tile_features_list.append(tile_features) # 保持獨立，不 cat

    # 將所有 Tile 特徵在 feature 維度水平拼接
    # 注意：這裡拼接的只是 Feature 級別（通常幾百個 tokens * 25），
    # 相比 LLM 的 Activation 矩陣，這點記憶體極小（約百餘 MB），不會造成 Peak VRAM 爆炸。
    all_image_features = torch.cat(tile_features_list, dim=1) # Shape: [B, Total_Image_Tokens, Hidden_Dim]

    # ---- 階段 2：獲取純文字的基礎 Embeddings ----
    text_embeds = model.model.language_model.get_input_embeddings()(input_ids)

    # 找出哪些位置是 <image> 預留標籤
    image_token_index = model.config.image_token_index
    is_image_token = (input_ids == image_token_index)[0] # 取出 Batch 0 的布林陣列

    # 這裡我們手動計算出最終完美的、順序正確的「虛擬完整序列」
    # 但我們**不實例化**它，只用邏輯指標來做虛擬切片
    
    # 建立一個指引地圖：每個位置對應到 text_embeds 還是某個 tile_feature
    # 由於 OneVision 有 Token Merging/Pooling，1個 image_token 會被擴展成 N 個特徵 tokens
    # 我們需要動態追蹤：

    # ---- 階段 3：真正的串流分塊與 LLM 遞迴 ----
    past_key_values = None
    curr_tile_idx = 0
    
    # 用來暫存目前分塊所需要的 Embeddings
    past_key_values = None
    llm_outputs = None
    
    pending_embeds_chunks = []
    current_chunk_len = 0

    # 遍歷原始 input_ids 的每一個位置
    for i in range(input_ids.shape[1]):
        if not is_image_token[i]:
            # 1. 遇到普通文字：取單個 token embedding
            current_embed = text_embeds[:, i:i+1, :]
            pending_embeds_chunks.append(current_embed)
            current_chunk_len += 1
        else:
            # 2. 遇到 <image> 標籤：直接把「整包拼接好的 25 個 Tiles 特徵」灌進去這個位置
            pending_embeds_chunks.append(all_image_features)
            current_chunk_len += all_image_features.shape[1]

        # 3. 檢查目前收集的 Token 長度是否大於等於設定的 chunk_size
        if current_chunk_len >= chunk_size:
            # 拼出當前分塊
            mini_chunk_embeds = torch.cat(pending_embeds_chunks, dim=1)
            
            with torch.no_grad():
                llm_outputs = model.model.language_model(
                    inputs_embeds=mini_chunk_embeds,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True
                )
                past_key_values = llm_outputs.past_key_values
            
            # 清空暫存，立刻釋放這批小張量的 Activation 記憶體
            pending_embeds_chunks = []
            current_chunk_len = 0

    # ---- 階段 4：處理最後剩餘不滿 chunk_size 的尾巴 ----
    if len(pending_embeds_chunks) > 0:
        mini_chunk_embeds = torch.cat(pending_embeds_chunks, dim=1)
        with torch.no_grad():
            llm_outputs = model.model.language_model(
                inputs_embeds=mini_chunk_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            past_key_values = llm_outputs.past_key_values

    return llm_outputs, past_key_values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size to test scaling")
    # 建議新增一個開關，讓你方便在 baseline(原本) 與 chunked(你的方法) 之間切換對比
    parser.add_argument("--method", type=str, default="chunked", choices=["baseline", "chunked"], help="Which prefill method to use")
    args = parser.parse_args()

    nvtx.range_push("Load_Model")
    dtype = torch.float16
    processor, model = build_model(DEVICE, dtype=dtype)
    nvtx.range_pop()
    print(f"Model loaded, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    # 註冊 NVTX Hooks
    register_nvtx_hooks(model)
    print(f"Hooks registered. Ready to profile.")

    # 建議使用高解析度圖片 (例如你之前的 4000x6000)，才能觸發 AnyRes 切分多個 Tile
    image = load_image("4000x6000.jpg") 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image in extreme detail?"},
            ],
        },
    ]

    nvtx.range_push("Preprocess_Data")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    texts = [prompt] * args.batch_size
    images = [image] * args.batch_size

    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE, dtype)
    num_tiles = inputs["pixel_values"].shape[1]
    nvtx.range_pop()

    print(f"Detected Batch Size: {args.batch_size}, Number of Tiles per Image: {num_tiles}")

    # ================= Warmup =================
    print(f"Running {args.warmup} warmup iterations...")
    nvtx.range_push("Warmup")
    for _ in range(args.warmup):
        if args.method == "baseline":
            with torch.no_grad():
                _ = model.generate(**inputs, max_new_tokens=1, pad_token_id=processor.tokenizer.eos_token_id)
        else:
            _, _ = run_chunked_prefill(model, inputs)
    
    nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats() # 重置統計，抓取最準確的 Peak VRAM

    # ================= Profiling =================
    times = []

    print(f"\nStarting Profile Region ({args.method.upper()})...")
    for it in range(args.repeat):
        torch.cuda.synchronize()
        nvtx.range_push(f"INFER_ITER_{it}")
        
        t0 = time.time()
        if args.method == "baseline":
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            
        else:
            llm_outputs, past_key_values = run_chunked_prefill(model, inputs)
        
        elapsed = time.time() - t0
            
        nvtx.range_pop()
        times.append(elapsed)
        print(f"Run iter {it}: {elapsed*1000:.2f} ms")
    
    # ================= Results =================
    avg_time = sum(times) / len(times)
    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    print("\n" + "="*40)
    print(f"Method:        {args.method.upper()}")
    print(f"Batch Size:    {args.batch_size}")
    print(f"Num Tiles:     {num_tiles}")
    print(f"Average TTFT:  {avg_time*1000:.2f} ms")
    print(f"Peak VRAM:     {peak_mem:.2f} GB")
    print("="*40 + "\n")


if __name__ == "__main__":
    main()
