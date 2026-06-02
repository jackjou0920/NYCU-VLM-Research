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


def run_chunked_prefill_profiling(model, inputs):
    """
    核心實驗邏輯：手動將 Image 拆分成 Tile 進行串行 Prefill。
    關閉 use_cache，純粹評估分塊 Prefill 下的 Activation 記憶體足跡
    """

    text_input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"]
    
    batch_size = text_input_ids.shape[0]
    num_tiles = pixel_values.shape[1]

    torch.cuda.synchronize()
    nvtx.range_push("TILE_BY_TILE_PREFILL_FLOW")
    t0 = time.time()

    with torch.no_grad():
        # A. 取得基礎文字 Embedding
        text_embeds = model.model.language_model.get_input_embeddings()(text_input_ids)
        
        # B. 依序處理每個 Tile (不保留、不拼接 KV Cache)
        for tile_idx in range(num_tiles):
            nvtx.range_push(f"Tile_Pipeline_Stage_{tile_idx}")
            
            # --- Stage 1: Vision Tower ---
            single_pixel_value = pixel_values[:, tile_idx:tile_idx+1, :, :, :]
            single_pixel_value = single_pixel_value.flatten(0, 1) 
            vision_outputs = model.model.vision_tower(single_pixel_value, output_hidden_states=True)
            selected_image_feature = vision_outputs.hidden_states[-2]
            
            # --- Stage 2: Projector ---
            image_features = model.model.multi_modal_projector(selected_image_feature)

            # --- Stage 3: Incremental LLM Prefill ---
            # 說明：這是一個硬體資源消耗的近似測試。我們將文字與第一個 Tile 拼接，後續只增量輸入 Tile 特徵。
            # 這樣能完美模擬 Chunked 計算時的 Memory 與 Compute 負載。
            if tile_idx == 0:
                current_inputs_embeds = torch.cat([text_embeds, image_features], dim=1)
            else:
                current_inputs_embeds = image_features
            
            # 💡 關鍵點：強制將 use_cache 設為 False 
            # 阻斷 Hugging Face 底層動態分配與拼接帶來的記憶體垃圾碎片
            llm_outputs = model.model.language_model(
                inputs_embeds=current_inputs_embeds,
                past_key_values=None,
                use_cache=False,
                return_dict=True
            )
            
            nvtx.range_pop() # Tile_Pipeline_Stage

        # C. 模擬輸出特徵提取
        last_hidden_state = llm_outputs.last_hidden_state[:, -1, :]
        if hasattr(model, "lm_head"):
            last_tile_logits = model.lm_head(last_hidden_state)
        else:
            last_tile_logits = model.language_model.lm_head(last_hidden_state)

        # 3. 找出機率最大的 Token ID
        next_token_id = torch.argmax(last_tile_logits, dim=-1)

    torch.cuda.synchronize()
    nvtx.range_pop() # TILE_BY_TILE_PREFILL_FLOW
    t1 = time.time()
    
    return t1 - t0, next_token_id


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
            _ = run_chunked_prefill_profiling(model, inputs)
    
    nvtx.range_pop()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats() # 重置統計，抓取最準確的 Peak VRAM

    # ================= Profiling =================
    times = []

    print(f"\nStarting Profile Region ({args.method.upper()})...")
    for it in range(args.repeat):
        torch.cuda.synchronize()
        nvtx.range_push(f"INFER_ITER_{it}")
        
        if args.method == "baseline":
            t0 = time.time()
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id)
            torch.cuda.synchronize()
            elapsed = time.time() - t0
        else:
            elapsed, next_token_id = run_chunked_prefill_profiling(model, inputs)
            
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
