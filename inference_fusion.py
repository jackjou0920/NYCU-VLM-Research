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
print(f"TRANSFORMERS PATH = {transformers.__file__}")
print(f"DEVICE: {DEVICE}")


def build_model(device, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16):
    device_map = Accelerator().device
    processor = AutoProcessor.from_pretrained(model_name)

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size to test scaling")
    args = parser.parse_args()

    nvtx.range_push("Load_Model")
    dtype = torch.float16
    processor, model = build_model(DEVICE, dtype=dtype)
    nvtx.range_pop()
    print(f"Model loaded, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")

    # 註冊 NVTX Hooks
    register_nvtx_hooks(model)
    print(f"Hooks registered. Ready to profile.")

    # Load images and duplicate for batch size
    image = load_image("resize.jpeg")
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
    
    # 建立 Batch
    texts = [prompt] * args.batch_size
    images = [image] * args.batch_size

    # 注意：Batch size > 1 時需要 padding
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE, dtype)
    nvtx.range_pop()

    print(f"Running {args.warmup} warmup iterations with Batch Size {args.batch_size}...")
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.generate(**inputs, max_new_tokens=2) # warmup 不需要生太多 token
    torch.cuda.synchronize()

    # Profile target region
    times = []
    print("Starting Profile Region...")
    for it in range(args.repeat):
        torch.cuda.synchronize()
        nvtx.range_push(f"INFER_ITER_{it}")
        t0 = time.time()

        with torch.no_grad():
            # max_new_tokens 設小一點，我們主要想看 Prefill 階段的龐大 Attention 計算
            generated_ids = model.generate(**inputs, max_new_tokens=20, pad_token_id=processor.tokenizer.eos_token_id)

        torch.cuda.synchronize()
        nvtx.range_pop()
        t1 = time.time()

        times.append(t1 - t0)
        print(f"Run iter {it}: {times[-1]*1000:.2f} ms")

    print(f"Average time over {args.repeat} runs: {sum(times)/len(times)*1000:.2f} ms")


if __name__ == "__main__":
    main()
