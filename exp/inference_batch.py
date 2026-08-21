import time
import argparse
import torch
import transformers
import functools
import numpy as np
import torch.cuda.nvtx as nvtx

from dataclasses import dataclass
from accelerate import Accelerator
from transformers.image_utils import load_image
from transformers import (
    AutoProcessor,
    LlavaOnevisionForConditionalGeneration,
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TOTAL_REQUESTS = 1
MAX_NEW_TOKENS = 20

print(f"TRANSFORMERS PATH = {transformers.__file__}")
print(f"DEVICE: {DEVICE}")


@dataclass
class Request:
    req_id: int
    arrival_time: float = 0
    start_time: float = 0
    first_token_time: float = 0
    end_time: float = 0


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
    def wrap_with_nvtx(name, original_forward):
        @functools.wraps(original_forward)
        def wrapper(*args, **kwargs):

            torch.cuda.synchronize()
            nvtx.range_push(name)

            try:
                res = original_forward(*args, **kwargs)
                torch.cuda.synchronize()
                return res

            finally:
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


def build_requests():
    requests = []
    now = time.time()
    for i in range(TOTAL_REQUESTS):
        r = Request(req_id=i)
        r.arrival_time = now
        requests.append(r)
    return requests


def run_batch(batch_requests, processor, model, prompt, image):
    batch_size = len(batch_requests)

    texts = [prompt] * batch_size
    images = [image] * batch_size

    inputs = processor(
        text=texts,
        images=images,
        return_tensors="pt",
        padding=True,
    ).to(DEVICE, torch.float16)

    for r in batch_requests:
        r.start_time = time.time()

    torch.cuda.synchronize()

    # -------------------------
    # TTFT approximation
    # -------------------------
    ttft_start = time.time()

    with torch.no_grad():
        # Prefill + first decode step
        generated = model.generate(
            **inputs,
            max_new_tokens=1,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    torch.cuda.synchronize()
    ttft_end = time.time()

    for r in batch_requests:
        r.first_token_time = ttft_end

    # -------------------------
    # Remaining decode
    # -------------------------
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            pad_token_id=processor.tokenizer.eos_token_id,
            use_cache=True,
        )

    torch.cuda.synchronize()
    end_time = time.time()

    for r in batch_requests:
        r.end_time = end_time


def compute_metrics(requests):
    ttfts = [
        r.first_token_time - r.arrival_time
        for r in requests
    ]
    latencies = [
        r.end_time - r.arrival_time
        for r in requests
    ]

    total_tokens = TOTAL_REQUESTS * MAX_NEW_TOKENS
    total_time = (
        max(r.end_time for r in requests)
        - min(r.arrival_time for r in requests)
    )

    throughput = total_tokens / total_time

    print("\n================ RESULTS ================")
    print(f"TTFT AVG      : {np.mean(ttfts):.4f} s")
    print(f"TTFT P95      : {np.percentile(ttfts,95):.4f} s")
    print(f"Latency AVG   : {np.mean(latencies):.4f} s")
    print(f"Latency P95   : {np.percentile(latencies,95):.4f} s")
    print(f"Throughput    : {throughput:.2f} tokens/s")
    print("=========================================\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat")
    parser.add_argument("--batch_size", type=int, default=1, choices=[1, 4, 16], help="batch size to test scaling")
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

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    
    print(f"Running {args.warmup} warmup iterations with Batch Size {args.batch_size}...")
    dummy_inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(DEVICE, torch.float16)
    with torch.no_grad():
        for _ in range(args.warmup):
            _ = model.generate(**dummy_inputs, max_new_tokens=2)
    torch.cuda.synchronize()


    # ----------------------------------
    # Build burst requests
    # ----------------------------------
    requests = build_requests()
    batch_size = args.batch_size
    print(f"\nRunning batch_size={batch_size}")
    print(f"Total requests={TOTAL_REQUESTS}")


    # ----------------------------------
    # Scheduler
    # ----------------------------------
    for i in range(0, TOTAL_REQUESTS, batch_size):
        batch_requests = requests[i:i + batch_size]
        nvtx.range_push(f"BATCH_{i//batch_size}")
        run_batch(batch_requests, processor, model, prompt, image)
        nvtx.range_pop()


    # ----------------------------------
    # Metrics
    # ----------------------------------
    compute_metrics(requests)


if __name__ == "__main__":
    main()
