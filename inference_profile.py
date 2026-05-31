import re
import time
import argparse
import torch
import transformers
import torch.cuda.nvtx as nvtx
from datasets import load_dataset
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations to run before profiling region")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat profiled region (avg)")
    args = parser.parse_args()

    nvtx.range_push("Load_Model")
    dtype = torch.float16
    processor, model = build_model(DEVICE, dtype=dtype)
    nvtx.range_pop()
    print(f"Model loaded, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9} GB")
    print(f"Model class: {type(model)}")

    # Load images (local + remote example)
    image = load_image("resize.jpeg")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]

    nvtx.range_push("Preprocess_Data")
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE, dtype)
    nvtx.range_pop()

    # Warmup (helps remove first-time kernel compile/alloc noise)
    print(f"Running {args.warmup} warmup iterations...")
    nvtx.range_push("Warmup_Model")
    with torch.no_grad():
        for i in range(args.warmup):
            _ = model.generate(**inputs, max_new_tokens=1)
    torch.cuda.synchronize()
    nvtx.range_pop()

    # Profile target region with NVTX ranges
    times = []
    for it in range(args.repeat):
        torch.cuda.synchronize()
        nvtx.range_push(f"INFER_{it}")
        t0 = time.time()

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.eos_token_id)

        torch.cuda.synchronize()
        nvtx.range_pop()
        t1 = time.time()

        times.append(t1 - t0)
        print(f"Run iter {it}: {times[-1]*1000:.2f} ms")

    print(f"Average profiled region time over {args.repeat} runs: {sum(times)/len(times)*1000:.2f} ms")


if __name__ == "__main__":
    main()
