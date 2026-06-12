import time
import argparse
import torch
import transformers
import functools
from accelerate import Accelerator
from transformers.image_utils import load_image
from transformers import AutoProcessor, LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 20

print(f"TRANSFORMERS PATH = {transformers.__file__}")
print(f"DEVICE: {DEVICE}")


def build_model(device, model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16):
    device_map = Accelerator().device
    # processor = AutoProcessor.from_pretrained(model_name)
    processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    print(f"processor type: {type(processor)}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        # attn_implementation="eager"  # 強制關閉 FlashAttention
    ).to(device)

    model.eval()
    return processor, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size to test scaling")
    parser.add_argument("--method", type=str, default="chunked", choices=["baseline", "chunked"], help="Which prefill method to use")
    args = parser.parse_args()

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

    dtype = torch.float16
    processor, model = build_model(DEVICE, dtype=dtype)
    print(f"Model loaded, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True) # 做 inference 時都要設True
    print("-"*50)
    print("Model Prompt:")
    print(prompt)
    print("-"*50)

    texts = [prompt] * args.batch_size
    images = [image] * args.batch_size

    # prompt -> tokenizer -> input_ids
    # image -> image_processor -> pixel_values
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE, dtype)
    print(f"pixel_values shape: {inputs['pixel_values'].shape}")

    num_tiles = inputs["pixel_values"].shape[1]
    print(f"Detected Batch Size: {args.batch_size}, Number of Tiles per Image: {num_tiles}")

    # ================= Profiling =================
    times = []

    print(f"\nStarting Profile Region ({args.method.upper()})...")
    for it in range(args.repeat):
        print(f"INFER_ITER_{it}")
        
        t0 = time.time()
        if args.method == "baseline":
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id)

        
        elapsed = time.time() - t0
        times.append(elapsed)
        print(f"Run iter {it}: {elapsed*1000:.2f} ms")
    
    # # ================= Results =================
    # avg_time = sum(times) / len(times)
    # peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    
    # print("\n" + "="*40)
    # print(f"Method:        {args.method.upper()}")
    # print(f"Batch Size:    {args.batch_size}")
    # print(f"Num Tiles:     {num_tiles}")
    # print(f"Average TTFT:  {avg_time*1000:.2f} ms")
    # print(f"Peak VRAM:     {peak_mem:.2f} GB")
    # print("="*40 + "\n")


if __name__ == "__main__":
    main()
