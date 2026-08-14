import os
import time
import glob
import torch
import argparse
from accelerate import Accelerator
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from transformers.image_utils import load_image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150

# ──────────────────────────────────────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
):
    device_map = Accelerator().device
    processor = AutoProcessor.from_pretrained(model_name)
    print(f"processor type: {type(processor)}")

    print(f"Loading model from {model_name} ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        attn_implementation="flash_attention_2",
    ).to(DEVICE)

    model.eval()
    return processor, model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name",  type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--image",       type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--batch_size",  type=int, default=1,   help="Image batch size")
    args = parser.parse_args()

    question = "What is shown in this image in extreme detail?"
    dtype = torch.bfloat16

    torch.cuda.synchronize()
    t0 = time.time()

    # ── 1. 載入模型 ──
    processor, model = build_model(args.model_name, dtype=dtype)
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    print(f"Load model time: {elapsed:.2f} s")

    # ── 2. 找出所有圖片路徑（先只收集路徑，不要一次把全部圖片前處理/載進記憶體）──
    if os.path.isdir(args.image):
        image_paths = sorted(sum(
            [glob.glob(os.path.join(args.image, e)) for e in ("*.jpg", "*.jpeg", "*.png")], []
        ))
        if not image_paths:
            raise FileNotFoundError(f"No images in {args.image}")
    else:
        image_paths = [args.image] * args.batch_size
    print(f"Found {len(image_paths)} image(s) to process (batch_size={args.batch_size})")


    image = load_image(args.image) 
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        },
    ]

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True) # 做 inference 時都要設True
    print("-"*50)
    print("Model Prompt:")
    print(prompt)
    print("-"*50)

    texts = [prompt] * args.batch_size
    images = [image] * args.batch_size
    
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE, dtype)
    print(f"pixel_values shape: {inputs['pixel_values'].shape}")

    for k,v in inputs.items():
        if torch.is_tensor(v):
            print(k,v.shape)

    print(model.visual)
    for name, module in model.visual.named_modules():
        print(name, type(module))

    return

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id)
    
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    print(output_text[0])


if __name__ == "__main__":
    main()
