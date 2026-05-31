import re
import torch
import transformers
import torch.cuda.nvtx as nvtx
from accelerate import Accelerator
from datasets import load_dataset
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
    dtype = torch.float16
    processor, model = build_model(DEVICE, dtype=dtype)

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
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
    inputs = processor(text=prompt, images=[image], return_tensors="pt").to(DEVICE, dtype)
    
    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=100, pad_token_id=processor.tokenizer.eos_token_id)

    # Decode output id to text
    print(processor.decode(generated_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
