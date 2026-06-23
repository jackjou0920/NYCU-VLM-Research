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


def trace_llm_input_shape(module, args):
    # args 是一個 tuple，第一個元素 args[0] 就是傳入該 Layer 的 hidden_states
    hidden_states = args[0]
    batch_size, seq_len, hidden_dim = hidden_states.shape
    
    # 我們只關心 Prefill 階段（seq_len > 1）
    if seq_len > 1:
        print(f"\n🔥 [HOOK CONFIRMATION] LLM Layer 0 實際接收到的總 Token 數 (Prefill): {seq_len}")
        print(f"   Tensor Shape (Batch, Seq_Len, Hidden_Dim): [{batch_size}, {seq_len}, {hidden_dim}]")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=1, help="warmup iterations")
    parser.add_argument("--repeat", type=int, default=1, help="how many times to repeat")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size to test scaling")
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

    print(model.config.image_grid_pinpoints)

    with torch.no_grad():
        model.model.language_model.layers[0].register_forward_pre_hook(trace_llm_input_shape)
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, pad_token_id=processor.tokenizer.eos_token_id)

    # Decode output id to text
    print(processor.decode(generated_ids[0], skip_special_tokens=True))
    


if __name__ == "__main__":
    main()
