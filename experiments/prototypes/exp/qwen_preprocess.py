import time
import torch
import contextlib
from PIL import Image
from accelerate import Accelerator
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@contextlib.contextmanager
def measure_peak_memory(tag: str, record_timeline: bool = False):
    if record_timeline:
        torch.cuda.memory._record_memory_history(max_entries=100000)

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()

    try:
        yield
    finally:
        torch.cuda.synchronize()
        elapsed = time.time() - t0
        peak_alloc = torch.cuda.max_memory_allocated() / 1e9
        peak_reserved = torch.cuda.max_memory_reserved() / 1e9

        print(f"\n[{tag}] time={elapsed:.2f} sec  "
              f"peak_allocated={peak_alloc:.3f} GB  "
              f"peak_reserved={peak_reserved:.3f} GB")

        if record_timeline:
            torch.cuda.memory._dump_snapshot(f"{tag}_snapshot.pickle")
            torch.cuda.memory._record_memory_history(enabled=None)


# ──────────────────────────────────────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────────────────────────────────────

def build_model(
    model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
    dtype: torch.dtype = torch.bfloat16,
):
    print(f"Loading processor from {model_name} ...")
    processor = AutoProcessor.from_pretrained(model_name)
    tokenizer = processor.tokenizer

    print(f"Loading model from {model_name} ...")
    device_map = Accelerator().device
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
        attn_implementation="flash_attention_2",  # 建議開啟,尤其多圖/長影片場景省記憶體
    ).to(DEVICE).eval()

    return tokenizer, processor, model
