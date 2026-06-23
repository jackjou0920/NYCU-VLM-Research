import time
import argparse
import torch
import inspect
import transformers
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torchvision import transforms
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from accelerate import Accelerator
from transformers.image_utils import load_image
from transformers import AutoProcessor, LlavaOnevisionProcessor, LlavaOnevisionForConditionalGeneration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_NEW_TOKENS = 150

print(f"TRANSFORMERS PATH = {transformers.__file__}")
print(f"DEVICE: {DEVICE}")


def build_model(model_name="llava-hf/llava-onevision-qwen2-7b-ov-hf", dtype=torch.float16):
    device_map = Accelerator().device
    # processor = AutoProcessor.from_pretrained(model_name)
    processor = LlavaOnevisionProcessor.from_pretrained(model_name)
    print(f"processor type: {type(processor)}")

    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_name,
        dtype=dtype,
        device_map=device_map,
        # attn_implementation="eager"  # 強制關閉 FlashAttention
        attn_implementation="flash_attention_2",
    ).to(DEVICE)

    model.eval()
    return processor, model


def generate_answer(model, processor, image, messages, batch_size=1):
    """
    Returns hidden states from OneVision full pipeline
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image in extreme detail?"},
            ],
        },
    ]
    """
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # 做 inference 時都要設True
    
    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    answer = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    answer = answer.replace(messages[0]["role"], "").strip()
    for item in messages[0]["content"]:
        if item["type"] == "text":
            answer = answer.replace(item["text"], "").strip()
    
    answer = answer.replace("assistant", "").strip()
    return answer


def evaluate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    score = scorer.score(
        reference,
        candidate
    )

    return score["rougeL"].fmeasure


def evaluate_sentence_transformer(reference, candidates):
    # 1. Load a pretrained Sentence Transformer model
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # The sentences to encode
    sentences = [reference] + candidates

    # 2. Calculate embeddings by calling model.encode()
    embeddings = model.encode(sentences)

    # 3. Calculate the embedding similarities
    similarities = model.similarity(embeddings, embeddings)
    return similarities[0][1:]


def run_online_vision_kv_pipeline(model, processor, image, messages, batch_size=1, chunk_size=1024, vit_batch=5):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

    # ==========================================
    # Step 1: 載入與前處理 (取得 25 個 Tiles)
    # ==========================================
    print("\n[Step 1] 影像前處理...")

    pixel_values = inputs.pixel_values.to(DEVICE, dtype=torch.float16)  # [1, 25, 3, 384, 384]
    image_sizes = inputs.image_sizes # 紀錄原始圖片比例，供後續 Unpad 使用
    print(f"-> pixel_values shape = {pixel_values.shape}")
    print(f"image_sizes: {image_sizes}")

    B, N, C, H, W = pixel_values.shape
    
    # ==========================================
    # Step 2: ViT 微批次串流 (粉碎 Activation 峰值)
    # ==========================================
    print("\n[Step 2] ViT 微批次特徵萃取 (Micro-batching)...")
    flat_pixels = pixel_values.view(B * N, C, H, W) # [25, 3, 384, 384]

    vit_feature_list = []
    # 每次只送入 vit_micro_batch (例如 5) 個 Tile，避免 OOM
    for i in range(0, B * N, vit_batch):
        chunk = flat_pixels[i : i + vit_batch]
        
        # Call Vision Tower (SigLIP)
        vision_outputs = model.model.vision_tower(chunk, output_hidden_states=True)
        # 依據 LLaVA 設定提取倒數第二層特徵
        selected_features = vision_outputs.hidden_states[model.config.vision_feature_layer]

        # 若是 SigLIP，通常不需要像 CLIP 那樣丟棄 CLS token，這由底層邏輯處理
        # 根據 vision_feature_select_strategy 切割
        if model.config.vision_feature_select_strategy == "default":
            selected_features = selected_features[:, 1:]

        vit_feature_list.append(selected_features)

    # 在記憶體中重新組合為約 1.8 萬個 Token
    vit_features = torch.cat(vit_feature_list, dim=0) # [25, 729, ViT_Dim]
    print(f"-> vit_features shape = {vit_features.shape}")

    # 通過 Multi-Modal Projector 映射至 LLM 維度
    image_features = model.model.multi_modal_projector(vit_features) # [25, 729, 3584]
    # 加回 Batch 維度 -> [1, 25, 729, 3584]
    image_features = image_features.unsqueeze(0)
    print(f"-> ViT 輸出組合完畢: image_features shape = {image_features.shape}")

    # ==========================================
    # Step 3: 特徵空間重組與壓縮 (pack_image_features)
    # ==========================================
    print("\n[Step 3] 執行 pack_image_features (空間重組與雙線性插值)...")
    # 呼叫 HF 模型內建的 pack_image_features 函數
    # 這一步會進行：Reshape(大畫布) -> Unpad(去邊界) -> Interpolate(下採樣) -> 加入 \n
    packed_output, feature_lens = model.model.pack_image_features(
        image_features,
        image_sizes,
        image_newline=model.model.image_newline
    )
    packed_image_features = packed_output[0].unsqueeze(0)

    total_vision_tokens = packed_image_features.shape[1]
    print(f"-> 壓縮完成，獲得最終視覺特徵: shape = {packed_image_features.shape} (約 {total_vision_tokens} tokens)")

    # ==========================================
    # Step 4: LLM Chunked KV 增量建構 (真正零峰值 Prefill)
    # ==========================================
    print(f"\n[Step 4] 開始 LLM Chunked Prefill (Chunk Size: {chunk_size})...")

    # 這裡的 prompt 就是你在前面定義的那個 chat_template 字串
    text_before, text_after = prompt.split("<image>")

    past_key_values = None

    # 4a. 先注入 <image> 之前的文字 (即 "<|im_start|>user ")
    if text_before:
        inputs_before = processor.tokenizer(text_before, return_tensors="pt").to(DEVICE)
        embeds_before = model.model.language_model.get_input_embeddings()(inputs_before.input_ids)
        attention_mask = torch.ones((1, embeds_before.shape[1]), dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=embeds_before,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
        past_key_values = outputs.past_key_values
        print("  ├─> 圖片前置文字注入完成")

    # 4b. 接著把那 7362 個 vision token 像切香腸一樣，分批餵進去
    for i in range(0, total_vision_tokens, chunk_size):
        chunk_end = min(i + chunk_size, total_vision_tokens)
        vision_chunk = packed_image_features[:, i:chunk_end, :]

        chunk_len = chunk_end - i
        past_seq_len = 0 if past_key_values is None else past_key_values.get_seq_length()
        attention_mask = torch.ones((1, past_seq_len + chunk_len), dtype=torch.long, device=DEVICE)

        with torch.no_grad():
            outputs = model.model.language_model(
                inputs_embeds=vision_chunk,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )

        past_key_values = outputs.past_key_values
        print(f"  ├─> Vision Chunk {i:04d} ~ {chunk_end:04d} / {total_vision_tokens} 注入完成")
        
        del vision_chunk, outputs, attention_mask
        torch.cuda.empty_cache()

    print("-> 視覺 KV Cache 建構完畢")
    
    # ==========================================
    # Step 5: 文字 Prefill 與手動流式解碼 (Decode)
    # ==========================================
    print("\n[Step 5] 注入後半段文字並啟動原生 Generate 解碼...")

    # 把後半段的文字轉成 Token
    text_inputs = processor.tokenizer(text_after, return_tensors="pt").to(DEVICE)
    text_input_ids = text_inputs.input_ids

    # 計算總 Mask 長度 = 過去累積的 (文字1 + 視覺) + 當前新傳入的文字2
    past_seq_len = past_key_values.get_seq_length()
    text_len = text_inputs.input_ids.shape[1]
    attention_mask = torch.ones((1, past_seq_len + text_len), dtype=torch.long, device=DEVICE)

    # 5a. 直接呼叫最外層的 model 本體，它會幫你過完 Qwen2 骨架並自動接上頂層的 lm_head
    with torch.no_grad():
        outputs = model(
            input_ids=text_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
            return_dict=True
        )
    
    past_key_values = outputs.past_key_values
    next_token_logits = outputs.logits[:, -1, :]
    next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)

    print("\n" + "="*50)
    token = processor.tokenizer.decode(next_token[0])
    answer = token
    print(token, end="", flush=True)

    # 進入自迴歸解碼迴圈 (Decode Loop)
    for step in range(MAX_NEW_TOKENS):
        past_seq_len = past_key_values.get_seq_length()
        attention_mask = torch.ones((1, past_seq_len + 1), dtype=torch.long, device=DEVICE)
        
        with torch.no_grad():
            outputs = model(
                input_ids=next_token,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True
            )
            
        past_key_values = outputs.past_key_values
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
        
        token_id = next_token.item()
        # 遇到結束 Token 則停止
        if token_id == processor.tokenizer.eos_token_id:
            break
            
        token = processor.tokenizer.decode([token_id])
        answer += token
        print(token, end="", flush=True)
        
    print("\n" + "="*50)
    return answer


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
    processor, model = build_model(dtype=dtype)
    print(model.config._attn_implementation)
    print(f"Model loaded, CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

    ref_answer = generate_answer(model, processor, image, messages)
    print("\n" + "="*50)
    print(ref_answer)
    print("\n" + "="*50)

    answer = run_online_vision_kv_pipeline(model, processor, image, messages)

    # 4. 指標評估
    print("\n[指標計算中...]")
    # ROUGE-L 字詞重合度
    rouge_l_score = evaluate_rouge(ref_answer, answer)
    
    # SentenceTransformer 語義相似度
    st_scores = evaluate_sentence_transformer(ref_answer, [answer])
    semantic_sim = st_scores[0].item()
    
    print("\n最終比對結果:")
    print(f"  - ROUGE-L Score:      {rouge_l_score:.4f}")
    print(f"  - Semantic Similarity: {semantic_sim:.4f}")



if __name__ == "__main__":
    main()
