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
MAX_NEW_TOKENS = 100

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


def build_prefix_image(image, k, grid=(5, 5)):
    """
    image: PIL image
    k: number of tiles to keep
    grid: (rows, cols)
    """

    W, H = image.size
    rows, cols = grid

    tile_w = W // cols
    tile_h = H // rows
    tiles = []

    # 1. crop tiles
    for r in range(rows):
        for c in range(cols):

            left = c * tile_w
            upper = r * tile_h
            right = (c + 1) * tile_w
            lower = (r + 1) * tile_h
            tiles.append(image.crop((left, upper, right, lower)))

    # 2. create blank canvas
    canvas = Image.new("RGB", (W, H), (0, 0, 0))

    # 3. paste first k tiles
    for idx in range(k):
        r = idx // cols
        c = idx % cols
        canvas.paste(tiles[idx], (c * tile_w, r * tile_h))

    return canvas


def build_anyres_tiles(image, grid=(5, 5), tile_size=384):
    """
    Simulate LLaVA-OneVision AnyRes tiling.

    Parameters
    ----------
    image : PIL.Image

    grid : tuple
        (rows, cols)

    tile_size : int
        resize size for each tile

    Returns
    -------
    List[PIL.Image]
    """

    W, H = image.size

    rows, cols = grid
    tile_w = W // cols
    tile_h = H // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            # handle last row/col
            right = W if c == cols - 1 else (c + 1) * tile_w
            lower = H if r == rows - 1 else (r + 1) * tile_h
            tile = image.crop((left, upper, right, lower,))

            tile = tile.resize(
                (tile_size, tile_size),
                Image.BICUBIC,
            )
            tiles.append(tile)

    return tiles


def extract_tile_feature(model, processor, tile):
    transform = transforms.Compose([
        transforms.Resize((384,384)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=processor.image_processor.image_mean,
            std=processor.image_processor.image_std
        )
    ])

    pixel_values = (transform(tile).unsqueeze(0).to(model.dtype).to(DEVICE))

    with torch.no_grad():
        vision_outputs = model.model.vision_tower(pixel_values=pixel_values)
        projected = model.model.multi_modal_projector(
            vision_outputs.last_hidden_state
        )

    return projected


def build_tile_feature(model, processor, tiles, k):
    feats = []
    for tile in tiles[:k]:
        feat = extract_tile_feature(model, processor, tile)
        feats.append(feat)

    return torch.cat(feats, dim=1)


def extract_kv_cache(model, feat):
    """
    feat:
        [1, N, 3584]

    return:
        past_key_values
    """

    with torch.no_grad():
        outputs = model.model.language_model(
            inputs_embeds=feat,
            use_cache=True,
            return_dict=True
        )

    return outputs.past_key_values


def extract_inputs_embeds(model, processor, image, messages, batch_size=1):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

    captured = {}
    def hook_fn(module, args, kwargs):
        #
        # language_model(
        #     inputs_embeds=...
        # )
        #
        captured["inputs_embeds"] = kwargs["inputs_embeds"].detach()

    handle = model.model.language_model.register_forward_pre_hook(hook_fn, with_kwargs=True)

    with torch.no_grad():
        _ = model(
            **inputs,
            return_dict=True
        )

    handle.remove()

    return (captured["inputs_embeds"], inputs)


def inspect_embedding_layout(model, processor, image, messages, batch_size=1):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)

    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)
    print("input_ids shape:", inputs["input_ids"].shape)

    print("input_ids:")
    print(inputs["input_ids"][0])

    image_token_index = model.config.image_token_index

    image_positions = (
        inputs["input_ids"][0]
        == image_token_index
    ).nonzero(as_tuple=False)

    print("image positions:")
    print(image_positions)
    # Token 0~2 = Chat Template
    # Token 3~7364 = Image Tokens
    # Token 7365~  = Question

    start_idx = image_positions.min().item()
    end_idx   = image_positions.max().item()

    return inputs, start_idx, end_idx


def run_forward(model, processor, image, messages, batch_size=1):
    """
    Returns hidden states from OneVision full pipeline
    """
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # 做 inference 時都要設True
    
    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    return outputs.hidden_states, inputs


def generate_answer(model, processor, image, messages, batch_size=1, max_new_tokens=100):
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
            max_new_tokens=max_new_tokens,
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


def generate_from_inputs_embeds(model, processor, inputs_embeds, attention_mask):

    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
        )

    answer = processor.batch_decode(
        output_ids,
        skip_special_tokens=True
    )[0]

    return answer


def feature_similarity(feat_a, feat_b):
    def mean_pool_feature(feat):
        """
        feat:
            [1,N,3584]

        return:
            [3584]
        """
        return feat.mean(dim=1).squeeze(0)

    feat_a = mean_pool_feature(feat_a)
    feat_b = mean_pool_feature(feat_b)

    return (
        F.cosine_similarity(
            feat_a.unsqueeze(0),
            feat_b.unsqueeze(0)
        )
        .item()
    )


def kv_similarity(kv_a, kv_b):
    """
    return:
        mean_k_sim,
        mean_v_sim
    """

    k_sims = []
    v_sims = []

    for layer_a, layer_b in zip(kv_a, kv_b):
        k1 = layer_a[0]
        v1 = layer_a[1]
        k2 = layer_b[0]
        v2 = layer_b[1]

        # ---------------------
        # Pool sequence dim
        # ---------------------
        k1 = k1.mean(dim=2)
        k2 = k2.mean(dim=2)
        v1 = v1.mean(dim=2)
        v2 = v2.mean(dim=2)

        # ---------------------
        # Flatten
        # ---------------------
        k1 = k1.flatten()
        k2 = k2.flatten()
        v1 = v1.flatten()
        v2 = v2.flatten()

        # ---------------------
        # Cosine
        # ---------------------

        k_sim = F.cosine_similarity(
            k1.unsqueeze(0),
            k2.unsqueeze(0)
        ).item()

        v_sim = F.cosine_similarity(
            v1.unsqueeze(0),
            v2.unsqueeze(0)
        ).item()

        k_sims.append(k_sim)
        v_sims.append(v_sim)

    return (
        sum(k_sims)/len(k_sims),
        sum(v_sims)/len(v_sims)
    )


def prefix_kv_similarity(kv_prev, kv_cur):
    """
    Compare:

        KV(prev)

    vs

        KV(cur) prefix

    Return:
        layer_k_sims
        layer_v_sims
    """

    layer_k_sims = []
    layer_v_sims = []

    for layer_prev, layer_cur in zip(kv_prev, kv_cur):

        k_prev = layer_prev[0]
        v_prev = layer_prev[1]

        k_cur = layer_cur[0]
        v_cur = layer_cur[1]

        seq_len = k_prev.shape[2]

        # --------------------------------
        # Align prefix
        # --------------------------------

        k_cur_prefix = k_cur[:, :, :seq_len, :]
        v_cur_prefix = v_cur[:, :, :seq_len, :]

        # --------------------------------
        # Flatten
        # --------------------------------

        k_prev_flat = k_prev.float().flatten()
        k_cur_flat = k_cur_prefix.float().flatten()

        v_prev_flat = v_prev.float().flatten()
        v_cur_flat = v_cur_prefix.float().flatten()

        # --------------------------------
        # Cosine
        # --------------------------------

        k_sim = F.cosine_similarity(
            k_prev_flat.unsqueeze(0),
            k_cur_flat.unsqueeze(0),
            dim=1
        ).item()

        v_sim = F.cosine_similarity(
            v_prev_flat.unsqueeze(0),
            v_cur_flat.unsqueeze(0),
            dim=1
        ).item()

        layer_k_sims.append(k_sim)
        layer_v_sims.append(v_sim)

    return layer_k_sims, layer_v_sims


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


def extract_hidden_states(model, feat):
    """
    feat:
        [1, N, hidden_size]

    return:
        tuple(
            layer0_hidden,
            layer1_hidden,
            ...
        )

    each:
        [1, seq_len, hidden_size]
    """

    with torch.no_grad():
        outputs = model.model.language_model(
            inputs_embeds=feat,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False
        )

    return outputs.hidden_states


def hidden_state_prefix_similarity_tokenwise(hs_prev, hs_cur):
    sims = []

    for h_prev, h_cur in zip(hs_prev, hs_cur):
        seq_len = h_prev.shape[1]
        h_cur_prefix = h_cur[:, :seq_len, :]
        token_sim = F.cosine_similarity(
            h_prev.float(),
            h_cur_prefix.float(),
            dim=-1
        )
        sims.append(
            token_sim.mean().item()
        )

    return sims


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


def plot_similarity_curve(baseline_answer, results):
    """
    baseline_answer: str

    results:
        {
            1: "...",
            5: "...",
            10: "...",
            ...
        }
    """

    # -------------------------
    # ROUGE-L
    # -------------------------
    rouge = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    # -------------------------
    # Sentence Transformer
    # -------------------------
    st_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    tile_counts = sorted(results.keys())
    rouge_scores = []
    semantic_scores = []

    baseline_emb = st_model.encode(
        baseline_answer,
        convert_to_tensor=True
    )

    for k in tile_counts:
        answer = results[k]

        # -------------------------
        # ROUGE-L
        # -------------------------
        rouge_l = rouge.score(
            baseline_answer,
            answer
        )["rougeL"].fmeasure

        rouge_scores.append(float(rouge_l))

        # -------------------------
        # Semantic Similarity
        # -------------------------
        emb = st_model.encode(answer, convert_to_tensor=True)
        sim = cos_sim(baseline_emb, emb).item()

        semantic_scores.append(sim)
        print(
            f"Tiles={k:2d} | "
            f"ROUGE-L={rouge_l:.4f} | "
            f"Semantic={sim:.4f}"
        )

    # -------------------------
    # Plot
    # -------------------------

    plt.figure(figsize=(8,5))
    plt.plot(
        tile_counts,
        rouge_scores,
        marker="o",
        label="ROUGE-L"
    )
    plt.plot(
        tile_counts,
        semantic_scores,
        marker="s",
        label="SentenceTransformer"
    )

    plt.xlabel("Number of Tiles")
    plt.ylabel("Similarity")
    plt.title("Tile vs Similarity to Full Image Answer")

    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("tile_similarity.png")

    return {
        "tiles": tile_counts,
        "rouge": rouge_scores,
        "semantic": semantic_scores,
    }


def plot_tiles(tiles):
    fig, axes = plt.subplots(5, 5, figsize=(12,12))

    for i in range(25):
        ax = axes[i//5][i%5]
        ax.imshow(tiles[i])
        ax.set_title(f"Tile {i+1}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("anyres_tiles.png")


def prefix_generation_experiment(model, processor, image, messages):
    baseline_answer = generate_answer(model, processor, image, messages)
    print(baseline_answer)
    print("="*80)
    print("Groud Truth Answer:")
    print(baseline_answer)

    results = {}
    for k in [1, 5, 10, 15, 20, 25]:
        prefix_img = build_prefix_image(image, k, grid=(5,5))
        candidate_answer = generate_answer(model, processor, prefix_img, messages)

        print("="*80)
        print(f"Prefix={k}")
        print(candidate_answer)
        results[k] = candidate_answer

    plot_similarity_curve(baseline_answer, results)


def run_feature_similarity_experiment(model, processor, image, messages):
    tiles = build_anyres_tiles(image)
    print(f"Number of Tiles: {len(tiles)}")
    print("Tile size:", tiles[0].size)
    # plot_tiles(tiles)

    reference_feat = build_tile_feature(model, processor, tiles, 25)

    similarities = []
    for k in [1, 5, 10, 15, 20, 25]:
        feat = build_tile_feature(model, processor, tiles, k)

        sim = feature_similarity(
            feat,
            reference_feat
        )
        similarities.append(sim)

        print(f"Tiles={k:2d} | " f"Feature Similarity={sim:.4f}")


def run_kv_similarity_experiment(model, processor, image, messages):
    tiles = build_anyres_tiles(image)

    # -------------------------
    # Reference
    # -------------------------

    reference_feat = build_tile_feature(
        model,
        processor,
        tiles,
        25
    )

    reference_kv = extract_kv_cache(
        model,
        reference_feat
    )

    # -------------------------
    # Incremental
    # -------------------------

    for k in [1, 5, 10, 15, 20, 25]:

        feat = build_tile_feature(
            model,
            processor,
            tiles,
            k
        )

        kv = extract_kv_cache(
            model,
            feat
        )

        k_sim, v_sim = kv_similarity(
            kv,
            reference_kv
        )

        print(
            f"Tiles={k:2d} | "
            f"K Similarity={k_sim:.4f} | "
            f"V Similarity={v_sim:.4f}"
        )


def run_hidden_state_preservation_experiment(model, processor, image, messages):
    """
    Measure:

        H(1)
        vs
        H(5) prefix

        H(5)
        vs
        H(10) prefix

        ...
    """

    tiles = build_anyres_tiles(image)

    tile_counts = [1, 5, 10, 15, 20, 25]
    hidden_dict = {}

    print("=" * 80)
    print("Building Hidden States")
    print("=" * 80)

    for k in tile_counts:
        feat = build_tile_feature(model, processor, tiles, k)
        hidden_dict[k] = extract_hidden_states(model, feat)

    for prev_k, cur_k in zip(tile_counts[:-1], tile_counts[1:]):
        sims = hidden_state_prefix_similarity_tokenwise(
            hidden_dict[prev_k],
            hidden_dict[cur_k]
        )

        print()
        print(f"{prev_k} -> {cur_k}")
        print(f"Mean Similarity = {np.mean(sims):.6f}")
        # print()

        # for layer_idx, sim in enumerate(sims):
        #     print(
        #         f"Layer {layer_idx:02d} | "
        #         f"Similarity={sim:.6f}"
        #     )


def run_embedding_reconstruction(model, processor, image, messages):
    print("=" * 80)
    print("Reference Generate")
    print("=" * 80)

    ref_answer = generate_answer(model, processor, image, messages)
    print(ref_answer)

    print()
    print("=" * 80)
    print("Extract Embeddings")
    print("=" * 80)

    inputs_embeds, inputs = extract_inputs_embeds(model, processor, image, messages)
    print("inputs_embeds:", inputs_embeds.shape)  # [1, 7380, 3584] 7380 = Image Tokens + Text Tokens

    print()
    print("=" * 80)
    print("Generate From Embeddings")
    print("=" * 80)

    answer = generate_from_inputs_embeds(model, processor, inputs_embeds, inputs["attention_mask"])
    print(answer)

    rouge = evaluate_rouge(
        ref_answer,
        answer
    )
    print()
    print(f"ROUGE-L = {rouge:.4f}")
    

def run_grid_row_incremental_generation(model, processor, image, messages, max_new_tokens=100):
    print("\n" + "="*40)
    # 1. 取得完整推論的 Ground Truth Embeddings 與 Mask
    inputs_embeds, inputs = extract_inputs_embeds(model, processor, image, messages)
    # print("inputs_embeds:", inputs_embeds.shape)  # [1, 7380, 3584] 7380 = Image Tokens + Text Tokens
    attention_mask = inputs["attention_mask"]

    # 2. 定位視覺 Token 的起止點
    inputs, start_idx, end_idx = inspect_embedding_layout(model, processor, image, messages)
    image_embeds = inputs_embeds[:, start_idx:end_idx+1, :]
    # print("start:", start_idx)
    # print("end:", end_idx)
    # print("image tokens:", end_idx - start_idx + 1)

    # image_embeds = inputs_embeds[:, start_idx:end_idx+1, :]
    # prefix_text_embeds = inputs_embeds[:, :start_idx, :]
    # suffix_text_embeds = inputs_embeds[:, end_idx+1:, :]
    # print("chat template token:", prefix_text_embeds.shape)
    # print("image token:", image_embeds.shape)
    # print("text token:", suffix_text_embeds.shape)

    # 3. 假設你的模型變數叫 model
    # 它的類型是 torch.nn.Parameter，形狀是 [hidden_size] (例如 Qwen2 骨幹通常是 3584 或 5120)
    if hasattr(model, "image_newline"):
        newline_embed = model.image_newline
    else:
        # 在某些 transformers 版本中，它可能封裝在內層的 model 裡
        newline_embed = model.model.image_newline
    print("Image Newline Embedding Shape:", newline_embed.shape)

    # 在視覺區段內尋找與 newline_embed 完全一致的 Token 位置
    is_newline = torch.all(torch.isclose(image_embeds[0], newline_embed, atol=1e-3), dim=-1)
    newline_indices = is_newline.nonzero(as_tuple=True)[0]
    print(f"[分析] 偵測到總共有 {len(newline_indices)} 個 image_newline tokens.")

    # 4. 辨識 Global Base Image 與 Local Tiles 的相對位置
    # AnyRes 的全域圖固定為 729 tokens (27x27)，且不含 newline。
    if len(newline_indices) > 0 and newline_indices[0] > 729:
        print("[佈局] 偵測到 Global Base Image 排在【前方】")
        global_at_front = True
        global_start, global_end = start_idx, start_idx + 729
        local_start, local_end = global_end, end_idx + 1
    else:
        print("[佈局] 偵測到 Global Base Image 排在【後方】")
        global_at_front = False
        local_start, local_end = start_idx, end_idx - 728
        global_start, global_end = local_end, end_idx + 1

    # 將 newline 索引轉換為相對於 inputs_embeds 的絕對座標
    abs_newline_indices = newline_indices + start_idx + (729 if global_at_front else 0)
    
    # ==========================================
    # 5. 切割 Grid-Rows (加上殘缺行保護機制)
    # ==========================================
    grid_rows = []
    curr_ptr = local_start
    lines_per_grid_row = 27
    total_newlines = len(abs_newline_indices)
    
    for i in range(0, total_newlines, lines_per_grid_row):
        # 隄防最後一行不滿 27 行的狀況
        current_row_end_line_idx = min(i + lines_per_grid_row - 1, total_newlines - 1)
        
        # 取得這一行最後一個 newline 在全圖中的絕對 Token 索引
        end_newline_idx = abs_newline_indices[current_row_end_line_idx].item()
        
        # 切出當前 Grid-Row 的特徵區段
        grid_rows.append(inputs_embeds[:, curr_ptr : end_newline_idx + 1, :])
        
        # 更新指標，下一行從這個 newline 的下一個位置開始
        curr_ptr = end_newline_idx + 1
        
    # 如果後面還有漏網之魚（通常不會，保險用）
    if curr_ptr < local_end:
        grid_rows.append(inputs_embeds[:, curr_ptr:local_end, :])
        
    print(f"[切片] 成功將 Local Tiles 拆解為 {len(grid_rows)} 個動態 Grid-Rows。")

    # ==========================================
    # 6. 模擬 Online 串流：增量建構 KV Cache (對純骨幹 model 呼叫)
    # ==========================================
    past_key_values = None
    llm_backbone = model.model.language_model

    # Step 0: 處理文字開頭 (Prefix) + Global Image (若在前方)
    prefix_text = inputs_embeds[:, :start_idx, :]
    if global_at_front:
        global_emb = inputs_embeds[:, global_start:global_end, :]
        step0_embeds = torch.cat([prefix_text, global_emb], dim=1)
    else:
        step0_embeds = prefix_text
        
    with torch.no_grad():
        outputs = llm_backbone(
            inputs_embeds=step0_embeds,
            use_cache=True,
            past_key_values=past_key_values
        )
        past_key_values = outputs.past_key_values
    print(f" -> [串流中] Step 0 (Prefix) 已寫入 KV Cache. Tokens: {step0_embeds.shape[1]}")

    
    # Step 1 ~ N: 逐行將 Grid-Rows 餵入模型 (Causal Append)
    for r_idx, row_emb in enumerate(grid_rows):
        with torch.no_grad():
            outputs = llm_backbone(
                inputs_embeds=row_emb,
                use_cache=True,
                past_key_values=past_key_values
            )
            past_key_values = outputs.past_key_values
        print(f" -> [串流中] Grid-Row {r_idx} 已追加至 KV Cache. Tokens: {row_emb.shape[1]}")
        
    # Step Final: 準備最後的輸入 (Global 若在後方 + 終端問題 Text)
    suffix_text = inputs_embeds[:, end_idx + 1:, :]
    if not global_at_front:
        global_emb = inputs_embeds[:, global_start:global_end, :]
        final_embeds = torch.cat([global_emb, suffix_text], dim=1)
    else:
        final_embeds = suffix_text

    # 【關鍵修正】：重構完整的 inputs_embeds 迎合 transformers 內部的切片機制
    # 將所有前面已經餵過、以及最後沒餵的片段，依序拼接成一個總長度為 7380 的完整 Tensor
    all_pieces = [step0_embeds] + grid_rows + [final_embeds]
    reconstructed_inputs_embeds = torch.cat(all_pieces, dim=1)
        
    # ==========================================
    # 7. 啟動 Autoregressive Decode (對 CausalLM 呼叫)
    # ==========================================
    past_length = past_key_values.get_seq_length()
        
    total_len = reconstructed_inputs_embeds.shape[1] # 應為 7380
    extended_attention_mask = torch.ones((1, total_len), dtype=torch.long, device=DEVICE)
    
    print(f" -> [串流完成] 當前快取長度: {past_length} tokens, 剩餘輸入: {final_embeds.shape[1]} tokens")
    print(f" -> 啟動語言模型解碼解題...")
    
    with torch.no_grad():
        output_ids = model.generate(
            inputs_embeds=reconstructed_inputs_embeds, # 傳入重構後的完整序列
            past_key_values=past_key_values,
            attention_mask=extended_attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        
    # 8. 解碼答案
    incremental_answer = processor.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return incremental_answer


def run_incremental_versus_baseline_experiment(model, processor, image, messages):
    print("\n" + "="*80)
    print(" 執行實驗：Full Image Baseline vs Grid-Row Online KV")
    print("="*80)

    # 1. 計算 Baseline 答案
    print("[1/2] 正在計算全圖 Baseline 答案...")
    baseline_answer = generate_answer(model, processor, image, messages, max_new_tokens=MAX_NEW_TOKENS)

    # 2. 計算 增量打包 答案
    print("\n[2/2] 正在計算 Grid-Row 增量串流答案...")
    incremental_answer = run_grid_row_incremental_generation(model, processor, image, messages, max_new_tokens=MAX_NEW_TOKENS)
    
    # 3. 秀出答案對比
    print("\n" + "="*80)
    print(f"【Baseline 全圖答案】:\n{baseline_answer}")
    print("-"*80)
    print(f"【Grid-Row 增量答案】:\n{incremental_answer}")
    print("="*80)
    
    # 4. 指標評估
    print("\n[指標計算中...]")
    # ROUGE-L 字詞重合度
    rouge_l_score = evaluate_rouge(baseline_answer, incremental_answer)
    
    # SentenceTransformer 語義相似度
    st_scores = evaluate_sentence_transformer(baseline_answer, [incremental_answer])
    semantic_sim = st_scores[0].item()
    
    print("\n最終比對結果:")
    print(f"  - ROUGE-L Score:      {rouge_l_score:.4f}")
    print(f"  - Semantic Similarity: {semantic_sim:.4f}")


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
    # print(inspect.getsource(model.pack_image_features))


    # prefix_generation_experiment(model, processor, image, messages)

    # run_feature_similarity_experiment(model, processor, image, messages)

    # run_kv_similarity_experiment(model, processor, image, messages)

    # run_hidden_state_preservation_experiment(model, processor, image, messages)

    # run_embedding_reconstruction(model, processor, image, messages)

    run_incremental_versus_baseline_experiment(model, processor, image, messages)

    # print(model.config.vision_aspect_ratio)
    # print(processor.image_processor)
    # # print(processor.image_processor.image_grid_pinpoints)




if __name__ == "__main__":
    main()
