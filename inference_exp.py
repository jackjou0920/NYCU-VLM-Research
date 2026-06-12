import time
import random
import argparse
import torch
import transformers
import functools
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from rouge_score import rouge_scorer
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


def build_random_image(image, selected_tiles, grid=(5,5)):
    W, H = image.size
    rows, cols = grid
    tile_w = W // cols
    tile_h = H // rows

    tiles = []
    for r in range(rows):
        for c in range(cols):
            left = c * tile_w
            upper = r * tile_h
            right = (c + 1) * tile_w
            lower = (r + 1) * tile_h

            tiles.append(image.crop((left, upper, right, lower)))

    canvas = Image.new("RGB", (W, H), (0, 0, 0))

    for idx in selected_tiles:
        r = idx // cols
        c = idx % cols
        canvas.paste(
            tiles[idx],
            (c * tile_w, r * tile_h)
        )
    return canvas


def get_hidden_states(model, processor, image, prompt):
    inputs = processor(text=prompt, images=image, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    return outputs.hidden_states


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


def run_forward_with_kv(model, processor, image, messages, batch_size=1):
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)  # 做 inference 時都要設True

    texts = [prompt] * batch_size
    images = [image] * batch_size
    inputs = processor(text=texts, images=images, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model(
            **inputs,
            use_cache=True,
            output_hidden_states=False,
            return_dict=True
        )
    return outputs.past_key_values


def generate_answer(model, processor, image, messages, batch_size=1, max_new_tokens=100):
    """
    Returns hidden states from OneVision full pipeline
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
    return answer


def cosine_sim(a, b):
    a = a.reshape(-1, a.shape[-1])
    b = b.reshape(-1, b.shape[-1])
    return F.cosine_similarity(a, b, dim=-1).mean().item()


def compare_layers(hs_prefix, hs_full):
    """
    hs_prefix: hidden_states from prefix image
    hs_full: hidden_states from full image
    """

    sims = []
    num_layers = len(hs_full)

    for layer in range(num_layers):
        h1 = hs_prefix[layer]
        h2 = hs_full[layer]
        sim = cosine_sim(h1, h2)
        sims.append(sim)
        print(f"Layer {layer:02d}: {sim:.4f}")
    return sims


def compare_incremental(hs_prev, hs_curr):
    sims = []
    num_layers = len(hs_curr)
    for layer in range(num_layers):
        h1 = hs_prev[layer]
        h2 = hs_curr[layer]

        sim = cosine_sim(h1, h2)
        sims.append(sim)
    return sims


def compare_kv_cache(kv_prev, kv_cur):
    k_sims = []
    v_sims = []

    for layer_idx, (layer_prev, layer_cur) in enumerate(zip(kv_prev, kv_cur)):
        k1 = layer_prev[0]
        v1 = layer_prev[1]

        k2 = layer_cur[0]
        v2 = layer_cur[1]

        # --------------------------------
        # Prefix k 的長度
        # --------------------------------

        old_len = min(
            k1.shape[2],
            k2.shape[2]
        )

        k1 = k1[:, :, :old_len, :]
        v1 = v1[:, :, :old_len, :]

        k2 = k2[:, :, :old_len, :]
        v2 = v2[:, :, :old_len, :]

        # --------------------------------
        # flatten
        # --------------------------------

        k1_flat = k1.reshape(-1, k1.shape[-1])
        k2_flat = k2.reshape(-1, k2.shape[-1])

        v1_flat = v1.reshape(-1, v1.shape[-1])
        v2_flat = v2.reshape(-1, v2.shape[-1])

        # --------------------------------
        # cosine similarity
        # --------------------------------

        k_sim = F.cosine_similarity(
            k1_flat,
            k2_flat,
            dim=-1
        ).mean().item()

        v_sim = F.cosine_similarity(
            v1_flat,
            v2_flat,
            dim=-1
        ).mean().item()

        k_sims.append(k_sim)
        v_sims.append(v_sim)

        print(
            f"Layer {layer_idx:02d} | "
            f"K={k_sim:.4f} "
            f"V={v_sim:.4f}"
        )

    return k_sims, v_sims


def plot_similarity(data):
    layers = list(range(29))

    # ---- plot ----
    plt.figure(figsize=(10,6))

    for k, vals in data.items():
        plt.plot(layers, vals, marker='o', linewidth=2, label=f'Prefix k={k}')

    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Layer-wise Similarity vs Prefix Tiles (OneVision)")
    plt.legend()
    plt.grid(True)
    plt.ylim(0.2, 0.95)
    plt.savefig("similarity.png")

    k_vals = [1, 5, 10, 15, 20, 25]
    final_layer_sim = [
        data[1][-1],
        data[5][-1],
        data[10][-1],
        data[15][-1],
        data[20][-1],
        data[25][-1],
    ]

    plt.figure(figsize=(6,4))
    plt.plot(k_vals, final_layer_sim, marker='o')
    plt.xlabel("Number of Prefix Tiles (k)")
    plt.ylabel("Final Layer Similarity")
    plt.title("Vision Context Saturation Curve")
    plt.grid(True)
    plt.savefig("vision_context.png")


def evaluate_answer_sim(reference, candidate):
    scorer = rouge_scorer.RougeScorer(
        ["rougeL"],
        use_stemmer=True
    )

    score = scorer.score(
        reference,
        candidate
    )

    return score["rougeL"].fmeasure


def plot_random_results(results_mean, results_std):
    layers = np.arange(29)
    plt.figure(figsize=(10,6))

    for k in results_mean:
        mean = results_mean[k]
        std = results_std[k]
        plt.plot(layers, mean, label=f'Random k={k}')

        plt.fill_between(layers, mean - std, mean + std, alpha=0.2)

    plt.xlabel("Layer")
    plt.ylabel("Cosine Similarity")
    plt.title("Random Tile Sampling")
    plt.legend()
    plt.grid(True)

    plt.savefig(
        "random_sampling_similarity.png",
        dpi=300,
        bbox_inches="tight"
    )


def plot_incremental_sim(heatmap):
    plt.figure(figsize=(12,8))

    plt.imshow(heatmap, aspect='auto', origin='lower')
    plt.colorbar(label="Cosine Similarity")
    plt.xlabel("Layer")
    plt.ylabel("Prefix Step")

    plt.title("Incremental Prefix Similarity")
    plt.yticks(
        range(24),
        [f"{i}->{i+1}" for i in range(1,25)]
    )
    plt.savefig(
        "incremental_prefix_heatmap.png",
        dpi=300,
        bbox_inches="tight"
    )


def plot_heatmap(heatmap, title, filename):
    plt.figure(figsize=(12,8))
    plt.imshow(heatmap, aspect='auto', origin='lower')
    plt.colorbar()

    plt.xlabel("Layer")
    plt.ylabel("Prefix Step")
    plt.title(title)

    plt.savefig(filename, dpi=300,bbox_inches="tight")



def prefix_experiment(model, processor, image, messages):
    # Full image baseline
    hs_full, _ = run_forward(model, processor, image, messages)

    results = {}
    for k in [1, 5, 10, 15, 20, 25]:
        prefix_img = build_prefix_image(image, k, grid=(5,5))
        # prefix_img.save(f"prefix_img_{str(k)}.png", "PNG")
        
        hs_prefix, _ = run_forward(model, processor, prefix_img, messages)
        results[k] = compare_layers(hs_prefix, hs_full)

        print("\n======================")
        print(f"Prefix Tiles = {k}")
        print(f"Similarity = {results[k]}")
        print("======================")

    plot_similarity(results)


def random_experiment(model, processor, image, messages):
    # Full image baseline
    hs_full, _ = run_forward(model, processor, image, messages)

    num_trials = 10
    results_mean = {}
    results_std = {}
    for k in [1, 5, 10, 15, 20]:
        all_sims = []
        for trial in range(num_trials):
            selected_tiles = random.sample(range(25), k)
            # print(f"k={k}, trial={trial}, tiles={selected_tiles}")

            rand_img = build_random_image(image, selected_tiles,grid=(5,5))

            hs_rand, _ = run_forward(model, processor, rand_img, messages)
            sims = compare_layers(hs_rand, hs_full)
            all_sims.append(sims)

        all_sims = np.array(all_sims)

        results_mean[k] = all_sims.mean(axis=0)
        results_std[k] = all_sims.std(axis=0)

        print("\n========================")
        print(f"k={k}")
        print(f"Mean:")
        print(results_mean[k])
        print(f"Std:")
        print(results_std[k])
        print("========================")

    plot_random_results(results_mean, results_std)


def incremental_experiment(model, processor, image, messages):
    heatmap = []

    prefix_img = build_prefix_image(image, 1, grid=(5,5))
    prev_hs, _ = run_forward(model, processor, prefix_img, messages)
  
    for k in range(2, 26):
        prefix_img = build_prefix_image(image, k, grid=(5,5))
        cur_hs, _ = run_forward(model, processor, prefix_img, messages)
        
        sims = compare_incremental(prev_hs, cur_hs)
        heatmap.append(sims)

        print()
        print(f"Prefix {k-1} -> {k}")
        for layer, sim in enumerate(sims):
            print(f"Layer {layer:02d}: {sim:.4f}")

        prev_hs = cur_hs

    heatmap = np.array(heatmap)
    plot_incremental_sim(heatmap)


def incremental_kv_experiment(model, processor, image, messages):
    kv_k_heatmap = []
    kv_v_heatmap = []

    prefix_img = build_prefix_image(image, 1, grid=(5,5))
    prev_kv = run_forward_with_kv(model, processor, prefix_img, messages)

    for k in range(2, 26):
        prefix_img = build_prefix_image(image, k, grid=(5,5))
        cur_kv = run_forward_with_kv(model, processor, prefix_img, messages)

        k_sims, v_sims = compare_kv_cache(prev_kv, cur_kv)
        kv_k_heatmap.append(k_sims)
        kv_v_heatmap.append(v_sims)

        print()
        print(f"{k-1}->{k}")
        print(f"K mean = {sum(k_sims)/len(k_sims):.4f}")
        print(f"V mean = {sum(v_sims)/len(v_sims):.4f}")

        prev_kv = cur_kv

    kv_k_heatmap = np.array(kv_k_heatmap)
    kv_v_heatmap = np.array(kv_v_heatmap)

    plot_heatmap(kv_k_heatmap, "Incremental Key Similarity", "kv_key_similarity.png")
    plot_heatmap(kv_v_heatmap, "Incremental Value Similarity", "kv_value_similarity.png")

    
def prefix_generation_experiment(model, processor, image, messages):
    ref_answer = generate_answer(model, processor, image, messages)
    print(ref_answer)
    print("="*80)
    print("Groud Truth Answer:")
    print(ref_answer)

    for k in [5, 10, 15, 20, 25]:
        prefix_img = build_prefix_image(image, k, grid=(5,5))
        candidate_answer = generate_answer(model, processor, prefix_img, messages)

        print("="*80)
        print(f"Prefix={k}")
        print(candidate_answer)
        
        score = evaluate_answer_sim(ref_answer, candidate_answer)
        print(f"ROUGE score = {score}\n")


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


    # # 1-1. Prefix experiments
    # prefix_experiment(model, processor, image, messages)

    # # 1-2. Random experiments
    # random_experiment(model, processor, image, messages)

    # # 2. Prefix incremental experiments
    # incremental_experiment(model, processor, image, messages)

    # # 3. Prefix incremental KV experiments
    # incremental_kv_experiment(model, processor, image, messages)

    # 4. Prefix KV Reuse + Generation Quality Evaluation
    prefix_generation_experiment(model, processor, image, messages)

    
    


if __name__ == "__main__":
    main()
