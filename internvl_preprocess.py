import math

import torch
from torchvision import transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


# ──────────────────────────────────────────────────────────────────────────────
# 影像前處理（動態高解析度，複製自官方 HF model card 範例）
# ──────────────────────────────────────────────────────────────────────────────

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def build_transform(input_size=448):
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def dynamic_grid(image, min_num=1, max_num=12, image_size=448):
    """官方 dynamic_preprocess 選的網格 (cols, rows)：比例最接近原圖、塊數最少（平手看面積）。"""
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    return find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )


def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    best_ratio = dynamic_grid(image, min_num, max_num, image_size)

    target_width  = image_size * best_ratio[0]
    target_height = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % best_ratio[0]) * image_size,
            (i // best_ratio[0]) * image_size,
            ((i % best_ratio[0]) + 1) * image_size,
            ((i // best_ratio[0]) + 1) * image_size,
        )
        processed_images.append(resized_img.crop(box))

    if use_thumbnail and len(processed_images) != 1:
        processed_images.append(image.resize((image_size, image_size)))

    return processed_images


def load_image_tiles(image, input_size=448, max_num=12):
    """回傳 pixel_values [N_tiles, 3, H, W] 以及 tile 數量 num_patches"""
    transform = build_transform(input_size)
    tiles = dynamic_preprocess(image, image_size=input_size, max_num=max_num, use_thumbnail=True)
    pixel_values = torch.stack([transform(t) for t in tiles])
    return pixel_values


def fill_grid(aspect_ratio, max_tiles, max_stretch=1.5):
    """Low-Res（fill）用的網格：i*j <= max_tiles，長寬比拉伸 <= max_stretch 倍的網格裡挑塊數最多的，
    平手挑拉伸最小的；沒有任何網格在容忍範圍內時，退回拉伸最小的網格。
    官方 find_closest_aspect_ratio 是「比例最接近、塊數最少」，常常用不滿 budget
    （例：4:3 + 上限 3 塊 → 1x1），這裡改成盡量用滿 budget。"""
    cands = [(i, j) for i in range(1, max_tiles + 1) for j in range(1, max_tiles + 1) if i * j <= max_tiles]
    stretch = lambda g: abs(math.log((g[0] / g[1]) / aspect_ratio))
    ok = [g for g in cands if stretch(g) <= math.log(max_stretch) + 1e-9] or [min(cands, key=stretch)]
    return max(ok, key=lambda g: (g[0] * g[1], -stretch(g)))


def _grid_tiles(image, cols, rows, input_size):
    """整張圖 resize 成 (input_size*cols, input_size*rows) 再逐列切塊（同 dynamic_preprocess 的順序）。"""
    resized = image.resize((input_size * cols, input_size * rows))
    return [resized.crop((c * input_size, r * input_size, (c + 1) * input_size, (r + 1) * input_size))
            for r in range(rows) for c in range(cols)]


def load_image_tiles_fill(image, input_size=448, max_tiles=4, max_stretch=1.5):
    """Low-Res（fill）baseline：整張圖 resize 成 fill_grid 選的網格再切塊，不加縮圖，
    tile 數 <= max_tiles（= budget // num_image_token），盡量用滿 budget。"""
    transform = build_transform(input_size)
    w, h = image.size
    cols, rows = fill_grid(w / h, max_tiles, max_stretch)
    return torch.stack([transform(t) for t in _grid_tiles(image, cols, rows, input_size)])


COVER_MAX_STRETCH = 1.75   # 覆蓋網格的長寬比拉伸上限。v0 用 1.5（同 Low-Res-Fill）時直式文件在 m=2/3 有 60%
                           # 只切得出 1x1（1x2 要拉伸 1.54 倍），名目 m 跟實際塊數差很多；1.75 讓常見的
                           # m = 2/4/6/8 大多切得滿，又不到 2 倍的明顯變形。m=1 不受影響（一律 1x1）。


def load_image_multiscale(image, input_size=448, max_num_grid=47, coverage_tiles=1,
                          max_stretch=COVER_MAX_STRETCH):
    """多尺度 SVMA 的輸入：
      cover  ：整張圖 resize 成 fill_grid(·, coverage_tiles, max_stretch) 的網格（低解析度全圖覆蓋，常駐 KV；
               切不滿 m 塊時用最接近的較小網格，實際塊數看 cover_grid）
      detail ：官方 dynamic_preprocess 的網格 tile（上限 max_num_grid 塊，不含縮圖），之後挑 K - m 塊寫進 KV。
               （試過依原生解析度切的細節網格：HR-Bench 4K 小幅提升、8K 明顯變差、文件類持平或變差，已移除）
    coverage_tiles = 1 時 cover 就是官方的 1x1 縮圖，跟原本的 SVMA 一樣。
    detail 不比 cover 細（塊數 <= cover 塊數，例如原圖本來就只切 1 塊）時，細節 tile 沒有意義，回傳 0 塊。
    回傳 dict(cover=[m, 3, H, W], cover_grid=(cols, rows), detail=[N, 3, H, W], detail_grid=(cols, rows))。"""
    transform = build_transform(input_size)
    w, h = image.size
    cc, cr = fill_grid(w / h, coverage_tiles, max_stretch)
    cover = torch.stack([transform(t) for t in _grid_tiles(image, cc, cr, input_size)])
    dc, dr = dynamic_grid(image, max_num=max_num_grid, image_size=input_size)
    if dc * dr <= cc * cr:
        detail, (dc, dr) = cover.new_zeros((0, *cover.shape[1:])), (0, 0)
    else:
        detail = torch.stack([transform(t) for t in _grid_tiles(image, dc, dr, input_size)])
    return {"cover": cover, "cover_grid": (cc, cr), "detail": detail, "detail_grid": (dc, dr)}


# ──────────────────────────────────────────────────────────────────────────────
# 模型載入
# ──────────────────────────────────────────────────────────────────────────────
 
def build_model(
    model_name: str = "OpenGVLab/InternVL3_5-8B",
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda:0"
):
    print(f"Loading tokenizer from {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
 
    print(f"Loading model from {model_name} ...")
    model = AutoModel.from_pretrained(
        model_name,
        dtype=dtype,
        trust_remote_code=True,
        use_flash_attn=True,  # 關閉 Flash Attention 以降低記憶體碎片（可改 True 加速）
        low_cpu_mem_usage=True,
    ).to(device).eval()
 
    return tokenizer, model
