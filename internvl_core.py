# ══════════════════════════════════════════════════════════════════════════════
# InternVL 專屬的低階函式（prompt 構造 / tile 編碼 / baseline 生成）。
#
# 獨立成這支「葉節點」模組，是為了避免循環 import：
#   internvl_svm.py（entry point）需要 InternVLAdapter，
#   internvl_adapter.py 需要這裡的函式，
# 如果這些函式留在 internvl_svm.py 裡，internvl_adapter.py import internvl_svm、
# internvl_svm.py 又 import internvl_adapter，就會是 A→B→A 的循環 import。
# 這裡的內容跟原本 internvl_svm.py 裡的版本逐行相同，純粹搬家，沒有改邏輯。
# ══════════════════════════════════════════════════════════════════════════════
import torch
import importlib
from stream_adapters import DEVICE

MAX_NEW_TOKENS = 300
IMG_START_TOKEN   = "<img>"
IMG_END_TOKEN     = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


def clean_answer(answers):
    for i, answer in enumerate(answers):
        answer = answer.replace("<|im_end|>", "").strip()
        answers[i] = answer
 
    return answers


# ──────────────────────────────────────────────────────────────────────────────
# 標準 generate 參考路徑（用於正確性比較）
# ──────────────────────────────────────────────────────────────────────────────
def generate_answer_standard(model, tokenizer, pixel_values_list, questions):
    """走官方 model.batch_chat() 路徑。batch_chat 本身就是為不同圖片/不同 tile 數設計的，
    不需要 padding：把每張圖的 tiles 直接沿 batch 維度 cat 起來，用 num_patches_list
    告訴模型每張圖各自佔幾個 tile 即可。"""
 
    assert len(pixel_values_list) == len(questions)
 
    generation_config = dict(max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    num_patches_list = [pv.shape[0] for pv in pixel_values_list]
    pixel_values_batch = torch.cat(pixel_values_list, dim=0).to(DEVICE, dtype=model.dtype)
    questions_fmt = [f'<image>\n{q}' for q in questions]
 
    responses = model.batch_chat(
        tokenizer,
        pixel_values_batch,
        num_patches_list=num_patches_list,
        questions=questions_fmt,
        generation_config=generation_config,
    )
    return responses


# ──────────────────────────────────────────────────────────────────────────────
# Prompt 工具
# ──────────────────────────────────────────────────────────────────────────────
# def build_prompt(tokenizer, model, question: str, num_image_tiles: int) -> str:
#     """
#     不依賴 internvl 套件的簡化版本：直接用 Qwen3 chat template。
#     InternVL3.5-8B (Qwen3 backbone) 的 system message 格式：
#         <|im_start|>system\n{sys}<|im_end|>\n
#         <|im_start|>user\n{content}<|im_end|>\n
#         <|im_start|>assistant\n
#     """
#     image_tokens = (
#         IMG_START_TOKEN
#         + IMG_CONTEXT_TOKEN * model.num_image_token * num_image_tiles
#         + IMG_END_TOKEN
#     )
#     content = image_tokens + "\n" + question
 
#     messages = [{"role": "user", "content": content}]
#     prompt = tokenizer.apply_chat_template(
#         messages,
#         tokenize=False,
#         add_generation_prompt=True,
#     )
#     return prompt


def build_prompt(tokenizer, model, question: str, num_image_tiles: int) -> str:
    """改用模型自己的 conversation template，跟 model.chat()/batch_chat() 內部
    組 prompt 的方式逐行一致，確保 baseline 與 streaming 路徑吃到完全相同的
    system message + 角色格式，只有 image token 數量依 budget 換算不同。"""
    model_module = importlib.import_module(type(model).__module__)
    get_conv_template = model_module.get_conv_template  # 跟 chat() 內部用的是同一個函式

    template = get_conv_template(model.template)
    template.system_message = model.system_message

    image_tokens = (
        IMG_START_TOKEN
        + IMG_CONTEXT_TOKEN * model.num_image_token * num_image_tiles
        + IMG_END_TOKEN
    )
    # 跟 generate_answer_standard 裡 `f'<image>\n{q}'` 的慣例一致：
    # 先放 placeholder，再用 replace 換成真正的 image token block，
    # 確保 image token 前後的文字排版跟官方路徑一模一樣。
    template.append_message(template.roles[0], f'<image>\n{question}')
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    query = query.replace('<image>', image_tokens, 1)

    return query


def split_prompt_at_vision(prompt: str, tokenizer):
    """
    把 prompt 在 <img>…</img> 區段切成三份：
        text_before : <img> 之前的文字
        vision_span : 整個 <img>...</img> 字串（僅用於計算，不 tokenize）
        text_after  : </img> 之後的文字
    """
    img_start = prompt.find(IMG_START_TOKEN)
    img_end   = prompt.find(IMG_END_TOKEN) + len(IMG_END_TOKEN)
 
    text_before = prompt[:img_start]
    text_after  = prompt[img_end:]
    return text_before, text_after


# ──────────────────────────────────────────────────────────────────────────────
# 串流核心：單一 tile 的 ViT → pixel_shuffle → mlp1
# ──────────────────────────────────────────────────────────────────────────────
def encode_tile(model, tile_pixel_values: torch.Tensor, dtype) -> torch.Tensor:
    """
    輸入  : tile_pixel_values [B_tile, 3, 448, 448]  (B_tile 通常 = vit_micro_batch)
    輸出  : [B_tile, num_image_token, D_llm]  (已經過 pixel_shuffle + mlp1，可直接餵 LLM)
 
    這個函式刻意保持「無狀態、單一 tile-batch 進、單一 tile-batch 出」，
    是讓 ViT 編碼能跟 chunked LLM prefill 交錯執行的關鍵：呼叫方可以一次只
    處理一小批 tile，處理完立刻丟去做 LLM prefill，不需要等其他 tile。
    這個特性被 InternVLAdapter.encode_and_bank() 完整保留（tile 之間彼此
    獨立，不像 LLaVA-OV 的 pack_image_features 需要整張圖的 patch 都到齊）。
    """
    select_layer = model.select_layer  # 通常 = -1 或某個負數
 
    with torch.no_grad():
        if select_layer == -1:
            vit_out = model.vision_model(
                pixel_values=tile_pixel_values,
                output_hidden_states=False,
                return_dict=True,
            )
            feats = vit_out.last_hidden_state
        else:
            vit_out = model.vision_model(
                pixel_values=tile_pixel_values,
                output_hidden_states=True,
                return_dict=True,
            )
            feats = vit_out.hidden_states[select_layer]
 
        feats = feats[:, 1:, :]   # 去 CLS token → [B_tile, HW, D_vit]
        del vit_out
 
        B_tile = feats.shape[0]
        h = w = int(feats.shape[1] ** 0.5)   # 32 for 448px tile
        feats_hw  = feats.reshape(B_tile, h, w, -1)
        shuffled  = model.pixel_shuffle(feats_hw, scale_factor=model.downsample_ratio)
        shuffled  = shuffled.reshape(B_tile, -1, shuffled.shape[-1])
        tile_tokens = model.mlp1(shuffled)   # [B_tile, num_image_token, D_llm]
 
        del feats, feats_hw, shuffled
 
    return tile_tokens.to(dtype)
