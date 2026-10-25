"""
獨立、不改動任何既有檔案的「IF-Prune × InternVL3.5」執行腳本。

參考：
  - 論文：Sun et al., "IF-Prune: Information-Flow Guided Token Pruning for
    Efficient Vision-Language Models", CVPR 2026（CVF open access PDF）
  - 官方原始碼：https://github.com/snap-research/EVLM-IF-Prune（commit d17c2af）
    實際 clone 下來讀完才寫這個檔案，剪枝『決策』與『小模型怎麼算 importance』
    逐行對照官方碼，行號如下（internvl_kl/ 底下）：

      (a) internvl/model/internvl_chat/modeling_internvl_chat.py
          - L108~118, L133~134：importance head 的結構 `mean_logvar_lgkld` =
            LayerNorm(C) → Linear(C,C) → GELU → Linear(C, C+1)，以及可學習
            的 prior_mu（初值 0）/ prior_logvar（初值 1），各 [C]。
          - L276~290（forward）：拿小 VLM『最後一層 hidden state』（post-norm）
            在視覺 token 位置的向量 V'，過 head 得 (mean_weight[C], logvar[1])，
                mean_weight = sigmoid(mean_weight)
                token_mean  = mean_weight * (V' - prior_mu) + prior_mu     # 論文 Eq.3
          - L402~408（cal_token_entropy）：
                kld = prior_logvar/2 - logvar/2
                      + (exp(logvar) + (mean - prior_mu)^2) / (2*exp(prior_logvar)) - 1/2
                importance = kld.mean(dim=-1)                              # 論文 Eq.2
            這個 [M] 向量（M = 視覺 token 總數）就是剪枝分數，越大越保留。

      (b) internvl/train/dataset.py L728~：preprocess_internvl2_5(question_first=True)
          小 VLM 的輸入是『question 在前、image 在後』：
              <|im_start|>system\n{system}<|im_end|>\n
              <|im_start|>user\n{question}\n<img>{IMG_CONTEXT*N}</img><|im_end|>\n
          （不含 assistant 起始 token）。question 在前才能讓 causal attention 下
          的視覺 token 看得到 query（論文 3.1 節「V' 融合 query」）。

      (c) eval_pruning/eval/vqa/evaluate_vqa_our.py L405~457：推論流程。小 VLM『只做
          一次 forward』（不生成答案）得到 importance，傳給大模型；大模型
          topk(int(vis_len * keep_ratio)) 保留。

      (d) internvl/model/qwen2/modeling_qwen2.py L940~991（大模型端）：在
          idx == K 那一層、prefill 時：
              keep_indexs = cat(arange(vis_start), topk_idx + vis_start,
                                arange(vis_end+1, seq_len)).sort()
              hidden_states = hidden_states[:, keep_indexs]
              position_ids  = keep_indexs.unsqueeze(0)     # 保留剪枝前的絕對位置
          K 是『第 K 層開始就只有保留的 token』，K=0 代表進第一層之前就剪。
          decode 階段被剪層的 KV cache 缺少被丟的 token，新 token 的 position 仍是
          絕對位置。這跟 fastv_internvl.py 的 inplace 版語意完全一樣，所以 decode
          step 直接 import fastv_decode_step()（吃『所有層共用的絕對位置計數器』）。

  論文的訓練設定（小 VLM = InternVL2.5-1B 凍結 + LoRA(r=32) + 兩層 MLP head +
  可學 prior，1 epoch）本腳本『不重新訓練』，直接載官方釋出的 checkpoint
  ZachSun/internvl2_5_1b_pruning_kl_lora32（README 指定的那份）。

【必須如實記錄的跟官方/論文的落差】
  1) 論文實驗的大模型是 InternVL2-26B / InternVL2-8B（論文 Table 1/2）；官方只釋出
     InternVL2.5-1B 的小模型 checkpoint。這裡的大模型是 InternVL3.5-8B（Qwen3 LLM），
     論文沒有測過。小模型能對得上的前提只有『每個 tile 的視覺 token 數與順序一致』：
     兩者都是 448px tile、downsample 0.5 → 256 token/tile，tile 排序也相同，所以
     importance [256*num_tiles] 逐 token 對得起來（腳本會 assert）。「小模型換一個
     大模型是否還準」是論文 4.4 節（One Can Serve Many，2.5-1B → 2-8B）在同
     系列內驗證過的事，換到 3.5-8B 是外推，請當成你實驗的一個假設看待。
  2) 論文/官方以 ratio K% 指定保留比例（keep = int(vis_len * K)）；你的其他方法都用
     絕對 budget（最終保留的視覺 token 數），這裡 keep_count = min(budget, vis_len)，
     等價於 K = budget / vis_len，好跟 stream_b<K> / fastv_b<K> 同基準比較。
  3) 官方沒有 KV cache 相容的多 token 生成之外的額外工程（HF generate 走 attention
     mask 裁切）。這裡的 prefill / decode 沿用 fastv_internvl.py 已經驗證過的
     『手動逐層 + 各層 KV 長度可以不同』做法，只是把 FastV 的「上一層 attention
     決定 keep」換成「小 VLM 的 KL importance 決定 keep」。
  4) 官方是 flash-attn + bf16，KL head 也在 bf16 算，這裡『預設照官方』（--head_dtype bf16）。
     --head_dtype fp32 是『偏離官方』的選項（公式相同、只有精度不同），僅供 ablation：
     實測 importance 分布極度雙峰（一張圖上只有約前 2~3% token 明顯偏離 prior，其餘 ~1e-4），
     KL 公式的相消項在 bf16 下的誤差（~1e-3）大於 bulk 內的真實差距，所以官方數值在
     前 ~3% 之後的排序是 bf16 雜訊；比較大的 budget（保留比例 >3%）時，多出來的 token 是
     雜訊排序選的。與官方 repo 實跑輸出比對：前 1~2% 完全相同、前 3% 98% 重疊。
     LoRA 用自己的輕量 wrapper（y = base(x) + scale * B(A(x))，scale = alpha/r = 2，
     官方 wrap_llm_lora 的 alpha = 2*r）保持跟 peft 相同的『分離計算』，不需要安裝 peft；
     已對照 peft 官方式路徑驗證 importance 一致（差異在 bf16 雜訊內）。
  5) 本腳本沿用你其他 *_internvl.py 的 prompt 組法（build_prompt + split_prompt_at_vision，
     <img>/</img> 兩個 token 不進序列），大模型端跟 fastv_internvl.py 完全同一條路，
     才能公平比較；小模型端則照官方 (b) 的格式（含 <img></img>）。

用法範例：
    python ifprune_internvl.py \
        --image img_datasets/img01.jpg --budget 512 --prune_layer 2 \
        --run_standard --run_ifprune --save

    # 小模型／base 預設從 HF 下載；也可以指本地路徑
    #   --small_checkpoint /path/to/internvl2_5_1b_pruning_kl_lora32
    #   --small_base       /path/to/InternVL2_5-1B
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time

import torch
from process_common import get_image
import torch.nn as nn
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.masking_utils import create_causal_mask

from internvl_stream import (
    MAX_NEW_TOKENS, IMG_START_TOKEN, IMG_END_TOKEN, IMG_CONTEXT_TOKEN,
    build_prompt, clean_answer, generate_answer_standard,
    split_prompt_at_vision, encode_tile, build_model, load_image_tiles,
    DEVICE, measure_peak_memory,
    DOCVQA_PROMPT, INFOVQA_PROMPT, HRBENCH_PROMPT, HRBENCH_SPLIT,
)
from sparsevlm_internvl import (
    _set_attn_implementation, _get_eos_token_id,
    estimate_kv_cache_mb, estimate_prompt_token_lengths,
    _segment_geometry,
)
# decode step 的語意跟 FastV inplace 版完全一致（各層 KV 長度不同、position 用『所有層
# 共用的絕對位置計數器』），見 fastv_internvl.fastv_decode_step() docstring，直接沿用。
from fastv_internvl import fastv_decode_step

DEFAULT_SMALL_CKPT = "ZachSun/internvl2_5_1b_pruning_kl_lora32"   # 官方 README 指定
DEFAULT_SMALL_BASE = "OpenGVLab/InternVL2_5-1B"                    # checkpoint 的 _name_or_path
LORA_SCALE = 2.0   # 官方 wrap_llm_lora(r, lora_alpha=2*r)（modeling_internvl_chat.py L127 附近）→ alpha/r = 2


# ──────────────────────────────────────────────────────────────────────────
# 小 VLM（guide）：InternVL2.5-1B + LoRA + KL importance head
# ──────────────────────────────────────────────────────────────────────────
class LoRALinear(nn.Module):
    """peft LoRA Linear 的 eval 版（dropout 在 eval 是 identity）：
    result = base(x) + lora_B(lora_A(x.to(A.dtype))) * scaling。"""

    def __init__(self, base: nn.Linear, A: torch.Tensor, B: torch.Tensor, scaling: float):
        super().__init__()
        self.base_layer = base
        self.lora_A = nn.Parameter(A, requires_grad=False)   # [r, in]
        self.lora_B = nn.Parameter(B, requires_grad=False)   # [out, r]
        self.scaling = scaling

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.base_layer(x)
        x = x.to(self.lora_A.dtype)
        return result + (x @ self.lora_A.t() @ self.lora_B.t()) * self.scaling


class KLImportanceHead(nn.Module):
    """官方 modeling_internvl_chat.py 的 `mean_logvar_lgkld` / `prior_mu` / `prior_logvar`。
    參數名跟 checkpoint 完全一樣，才能直接 load_state_dict。"""

    def __init__(self, hidden_size: int):
        super().__init__()
        C = hidden_size
        self.mean_logvar_lgkld = nn.Sequential(
            nn.LayerNorm(C), nn.Linear(C, C), nn.GELU(), nn.Linear(C, C + 1),
        )
        self.prior_mu = nn.Parameter(torch.zeros(C))
        self.prior_logvar = nn.Parameter(torch.ones(C))

    def forward(self, v_prime: torch.Tensor) -> torch.Tensor:
        """v_prime: [M, C]（小 VLM 最後一層 post-norm hidden state 在視覺 token 位置的向量）
        → importance [M] = 每個 token 的 KL(Q(z|V') || P(z))，channel 平均。"""
        C = v_prime.shape[-1]
        out = self.mean_logvar_lgkld(v_prime)
        mean_weight, logvar = out.split([C, 1], dim=-1)
        mean_weight = torch.sigmoid(mean_weight)
        mean = mean_weight * (v_prime - self.prior_mu) + self.prior_mu          # 論文 Eq.3
        kld = (self.prior_logvar / 2 - logvar / 2
               + (torch.exp(logvar) + (mean - self.prior_mu).pow(2)) / (2 * torch.exp(self.prior_logvar))
               - 0.5)                                                           # cal_token_entropy
        return kld.mean(dim=-1)


class IFPruneGuide:
    """小 VLM + head 的打包，對外只有 importance(pixel_values, question)。"""

    def __init__(self, tokenizer, model, head, device, dtype, head_dtype):
        self.tokenizer = tokenizer
        self.model = model
        self.head = head
        self.device = device
        self.dtype = dtype
        self.head_dtype = head_dtype
        self.img_start_id = tokenizer.convert_tokens_to_ids(IMG_START_TOKEN)
        self.img_end_id = tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        self.img_context_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        model.img_context_token_id = self.img_context_id

    def offload(self):
        """把小 VLM 移到 CPU。bench_metrics.py 用：同一個 process 裡還要跑 uncompressed / stream /
        fastv 等其他方法，常駐的小模型（~2GB）不該算進它們的 peak memory；只在 IF-Prune 區段
        onload() 回 GPU，peak 才只算在 IF-Prune 頭上（部署時小模型本來就常駐，這樣才公平）。"""
        self.model.to("cpu")
        self.head.to("cpu")
        torch.cuda.empty_cache()

    def onload(self):
        self.model.to(self.device)
        self.head.to(self.device)

    def _build_input_ids(self, question: str, num_tokens: int) -> torch.Tensor:
        """官方 preprocess_internvl2_5(question_first=True) 的序列，
        逐段 tokenize 再接起來（官方也是逐 turn tokenize 後 concatenate）。
        <img>/<IMG_CONTEXT>/</img> 都是 special token，直接用 id 拼，跟 tokenize
        整段字串等價，但不用對上萬個 IMG_CONTEXT 跑慢速 tokenizer。"""
        tok = self.tokenizer
        text = question.replace("<image>\n", "").replace("<image>", "")
        head_text = (f"<|im_start|>system\n{self.model.system_message}<|im_end|>\n"
                     f"<|im_start|>user\n{text}\n")
        tail_text = "<|im_end|>\n"
        head_ids = tok(head_text, return_tensors="pt").input_ids[0].tolist()
        tail_ids = tok(tail_text, return_tensors="pt").input_ids[0].tolist()
        ids = (head_ids + [self.img_start_id] + [self.img_context_id] * num_tokens
               + [self.img_end_id] + tail_ids)
        return torch.tensor([ids], device=self.device)

    @torch.no_grad()
    def importance(self, pixel_values: torch.Tensor, question: str, vit_batch: int = 4) -> torch.Tensor:
        m = self.model
        num_tiles = pixel_values.shape[0]
        vit = torch.cat([
            m.extract_feature(pixel_values[i:i + vit_batch].to(self.device, dtype=self.dtype))
            for i in range(0, num_tiles, vit_batch)
        ], dim=0)                                                   # [T, n_img_tok, C]
        C = vit.shape[-1]
        vit = vit.reshape(-1, C)

        input_ids = self._build_input_ids(question, vit.shape[0])
        embeds = m.language_model.get_input_embeddings()(input_ids)
        selected = input_ids[0] == self.img_context_id
        assert int(selected.sum()) == vit.shape[0]
        embeds[0, selected] = vit.to(embeds.dtype)

        # 官方 forward 取的是 outputs.hidden_states[-1]；HF/官方 Qwen2Model 的最後一個 hidden
        # state 是 final norm 之後，等於 Qwen2Model 回傳的 last_hidden_state。
        hidden = m.language_model.model(inputs_embeds=embeds, use_cache=False).last_hidden_state
        return self.head(hidden[0, selected].to(self.head_dtype))   # [M]


def load_ifprune_guide(
    ckpt: str = DEFAULT_SMALL_CKPT, base: str = DEFAULT_SMALL_BASE,
    dtype: torch.dtype = torch.bfloat16, device=DEVICE,
    head_dtype: torch.dtype = torch.bfloat16,
) -> IFPruneGuide:
    """用 base（InternVL2.5-1B 的官方 HF remote code）當骨架，再把官方 IF-Prune
    checkpoint 的權重灌進去：ViT / mlp1 / LLM base / LoRA / KL head。"""
    print(f"[IF-Prune] loading small VLM skeleton from {base} ...")
    tokenizer = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True, use_fast=False)
    small = AutoModel.from_pretrained(
        base, dtype=dtype, trust_remote_code=True, use_flash_attn=True, low_cpu_mem_usage=True,
    ).eval()

    print(f"[IF-Prune] loading IF-Prune weights from {ckpt} ...")
    path = ckpt if os.path.isfile(ckpt) else (
        os.path.join(ckpt, "model.safetensors") if os.path.isdir(ckpt)
        else hf_hub_download(ckpt, "model.safetensors"))
    sd = load_file(path)

    lm_prefix = "language_model.base_model.model."
    mapped, head_sd = {}, {}
    for k, v in sd.items():
        if k.startswith(lm_prefix):
            if ".lora_A." in k or ".lora_B." in k:
                continue
            mapped["language_model." + k[len(lm_prefix):].replace(".base_layer.", ".")] = v
        elif k.startswith(("mean_logvar_lgkld.", "prior_mu", "prior_logvar")):
            head_sd[k] = v
        else:
            mapped[k] = v                                           # vision_model.* / mlp1.*
    missing, unexpected = small.load_state_dict(mapped, strict=False)
    missing = [k for k in missing if "rotary" not in k]
    if missing or unexpected:
        raise RuntimeError(f"IF-Prune checkpoint 與骨架對不上：missing={missing[:5]} unexpected={unexpected[:5]}")

    # LoRA：把目標 Linear 換成 LoRALinear（A/B 直接用 checkpoint 的 bf16 權重，分離計算）
    n_lora = 0
    for k in sd:
        if k.startswith(lm_prefix) and ".lora_A." in k:
            mod = k[len(lm_prefix):k.index(".lora_A.")]
            parent_name, _, attr = f"language_model.{mod}".rpartition(".")
            parent = small.get_submodule(parent_name)
            A = sd[k].to(dtype)
            B = sd[k.replace(".lora_A.", ".lora_B.")].to(dtype)
            setattr(parent, attr, LoRALinear(getattr(parent, attr), A, B, LORA_SCALE))
            n_lora += 1
    print(f"[IF-Prune] attached {n_lora} LoRA modules (scale={LORA_SCALE:g}, unmerged)")

    head = KLImportanceHead(small.config.llm_config.hidden_size)
    head.load_state_dict(head_sd, strict=True)

    small = small.to(device)
    head = head.to(device=device, dtype=head_dtype).eval()   # bf16 權重 → fp32 是精確的 upcast
    return IFPruneGuide(tokenizer, small, head, device, dtype, head_dtype)


# ──────────────────────────────────────────────────────────────────────────
# 剪枝決策 + 大模型 prefill
# ──────────────────────────────────────────────────────────────────────────
def ifprune_select_keep_indices(
    importance: torch.Tensor,   # [vis_len]：小 VLM 的 KL importance
    vis_start: int, vis_len: int, total_len: int, keep_count: int, device,
) -> torch.Tensor:
    """對應官方 modeling_qwen2.py L974~978：
        topk(int(vis_len*keep_ratio)).indices + vis_start；
        cat(prefix, kept, suffix).sort()。"""
    assert importance.shape[0] == vis_len, (
        f"小模型 importance 長度 {importance.shape[0]} != 大模型視覺 token 數 {vis_len}："
        "小/大模型每個 tile 的 token 數或 tile 順序對不上")
    keep_count = max(0, min(vis_len, keep_count))
    top_idx = importance.topk(keep_count).indices + vis_start
    keep = torch.cat([
        torch.arange(vis_start, device=device),
        top_idx.to(device),
        torch.arange(vis_start + vis_len, total_len, device=device),
    ])
    return keep.sort().values


def ifprune_prefill(model, inputs_embeds: torch.Tensor, keep_indices: torch.Tensor, prune_layer: int):
    """手動逐層跑 Qwen3Model 的 decoder layers（跟 fastv_prefill 同一招），在
    layer_idx == prune_layer 把序列縮短成 keep_indices，之後所有層沿用縮短後長度。
    prune_layer=0 ＝ 進第一層前就剪（官方 K=0），整個 LLM 都只看保留的 token。

    重點：IF-Prune 的 importance 來自小 VLM，跟大模型任何一層的 attention 無關，所以
    這裡完全不需要 output_attentions / 手算 attention（論文強調的 FlashAttention 相容）。
    回傳 (hidden_states, past_key_values)。"""
    lm_model = model.language_model.model  # Qwen3Model
    num_layers = len(lm_model.layers)
    if not 0 <= prune_layer < num_layers:
        raise ValueError(f"prune_layer 必須在 [0, {num_layers - 1}]，收到 {prune_layer}")

    device = inputs_embeds.device
    past_key_values = DynamicCache(config=lm_model.config)

    hidden_states = inputs_embeds
    position_ids, cache_position = _segment_geometry(hidden_states, device)
    causal_mask = create_causal_mask(
        config=lm_model.config, input_embeds=hidden_states, attention_mask=None,
        cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
    )
    position_embeddings = lm_model.rotary_emb(hidden_states, position_ids)

    for layer_idx, decoder_layer in enumerate(lm_model.layers):
        if layer_idx == prune_layer:
            hidden_states = hidden_states[:, keep_indices, :]
            # position_ids 保留剪枝前的絕對位置（官方 `position_ids = keep_indexs.unsqueeze(0)`）；
            # cache_position 只是 DynamicCache 的 append 順序，用連續 0..new_len-1 即可
            # （理由同 fastv_prefill 註解）。
            cache_position = torch.arange(hidden_states.shape[1], device=device)
            position_ids = keep_indices.unsqueeze(0)
            causal_mask = create_causal_mask(
                config=lm_model.config, input_embeds=hidden_states, attention_mask=None,
                cache_position=cache_position, past_key_values=past_key_values, position_ids=position_ids,
            )
            position_embeddings = lm_model.rotary_emb(hidden_states, position_ids)

        hidden_states = decoder_layer(
            hidden_states, attention_mask=causal_mask, position_ids=position_ids,
            past_key_values=past_key_values, use_cache=True,
            cache_position=cache_position, position_embeddings=position_embeddings,
        )

    hidden_states = lm_model.norm(hidden_states)
    return hidden_states, past_key_values


# ──────────────────────────────────────────────────────────────────────────
# 頂層：小 VLM importance → 大模型 ViT → prefill(剪枝) → greedy decode
# ──────────────────────────────────────────────────────────────────────────
@torch.no_grad()
def generate_ifprune(
    model, tokenizer, pixel_values: torch.Tensor, question: str, guide: IFPruneGuide,
    budget: int, prune_layer: int = 2,
    vit_batch: int = 4, max_new_tokens: int = MAX_NEW_TOKENS,
):
    """budget = 剪枝後保留的視覺 token 數（等價於官方 keep_ratio = budget / vis_len），
    final_vis_tokens 恆等於 min(budget, raw_vis_tokens)。"""
    device = DEVICE
    dtype = model.dtype
    num_tiles = pixel_values.shape[0]

    # ── 1. 小 VLM 單次 forward → 每個視覺 token 的 KL importance ──
    torch.cuda.synchronize(device)
    t_guide = time.time()
    importance = guide.importance(pixel_values, question, vit_batch=vit_batch)
    torch.cuda.synchronize(device)
    guide_s = time.time() - t_guide

    # ── 2. 大模型：組 prompt、ViT ──
    prompt = build_prompt(tokenizer, model, question, num_tiles)
    text_before, text_after = split_prompt_at_vision(prompt, tokenizer)

    embed_fn = model.language_model.get_input_embeddings()
    ids_before = tokenizer(text_before, return_tensors="pt").input_ids.to(device)
    ids_after = tokenizer(text_after, return_tensors="pt").input_ids.to(device)
    embeds_before = embed_fn(ids_before)
    embeds_after = embed_fn(ids_after)

    vision_chunks = []
    for i in range(0, num_tiles, vit_batch):
        chunk = pixel_values[i:i + vit_batch].to(device, dtype=dtype)
        vision_chunks.append(encode_tile(model, chunk, dtype=dtype))
        del chunk
    vision_tokens = torch.cat(vision_chunks, dim=0).reshape(1, -1, embeds_before.shape[-1])

    inputs_embeds = torch.cat([embeds_before, vision_tokens, embeds_after], dim=1)
    vis_start = embeds_before.shape[1]
    vis_len = vision_tokens.shape[1]
    total_len = inputs_embeds.shape[1]

    # ── 3. 剪枝決策 + prefill ──
    keep_indices = ifprune_select_keep_indices(
        importance, vis_start, vis_len, total_len, budget, device)
    final_vis_len = min(budget, vis_len)
    del importance

    hidden_states, past_key_values = ifprune_prefill(model, inputs_embeds, keep_indices, prune_layer)

    num_layers = model.language_model.config.num_hidden_layers
    seq_lens_per_layer = [past_key_values.get_seq_length(i) for i in range(num_layers)]
    kv_cache_mb = estimate_kv_cache_mb(model, seq_lens_per_layer)

    # ── 4. greedy decode（跟 baseline 的 do_sample=False 對齊）──
    logits = model.language_model.lm_head(hidden_states[:, -1, :])
    next_id = logits.argmax(dim=-1)

    eos_id = _get_eos_token_id(tokenizer, model)
    generated = [next_id.item()]
    next_position = total_len   # 所有層共用的絕對位置計數器（見 fastv_decode_step docstring）
    for _ in range(max_new_tokens - 1):
        if generated[-1] == eos_id:
            break
        next_embed = embed_fn(next_id.view(1, 1))
        logits = fastv_decode_step(model, next_embed, past_key_values, next_position, device)
        next_id = logits[:, -1, :].argmax(dim=-1)
        generated.append(next_id.item())
        next_position += 1

    answer = tokenizer.decode(generated, skip_special_tokens=True)
    stats = {
        "raw_vis_tokens": vis_len,
        "final_vis_tokens": final_vis_len,
        "budget": budget,
        "prune_layer": prune_layer,
        "compression_ratio": round(vis_len / max(final_vis_len, 1), 3),
        "guide_s": round(guide_s, 4),
        "kv_cache_mb": round(kv_cache_mb, 2),
    }
    return answer, stats


# ──────────────────────────────────────────────────────────────────────────
# CLI：跟 fastv_internvl.py 同款輸出格式，方便沿用既有評測腳本
# ──────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--small_checkpoint", type=str, default=DEFAULT_SMALL_CKPT,
                        help="官方 IF-Prune 小 VLM checkpoint（HF repo id 或本地資料夾/safetensors）")
    parser.add_argument("--small_base", type=str, default=DEFAULT_SMALL_BASE,
                        help="小 VLM 骨架（InternVL2.5-1B 官方 HF remote code）")
    parser.add_argument("--head_dtype", choices=["fp32", "bf16"], default="bf16",
                        help="KL head 的計算精度；bf16 = 官方數值（預設），fp32 = 偏離官方的 ablation（見檔頭第 4 點）")
    parser.add_argument("--max_num", type=int, default=48, help="dynamic tile 上限（含縮圖），跟 internvl_stream.py 同語意")
    parser.add_argument("--vit_batch", type=int, default=4)
    parser.add_argument("--budget", type=int, default=512,
                        help="剪枝後保留的視覺 token 數（= 官方 keep_ratio * vis_len）")
    parser.add_argument("--prune_layer", type=int, default=2,
                        help="官方 large_model_prune_layer 換算後的層 index K（0-indexed）："
                             "第 K 層起只剩保留的 token。論文主表 L=0 與 L=2 都有報，"
                             "Table 3 的 5%% 設定用 L=2")
    parser.add_argument("--max_new_tokens", type=int, default=MAX_NEW_TOKENS)

    parser.add_argument("--run_standard", action="store_true", help="跑未壓縮的 baseline（model.batch_chat）")
    parser.add_argument("--run_ifprune", action="store_true", help="跑 IF-Prune")

    parser.add_argument("--use_ds", action="store_true", help="Use HF dataset instead of local images")
    parser.add_argument("--dataset", type=str, default="hrbench", choices=["docvqa", "infovqa", "hrbench", "mme_rw_lite", "vstar", "realworldqa"],
                        help="which HF dataset to load when --use_ds is set")
    parser.add_argument("--num_images", type=int, default=None, help="Number of images/questions to inference (None means all)")
    parser.add_argument("--image", type=str, default="img_datasets/4000x6000.jpg", help="image")
    parser.add_argument("--save", action="store_true", help="save output_json")
    parser.add_argument("--output_json", type=str, default="output_results_ifprune.json")
    args = parser.parse_args()

    dtype = torch.bfloat16
    print(f"Current device      : {DEVICE}")
    print(f"Model data type     : {dtype}")

    torch.cuda.synchronize(DEVICE)
    t0 = time.time()

    # ── 1. 載入大模型 + 小 VLM ──
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=DEVICE)
    baseline_attn_impl = model.language_model.config._attn_implementation
    print(f"num_image_token per tile : {model.num_image_token}")
    print(f"Model loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")

    guide = None
    if args.run_ifprune:
        guide = load_ifprune_guide(args.small_checkpoint, args.small_base, dtype=dtype, device=DEVICE,
                                   head_dtype=torch.float32 if args.head_dtype == "fp32" else torch.bfloat16)
        assert guide.model.num_image_token == model.num_image_token, (
            f"小模型每 tile {guide.model.num_image_token} token != 大模型 {model.num_image_token}，"
            "importance 無法逐 token 對齊")
        print(f"Small VLM loaded, peak CUDA alloc: {torch.cuda.max_memory_allocated(DEVICE)/1e9:.2f} GB")

    torch.cuda.synchronize(DEVICE)
    print(f"Load model time: {time.time() - t0:.2f} s")

    # ── 2. 載入圖像與問題（跟 fastv_internvl.py 同一套 --use_ds/--dataset 選項）──
    if args.use_ds:
        from process_common import load_docvqa, load_infovqa, load_hrbench
        if args.dataset == "docvqa":
            datasets = load_docvqa(dataset="lmms-lab-encoder/DocVQA", subject="DocVQA", split="validation",
                                    num_image=args.num_images, prompt_mode=DOCVQA_PROMPT)
        elif args.dataset == "infovqa":
            # InfoVQA validation（HF split 名為 "val"）：同 internvl_stream.py
            datasets = load_infovqa(dataset="Ahren09/InfoVQA", split="val",
                                    num_image=args.num_images, prompt_mode=INFOVQA_PROMPT)
        elif args.dataset == "hrbench":
            datasets = load_hrbench(dataset="DreamMr/HR-Bench", split=HRBENCH_SPLIT,
                                     num_image=args.num_images, prompt_mode=HRBENCH_PROMPT)
        elif args.dataset in ("mme_rw_lite", "vstar", "realworldqa"):
            # 大圖、圖片延遲開檔：沿用 internvl_stream 的資料集註冊（process_common.load_mme_realworld / load_vstar）
            from internvl_stream import load_eval_dataset
            datasets = load_eval_dataset(args.dataset, args.num_images)
        else:
            return
    else:
        from process_common import load_local
        datasets = load_local(args.image, batch_size=1, num_image=args.num_images)
    print(f"Found {len(datasets)} image(s) to process")

    # ── 3. 讀取舊的 output json（如果存在），支援中斷後繼續跑 ──
    if args.save and os.path.exists(args.output_json):
        with open(args.output_json, "r", encoding="utf-8") as f:
            output_results = json.load(f)
        output_results.setdefault("references", [])
        output_results.setdefault("candidates", {})
        print(f"Resuming from existing {args.output_json} "
              f"({len(output_results.get('references', {}))} images already done)")
    else:
        output_results = {"references": [], "candidates": {}}

    if datasets and "answer" in datasets[0]:
        output_results["meta"] = [
            {"answer": d.get("answer", ""), "options": d.get("options", []),
             "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
             "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
             "answers": d.get("answers", [])}
            for k, d in enumerate(datasets)
        ]

    tag = f"ifprune_budget={args.budget}_L={args.prune_layer}"
    if tag in output_results["candidates"] and len(output_results["candidates"][tag]) == len(datasets):
        torch.cuda.synchronize(DEVICE)
        print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
        return

    # ── 4. 逐張處理：不同圖片 tile 數不同，序列長度天生不同，固定一次一張 ──
    for i, d in enumerate(datasets):
        # 斷點續跑：這張圖兩邊都已經有結果就跳過（2026-10-14 補上：之前沒做完的 JSON 重跑會從第 1 張
        # 重新 append，答案跟 meta 錯位；只動續跑邏輯，不影響方法本身）
        need_ref = args.run_standard and len(output_results["references"]) <= i
        need_cand = args.run_ifprune and len(output_results["candidates"].get(tag, [])) <= i
        if not (need_ref or need_cand):
            continue
        print(f"\n{'='*70}")
        print(f"[{i+1}/{len(datasets)}] Processing image...")
        print(f"{'='*70}")

        pixel_values = load_image_tiles(get_image(d), input_size=448, max_num=max(1, args.max_num - 1))
        print(f"  tiles = {pixel_values.shape[0]}  raw_vis_tokens = {pixel_values.shape[0] * model.num_image_token}")

        torch.cuda.reset_peak_memory_stats(DEVICE)

        baseline_kv_mb = None
        ifprune_kv_mb = None

        try:
            if need_ref:
                _set_attn_implementation(model, baseline_attn_impl)

                num_layers = model.language_model.config.num_hidden_layers
                tb_len, v_len, ta_len = estimate_prompt_token_lengths(
                    tokenizer, model, d["question"], pixel_values.shape[0])
                total_len = tb_len + v_len + ta_len
                baseline_kv_mb = estimate_kv_cache_mb(model, [total_len] * num_layers)

                print(f"\n[Standard] Running batch_chat() ...")
                with measure_peak_memory("ifprune_script_standard_generate"):
                    ref = generate_answer_standard(
                        model, tokenizer, [pixel_values], [d["question"]],
                        generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
                    )
                ref = clean_answer(ref)
                output_results["references"].append(ref[0])
                print(f"  [KV cache, prompt-only] {baseline_kv_mb:.1f} MB "
                      f"({total_len} tokens x {num_layers} layers)")
                print(f"\n[Baseline Answer]\n{ref[0]}")

            if need_cand:
                output_results["candidates"].setdefault(tag, [])

                print(f"\n[IF-Prune] small-VLM guided pruning at layer {args.prune_layer} "
                      f"(keep={args.budget}) ...")
                with measure_peak_memory("ifprune_script_ifprune_generate"):
                    answer, stats = generate_ifprune(
                        model, tokenizer, pixel_values, d["question"], guide,
                        budget=args.budget, prune_layer=args.prune_layer,
                        vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                    )
                ifprune_kv_mb = stats["kv_cache_mb"]
                print(f"  {stats['raw_vis_tokens']} -> {stats['final_vis_tokens']} tokens "
                      f"(compression={stats['compression_ratio']}x)  small-VLM guide={stats['guide_s']}s")
                print(f"  [KV cache, prompt-only] {ifprune_kv_mb:.1f} MB")
                output_results["candidates"][tag].append(answer)
                print(f"\n[IF-Prune Answer]\n{answer}")

            if baseline_kv_mb is not None and ifprune_kv_mb is not None:
                reduction = 100 * (1 - ifprune_kv_mb / baseline_kv_mb)
                print(f"\n[KV cache, prompt-only] baseline={baseline_kv_mb:.1f} MB  "
                      f"ifprune={ifprune_kv_mb:.1f} MB  ({reduction:.1f}% 減少)")

            del pixel_values

        except torch.cuda.OutOfMemoryError as e:
            print(f"\n[OOM] Failed on image {i}: {e}")
            raise

        gc.collect()
        torch.cuda.empty_cache()

        if args.save:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(output_results, f, indent=2, ensure_ascii=False)
            print(f"\n[Saved] {args.output_json} updated "
                  f"({len(output_results['references'])}/{len(datasets)} images done)")

    torch.cuda.synchronize(DEVICE)
    print(f"\nAll done. Total time: {time.time() - t0:.2f} s")
    if args.save:
        print(f"Results saved to {args.output_json}")


if __name__ == "__main__":
    main()
