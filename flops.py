"""
TTFT 階段（ViT + LLM prefill 到第一個 token）的解析 FLOPs，給 bench_metrics.py 對每個方法用同一套算法。

算法（token pruning 文獻的慣例，例如 FastV / SparseVLM / VisionZip 報的 FLOPs）：
  每個 transformer layer 處理 n 個新 token、前面已有 o 個 token 的 context（causal）：
      FLOPs = 2 * n * P_lin  +  4 * n * c * H * d_head,   c = o + (n + 1) / 2（causal 平均可見長度）
  P_lin = 該層所有 2D 權重（q/k/v/o + MLP）的參數量，直接從模型讀，不手寫維度；
  第二項是 QK^T 與 AV 兩個矩陣乘（各 2 * n * c * H * d_head）。
  忽略：norm / RoPE / softmax / 打分時多手算的少數 attention 列 / lm_head（只算一個位置）/ decode。
  ViT 每塊 tile：n = (448/14)^2 + 1（CLS）個 token、o = 0、非 causal（c = n），再加 mlp1（pixel shuffle 後 256 個 token）。

每個方法把自己的「每段 forward 跑幾層、幾個 token、前面多長」描述成 segments = [(層數, n, o), ...]
（同一段的每層長度相同），由 FlopCounter 加總。各方法的 segments 見 *_segments() 函式。
"""
from __future__ import annotations

import torch


def _lin_params(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.dim() == 2)


class FlopCounter:
    """從 InternVL 模型讀出 LLM / ViT 的層參數量與 attention 維度。小模型（IF-Prune 的 guide）也能用。"""

    def __init__(self, model):
        lm = model.language_model.model
        cfg = model.language_model.config
        self.n_layers = len(lm.layers)
        self.p_layer = _lin_params(lm.layers[0])
        self.heads = cfg.num_attention_heads
        self.head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads

        vit = model.vision_model
        vcfg = vit.config
        self.v_layers = len(vit.encoder.layers)
        self.v_p_layer = _lin_params(vit.encoder.layers[0])
        self.v_heads = vcfg.num_attention_heads
        self.v_head_dim = vcfg.hidden_size // vcfg.num_attention_heads
        self.v_tokens = (vcfg.image_size // vcfg.patch_size) ** 2 + 1
        self.mlp1_p = _lin_params(model.mlp1)
        self.num_image_token = model.num_image_token

    def llm(self, segments) -> float:
        total = 0.0
        for layers, n, o in segments:
            if layers <= 0 or n <= 0:
                continue
            c = o + (n + 1) / 2
            total += layers * (2 * n * self.p_layer + 4 * n * c * self.heads * self.head_dim)
        return total

    def vit(self, n_tiles: int) -> float:
        n = self.v_tokens
        per_tile = self.v_layers * (2 * n * self.v_p_layer + 4 * n * n * self.v_heads * self.v_head_dim)
        per_tile += 2 * self.num_image_token * self.mlp1_p
        return n_tiles * per_tile


# ── 各方法的 segments（tb / ta = image 前 / 後的文字 token 數）──────────────────────
def full_segments(fc: FlopCounter, tb: int, n_vis: int, ta: int):
    """uncompressed / VisionZip（token 在進 LLM 前就固定）：所有層同一個長度。"""
    return [(fc.n_layers, tb + n_vis + ta, 0)]


def one_shot_prune_segments(fc: FlopCounter, tb: int, n_vis: int, ta: int, keep: int, layer: int):
    """FastV / IF-Prune：前 layer 層用完整長度，第 layer 層起只剩 keep 個 vision token。"""
    return [(layer, tb + n_vis + ta, 0), (fc.n_layers - layer, tb + min(keep, n_vis) + ta, 0)]


def sparsevlm_segments(fc: FlopCounter, tb: int, n_vis: int, ta: int, stage_logs: list, final_vis: int):
    """SparseVLM：pruning 層本身用剪枝前的長度跑完，輸出才變短；最後一個 stage 可能再裁切到 final_vis。"""
    segs, start, vis = [], 0, n_vis
    for i, st in enumerate(stage_logs):
        layer = int(st["layer"])
        segs.append((layer + 1 - start, tb + vis + ta, 0))
        start = layer + 1
        vis = int(st.get("vis_len_after", vis))
        if i == len(stage_logs) - 1:
            vis = min(vis, final_vis)
    segs.append((fc.n_layers - start, tb + vis + ta, 0))
    return segs


def foveatile_segments(fc: FlopCounter, tb: int, ta: int, stats: dict, prior_layers: int, qattn_layers: int):
    """FoveaTile：text_before prefill → prior 短 forward（到第 prior_layers 層）→ 每塊邊界 tile 的 q_attn
    （[tile][text_after] 接在 [text_before][覆蓋 tile] 前綴後，跑 qattn_layers 層）→ 寫 KV（全部層）→ text_after。"""
    m = int(stats["coverage_tiles"])
    n_img = fc.num_image_token
    segs = [(fc.n_layers, tb, 0)]
    if stats.get("prior_forward"):
        segs.append((prior_layers, tb + m * n_img + ta, 0))
    for _ in range(int(stats.get("qattn_scored", 0))):
        segs.append((qattn_layers, n_img + ta, tb + m * n_img))
    segs.append((fc.n_layers, int(stats["final_size"]), tb))
    segs.append((fc.n_layers, ta, tb + int(stats["final_size"])))
    return segs
