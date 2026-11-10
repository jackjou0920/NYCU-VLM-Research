"""跑 FoveaTile（v1.0 預設，b1536）在 V* direct_attributes 前 N 題，存每題的 coverage saliency、prior、選擇結果，給論文圖用。
用法（repo 根目錄）：python <this> --n 60 --out <dir>"""
import argparse, json, os, sys
sys.path.insert(0, os.getcwd())
import torch
import foveatile_internvl as F
from foveatile_config import default_coverage_tiles, default_qattn_tiles
from process_common import get_image

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=60)
ap.add_argument("--budget", type=int, default=1536)
ap.add_argument("--out", required=True)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)

# 攔截 prior forward 的 attention 列，重算 coverage patch saliency（與 coverage_prior 內完全相同的公式）
captured = {}
_orig = F.prefix_attention_rows
def _wrap(model, h, q_start, layers, **kw):
    out = _orig(model, h, q_start, layers, **kw)
    if len(set(layers)) > 1:
        captured["rows"] = {l: v.detach().float().cpu() for l, v in out.items()}
        captured["q_start"] = q_start
    return out
F.prefix_attention_rows = _wrap
_orig_prior = F.InternVLAdapter.coverage_prior
def _prior(self, model, cover_tokens, *a, **kw):
    prior, pkv = _orig_prior(self, model, cover_tokens, *a, **kw)
    m, n_patch, _ = cover_tokens.shape
    q0 = captured["q_start"]; s0 = q0 - m * n_patch
    lo, hi = F.THUMB_ATTN_LAYERS
    sal = torch.stack([captured["rows"][l][0, :, s0:q0].mean(0) for l in range(lo, hi + 1)]).mean(0)
    sal = (sal - sal.min()) / (sal.max() - sal.min() + 1e-6)
    captured["saliency"] = sal.reshape(m, 16, 16).tolist()
    captured["prior"] = [prior[k] for k in sorted(prior)]
    return prior, pkv
F.InternVLAdapter.coverage_prior = _prior

data = [d for d in F.load_eval_dataset("vstar") if d.get("category") == "direct_attributes"][: args.n]
tok, model = F.build_model("OpenGVLab/InternVL3_5-8B", dtype=torch.bfloat16, device=F.DEVICE)
adapter = F.InternVLAdapter(tok, model, budget=args.budget)
m = default_coverage_tiles(args.budget); S = default_qattn_tiles(args.budget, m)
recs = []
for i, d in enumerate(data):
    img = get_image(d)
    inp = F.prepare_inputs([img], "foveatile", args.budget, 48)[0]
    ans, stats, timing = F.run_online_kv_with_memory_bank(model, adapter, [inp], [d["question"]],
                                                          budget=args.budget, qattn_tiles=S, max_new_tokens=10)
    st = stats[0]
    recs.append({"idx": i, "id": d.get("id"), "question": d["question"], "answer": d.get("answer"), "pred": ans[0],
                 "image_size": list(img.size), "cover_grid": st["cover_grid"], "detail_grid": st["detail_grid"],
                 "prior": captured["prior"], "saliency": captured["saliency"], "prior_ranked": st["prior_ranked"],
                 "direct_kept": st["direct_kept"], "window": st["window"], "candidate_scores": st["candidate_scores"],
                 "selected": st["committed_tile_indices"], "boundary_a_b": st["boundary_a_b"]})
    json.dump(recs, open(os.path.join(args.out, "examples.json"), "w"))
print("saved", len(recs))
