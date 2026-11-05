"""觀察實驗：b1536 全部 V* 題，記錄 prior（原問題／換問題／中性問題）、全部細節 tile 的 q_attn 與實際選擇。只讀不改 repo。"""
import json, os, sys
sys.path.insert(0, os.getcwd())
import torch
import foveatile_internvl as F
from foveatile_config import default_coverage_tiles, default_qattn_tiles
from process_common import get_image
out = sys.argv[1]; B = 1536
data = list(F.load_eval_dataset("vstar"))
qtext = [d["question"] for d in data]
cur = {}
captured = {}
_orig_rows = F.prefix_attention_rows
def _rows(model, h, q_start, layers, **kw):
    r = _orig_rows(model, h, q_start, layers, **kw)
    if len(set(layers)) > 1:
        captured["rows"] = {l: v.detach().float().cpu() for l, v in r.items()}; captured["q"] = q_start
    return r
F.prefix_attention_rows = _rows
_orig_prior = F.InternVLAdapter.coverage_prior
def sal_now(m):
    q0 = captured["q"]; s0 = q0 - m * 256; lo, hi = F.attn_layers(36)[0]
    s = torch.stack([captured["rows"][l][0, :, s0:q0].mean(0) for l in range(lo, hi + 1)]).mean(0)
    return s.reshape(m, 16, 16).tolist()
def _prior(self, model, cover_tokens, cover_grid, detail_grid, text_before, text_after, capture_kv_upto=None):
    m = cover_tokens.shape[0]
    q = cur["q"]; assert q in text_after
    v = {}
    for name, alt in (("swapped", cur["other"]), ("neutral", "Describe this image.")):
        p, _ = _orig_prior(self, model, cover_tokens, cover_grid, detail_grid, text_before, text_after.replace(q, alt), capture_kv_upto=None)
        v[name] = [p[k] for k in sorted(p)]; v[name + "_sal"] = sal_now(m)
    p, kv = _orig_prior(self, model, cover_tokens, cover_grid, detail_grid, text_before, text_after, capture_kv_upto=capture_kv_upto)
    v["prior"] = [p[k] for k in sorted(p)]; v["sal"] = sal_now(m)
    cur["v"] = v
    return p, kv
F.InternVLAdapter.coverage_prior = _prior
_orig_stream = F.InternVLAdapter.stream_image_to_kv_cascade
def _stream(self, *a, **kw):
    model, px, kv, ta = kw["model"], kw["detail_px"], kw["prefix_kv"], kw["text_after"]
    with torch.no_grad():
        emb_a = self.get_input_embeddings(model)(self.tokenizer([ta], return_tensors="pt").input_ids.to(F.DEVICE))
        sc = []
        for i in range(0, px.shape[0], 8):
            t = F.encode_tile(model, px[i:i + 8].to(F.DEVICE, dtype=kw.get("dtype", torch.bfloat16)), dtype=kw.get("dtype", torch.bfloat16))
            sc += F.qattn_scores_with_prefix(model, t, kv, emb_a).tolist()
    cur["q_all"] = sc
    return _orig_stream(self, *a, **kw)
F.InternVLAdapter.stream_image_to_kv_cascade = _stream
tok, model = F.build_model("OpenGVLab/InternVL3_5-8B", dtype=torch.bfloat16, device=F.DEVICE)
ad = F.InternVLAdapter(tok, model, budget=B)
m = default_coverage_tiles(B); S = default_qattn_tiles(B, m)
recs = []
for i, d in enumerate(data):
    cur.clear(); cur["q"] = d["question"]
    # 換問題：同類別、位移 37 題（確定性），保留選項格式
    same = [j for j, x in enumerate(data) if x["category"] == d["category"]]
    cur["other"] = qtext[same[(same.index(i) + 37) % len(same)]]
    inp = F.prepare_inputs([get_image(d)], "foveatile", B, 48)[0]
    ans, stats, _ = F.run_online_kv_with_memory_bank(model, ad, [inp], [d["question"]], budget=B, qattn_tiles=S, max_new_tokens=10)
    s = stats[0]
    recs.append({"i": i, "id": d["id"], "category": d["category"], "answer": d["answer"], "pred": ans[0],
                 "detail_grid": s["detail_grid"], "cover_grid": s["cover_grid"], "prior_ranked": s["prior_ranked"],
                 "direct_kept": s["direct_kept"], "window": s["window"], "candidate_scores": s["candidate_scores"],
                 "committed": s["committed_tile_indices"], "q_all": cur["q_all"], **cur["v"]})
    if i % 20 == 0 or i == len(data) - 1:
        json.dump(recs, open(out, "w"))
json.dump(recs, open(out, "w")); print("DONE", len(recs))
