"""
主張指標 harness：量「peak memory / KV cache / accuracy vs budget」，
未壓縮 / streaming(你的方法) / SparseVLM 三方對照，同一組 budget 公平比較。

論文的賣點是「省 peak memory、accuracy 不掉太多」，純 accuracy 腳本量不到記憶體，
這支獨立跑一小批圖（預設 24 張就夠，記憶體指標不需要大 n；accuracy 要準的話
建議另外用 --save 存下 output_json、拿去跑正式的 evaluate_*.py，n 要夠大）：

  uncompressed  ：batch_chat 路徑，max_new_tokens=1 → TTFT；prefill 當下的 peak
                  allocated / reserved。budget 無關（吃滿 --max_num 個 tile）。
  stream b<K>   ：run_online_kv_with_memory_bank，回傳的 timing["TTFT"] /
                  timing["Vision+OnlineKV"] / timing["_peak_mem_gb"]["OVERALL"]。
  sparsevlm b<K>：generate_sparsevlm()，預設用 auto_tune_lam() 自動校準 λ 讓
                  Eq.7 的漸進剪枝『自然』壓到接近 budget（不是靠保底硬裁切），
                  公平代表 SparseVLM 本身的機制，跟 stream_b<K> 站在同一個
                  『最終 token 數』基準上比較（--sparsevlm_lam 可以關掉自動校準，
                  改用固定 λ，代價是可能不公平代表 SparseVLM，見 sparsevlm_
                  internvl.py 的 auto_tune_lam() docstring）。
  fastv b<K>    ：generate_fastv()，budget 直接對應官方的 ATTENTION_RANK
                  （保留幾個 image token），final_vis_tokens 恆等於
                  min(budget, raw_vis_tokens)，沒有 SparseVLM 那種需要校準的
                  超參數，跟 stream_b<K>/sparsevlm_b<K> 站在同一個『最終
                  token 數』基準上比較。

  ifprune b<K>  ：generate_ifprune()，小 VLM（InternVL2.5-1B + KL head）單次 forward 給每個視覺
                  token 的 KL importance，大模型在第 --ifprune_prune_layer 層 hard-prune 到 budget
                  個 token（見 ifprune_internvl.py）。小 VLM 平常放 CPU，只在 IF-Prune 區段上 GPU，
                  所以它的 peak_alloc 含小模型權重、其他方法不含。method 命名 ifprune_b<K>。

  visionzip b<K>：generate_visionzip()，budget = 所有 tile 合計保留的視覺 token 數
                  （平均分給每個 tile，見 visionzip_internvl.py 檔頭）。token 在進 LLM 前
                  就縮短，所以每層 KV 一樣長。method 命名 visionzip_b<K>。
                  TTFT 另外用 max_new_tokens=1 量一次（含 ViT + VisionZip + prefill +
                  第一個 token，跟 stream 的 timing["TTFT"] 同一個定義）。

TTFT 定義（所有方法一致）：另外跑一次 max_new_tokens=1，量牆鐘時間（含 ViT + 方法本身
+ prefill + 第一個 token）；答案與 peak memory 則來自 --max_new_tokens 那次完整生成。
SparseVLM 的 TTFT 不含 auto_tune_lam 校準（見該處註解）。

每個 (model, dataset, method, budget) 匯總 mean±std 到一份 CSV（peak_alloc_gb /
kv_cache_mb 兩種記憶體指標都有，見 S.estimate_kv_cache_mb() docstring 為什麼
兩個都要看）。加 --save 還會另外存一份 output_json（{"references":[...],
"candidates":{tag:[...]}}），格式跟 internvl_stream.py / sparsevlm_internvl.py
一致，可以直接餵給既有的 evaluate_anls.py / evaluate_hrbench*.py /
evaluate_mmmu.py 算 accuracy——這支腳本本身不重算 accuracy（不同 dataset 的
算法不一樣，直接沿用你已經寫好、驗證過的評測腳本比自己重寫可靠）。

用法：
  python bench_metrics.py --model_name OpenGVLab/InternVL3_5-8B --dataset hrbench \
      --num_images 24 --budgets 1024,2048,3072,4096,5120 \
      --run_sparsevlm --run_fastv --save --out_json compare_hrbench.json \
      --out_csv metrics_internvl8b_hrbench4k.csv
  python evaluate_hrbench.py --input_json compare_hrbench.json --prompt_mode open \
      --baseline_tag "stream_b4096" ...   # accuracy，tag 命名見下方 candidates key

  （HR-Bench split / prompt、DocVQA prompt 沒有旗標，改 internvl_stream.py 頂端的
   HRBENCH_SPLIT / HRBENCH_PROMPT / DOCVQA_PROMPT 常數）
"""
import gc
import csv
import json
import time
import argparse
import statistics as st

import torch

from internvl_preprocess import build_model, load_image_tiles, load_image_tiles_fill
import internvl_stream as S
import sparsevlm_internvl as SV
import fastv_internvl as FV
import visionzip_internvl as VZ
import flops as FL

MODEL_LOAD_PEAK_GB = 0.0     # main() 載完模型後設定；mem_delta_gb = peak_alloc_gb - 這個值（扣掉權重）
BREAKDOWN_KEYS = ["text_prefill", "cover_vit", "prior", "detail_vit", "qattn", "kv_commit", "first_token"]


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize(S.DEVICE)


def _peak_reset():
    _sync()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(S.DEVICE)


def _peak_alloc_gb():
    _sync()
    return torch.cuda.max_memory_allocated(S.DEVICE) / 1e9 if torch.cuda.is_available() else 0.0


def _peak_resv_gb():
    return torch.cuda.max_memory_reserved(S.DEVICE) / 1e9 if torch.cuda.is_available() else 0.0


def _flops(fc, vit_tiles, segs, extra_tflops=0.0):
    """TTFT 階段的解析 FLOPs（TFLOPs），見 flops.py。extra_tflops：額外模型（IF-Prune 的小 VLM）。"""
    v, l = fc.vit(vit_tiles) / 1e12, fc.llm(segs) / 1e12
    return {"vit_tflops": v, "llm_tflops": l, "total_tflops": v + l + extra_tflops,
            "aux_tflops": extra_tflops or None}


def _agg(rows, keys):
    """rows: list[dict] → {key: (mean, std)} for the numeric keys；另外給 ttft_s 的中位數（key "ttft_s_median"）。"""
    out = {}
    for r in rows:
        if r.get("peak_alloc_gb") is not None and "mem_delta_gb" not in r:
            r["mem_delta_gb"] = r["peak_alloc_gb"] - MODEL_LOAD_PEAK_GB
    tt = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
    out["ttft_s_median"] = st.median(tt) if tt else None
    for k in keys:
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            out[k] = (None, None)
        elif len(vals) == 1:
            out[k] = (vals[0], 0.0)
        else:
            out[k] = (st.mean(vals), st.pstdev(vals))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model_name", default="OpenGVLab/InternVL3_5-8B")
    ap.add_argument("--dataset", default="hrbench_4k", choices=["hrbench"] + sorted(S.DATASETS),
                    help="同 internvl_stream.py；hrbench = 舊名稱，split 看環境變數 HRBENCH_SPLIT")
    ap.add_argument("--num_images", type=int, default=24)
    ap.add_argument("--max_num", type=int, default=48, help="dynamic tiles 上限（含縮圖）")
    ap.add_argument("--budgets", default="1024,2048,4096")
    ap.add_argument("--vit_batch", type=int, default=4)
    ap.add_argument("--max_new_tokens", type=int, default=8, help="指標只需要量到第一個 token + 幾步 decode")
    ap.add_argument("--skip_uncompressed", action="store_true")
    ap.add_argument("--lowres_fill", action="store_true",
                    help="Low-Res（fill）：uncompressed 改吃 load_image_tiles_fill（網格 <= --max_num、不加縮圖），"
                         "不跑 streaming（沒有縮圖可當 prior）")

    ap.add_argument("--run_sparsevlm", action="store_true",
                     help="每個 budget 額外跑一次 SparseVLM，跟 stream_b<K> 同一組 budget 公平比較")
    ap.add_argument("--sparsevlm_lams", default=None,
                    help="每個 budget 各自的固定 λ，例如 1024:0.5666,2048:0.3794（跟 accuracy 實驗用同一組 λ）；"
                         "沒列到的 budget 退回 --sparsevlm_lam / auto_lam")
    ap.add_argument("--sparsevlm_lam", type=float, default=None,
                     help="固定 λ，不做 auto_tune_lam 校準（預設會自動校準，關掉的話"
                          "可能不公平代表 SparseVLM，見 SV.auto_tune_lam() docstring）")
    ap.add_argument("--sparsevlm_auto_lam_lo", type=float, default=0.005)
    ap.add_argument("--sparsevlm_auto_lam_hi", type=float, default=3.0)
    ap.add_argument("--sparsevlm_auto_lam_max_evals", type=int, default=6)
    ap.add_argument("--sparsevlm_auto_lam_overshoot_tol", type=float, default=0.5)

    ap.add_argument("--run_fastv", action="store_true",
                     help="每個 budget 額外跑一次 FastV，跟 stream_b<K>/sparsevlm_b<K> 同一組 budget 公平比較")
    ap.add_argument("--fastv_agg_layer", type=int, default=2,
                     help="AGG_LAYER：第幾層之後剪枝，官方預設 2（見 fastv_internvl.py）")

    ap.add_argument("--run_ifprune", action="store_true",
                     help="每個 budget 額外跑一次 IF-Prune（小 VLM 的 KL importance 引導剪枝，見 ifprune_internvl.py）。"
                          "小 VLM 只在這個區段 onload 到 GPU，peak memory 才只算在 IF-Prune 頭上")
    ap.add_argument("--ifprune_prune_layer", type=int, default=2,
                     help="剪枝層 index（0-indexed），論文主表 L=0 與 L=2，見 ifprune_internvl.py")
    ap.add_argument("--ifprune_small_checkpoint", default="ZachSun/internvl2_5_1b_pruning_kl_lora32")
    ap.add_argument("--ifprune_small_base", default="OpenGVLab/InternVL2_5-1B")
    ap.add_argument("--ifprune_head_dtype", choices=["bf16", "fp32"], default="bf16",
                     help="bf16 = 官方數值（預設）；fp32 = 偏離官方的 ablation")

    ap.add_argument("--run_visionzip", action="store_true",
                     help="每個 budget 額外跑一次 VisionZip，跟 stream/sparsevlm/fastv 同一組 budget 公平比較")
    ap.add_argument("--visionzip_contextual_ratio", type=float, default=VZ.CONTEXTUAL_RATIO)

    ap.add_argument("--svma_m_fracs", default=None,
                    help="逗號分隔的 m/K 比例（例如 0.5,0.75,1.0），每個 budget 各量一次 m = max(1, floor(f*K))；"
                         "不給 = 只量 svma_config 的預設 m。method 命名 stream_b<B>_m<m>")
    ap.add_argument("--svma_s_mults", default=None,
                    help="逗號分隔的 S 倍率（例如 0,0.5,1,1.5,2）：m 用預設，S = floor(倍率 * 預設 S(K, m))；"
                         "可跟 --svma_m_fracs 擇一使用。method 命名 stream_b<B>_m<m>_S<S>")
    ap.add_argument("--svma_grid_fracs", default=None,
                    help="逗號分隔的 qattn_grid_frac（0 = 關閉 = 預設），m、S 用預設；method 命名 stream_b<B>_gf<f>")
    ap.add_argument("--warmup_images", type=int, default=1,
                    help="前幾張圖照跑但不算進統計（第一次呼叫含 CUDA / kernel 初始化，會拉高 TTFT）")
    ap.add_argument("--save", action="store_true", help="額外存一份 output_json（references+candidates），"
                                                          "拿去跑 evaluate_*.py 算 accuracy")
    ap.add_argument("--out_json", default="bench_metrics_answers.json")
    ap.add_argument("--out_csv", default="bench_metrics.csv")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",") if b.strip()]
    lam_map = {int(k): float(v) for k, v in (x.split(":") for x in (args.sparsevlm_lams or "").split(",") if x.strip())}
    dtype = torch.bfloat16
    print(f"device={S.DEVICE}  model={args.model_name}  dataset={args.dataset}"
          + (f":{S.HRBENCH_SPLIT}" if args.dataset == "hrbench" else ""))

    _peak_reset()
    tokenizer, model = build_model(args.model_name, dtype=dtype, device=S.DEVICE)
    model_load_peak = _peak_alloc_gb()
    global MODEL_LOAD_PEAK_GB
    MODEL_LOAD_PEAK_GB = model_load_peak
    fc = FL.FlopCounter(model)
    print(f"model loaded, peak_alloc={model_load_peak:.2f} GB  num_image_token={model.num_image_token}")

    ds = S.load_eval_dataset(args.dataset, args.num_images)
    max_num_grid = max(1, args.max_num - 1)

    output_results = {"references": [], "candidates": {}}

    fracs = None if args.svma_m_fracs is None else [float(x) for x in args.svma_m_fracs.split(",") if x.strip()]
    s_mults = None if args.svma_s_mults is None else [float(x) for x in args.svma_s_mults.split(",") if x.strip()]
    g_fracs = None if args.svma_grid_fracs is None else [float(x) for x in args.svma_grid_fracs.split(",") if x.strip()]

    def svma_ms(b):
        """這個 budget 要量的 (m, S[, grid_frac]) 列表；m / S 為 None = 用預設。"""
        if g_fracs is not None:
            return [(None, None, g) for g in g_fracs]
        if s_mults is not None:
            m = S.default_coverage_tiles(b)
            s0 = S.default_qattn_tiles(b, m)
            return [(m, sv) for sv in sorted({int(x * s0) for x in s_mults})]
        if fracs is None:
            return [(None, None)]
        return [(m, None) for m in sorted({max(1, int(f * (b // 256))) for f in fracs})]

    def stream_tag_of(b, ms_):
        m, sv = ms_[0], ms_[1]
        gf = ms_[2] if len(ms_) > 2 else None
        return (f"stream_b{b}" + ("" if m is None else f"_m{m}") + ("" if sv is None else f"_S{sv}")
                + (f"_gf{gf:g}" if gf else ""))

    unc_rows = []
    stream_rows = {stream_tag_of(b, c): [] for b in budgets for c in svma_ms(b)}
    sparsevlm_rows = {b: [] for b in budgets} if args.run_sparsevlm else {}
    fastv_rows = {b: [] for b in budgets} if args.run_fastv else {}
    vz_rows = {b: [] for b in budgets} if args.run_visionzip else {}
    ifprune_rows = {b: [] for b in budgets} if args.run_ifprune else {}

    ifprune_guide = None
    if args.run_ifprune:
        import ifprune_internvl as IP     # lazy import：沒開 --run_ifprune 時不影響其他方法
        ifprune_guide = IP.load_ifprune_guide(
            args.ifprune_small_checkpoint, args.ifprune_small_base, dtype=dtype, device=S.DEVICE,
            head_dtype=torch.float32 if args.ifprune_head_dtype == "fp32" else torch.bfloat16)
        assert ifprune_guide.model.num_image_token == model.num_image_token
        ifprune_guide.offload()           # 常駐 CPU，只在 IF-Prune 區段上 GPU
        fc_guide = FL.FlopCounter(ifprune_guide.model)

    for i, d in enumerate(ds):
        q = d["question"]
        img = S.get_image(d)
        keep = i >= args.warmup_images      # warmup 的圖照跑，但不算進統計
        pv = (load_image_tiles_fill(img, input_size=448, max_tiles=args.max_num) if args.lowres_fill
              else load_image_tiles(img, input_size=448, max_num=max_num_grid))
        n_tiles = pv.shape[0]
        n_vis = n_tiles * model.num_image_token
        tb_q, ta_q = S.estimate_prompt_token_lengths(
            tokenizer, *S.split_prompt_at_vision(S.build_prompt(tokenizer, model, q, 1), tokenizer))

        # ---- uncompressed（budget 無關）----
        if not args.skip_uncompressed:
            adapter = S.InternVLAdapter(tokenizer, model, budget=max(budgets))
            _peak_reset()
            _sync()
            t0 = time.time()
            S.generate_answer_standard(
                model, tokenizer, [pv], [q],
                generation_config=dict(max_new_tokens=1, do_sample=False),
            )
            _sync()
            unc_ttft = time.time() - t0     # 只到第一個 token，跟 stream 的 timing["TTFT"] 同一個定義
            ref = S.generate_answer_standard(
                model, tokenizer, [pv], [q],
                generation_config=dict(max_new_tokens=args.max_new_tokens, do_sample=False),
            )
            _sync()
            output_results["references"].append(S.clean_answer(ref)[0])

            # KV cache（analytical，跟 SparseVLM 論文同一種量測方式，見
            # S.estimate_kv_cache_mb() docstring）：generate_answer_standard 是
            # 黑盒的 model.batch_chat()，拿不到它內部真正的 past_key_values，
            # 用跟 S.main() baseline 那邊一樣的算法反推——沒有剪枝，所有層
            # 長度一樣（raw tile 數 * num_image_token，不是 budget）。
            prompt = S.build_prompt(tokenizer, model, q, n_tiles)
            tb, ta = S.split_prompt_at_vision(prompt, tokenizer)
            tb_len, ta_len = S.estimate_prompt_token_lengths(tokenizer, tb, ta)
            unc_total_len = tb_len + n_tiles * model.num_image_token + ta_len
            unc_kv_mb = S.estimate_kv_cache_mb(
                model, [unc_total_len] * model.language_model.config.num_hidden_layers)

            (unc_rows if keep else []).append({
                "raw_tiles": n_tiles,
                "ttft_s": unc_ttft,
                "peak_alloc_gb": _peak_alloc_gb(),
                "peak_resv_gb": _peak_resv_gb(),
                "vision_s": None,
                "kv_cache_mb": unc_kv_mb,
                **_flops(fc, n_tiles, FL.full_segments(fc, tb_q, n_vis, ta_q)),
            })
            del adapter
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- SVMA @ each budget（m、S 都由 budget 決定，跟 internvl_stream.py --mode svma 一樣）----
        for b, cfg in [(b, c) for b in ([] if args.lowres_fill else budgets) for c in svma_ms(b)]:
            m_eff = S.default_coverage_tiles(b) if cfg[0] is None else cfg[0]
            s_eff = S.default_qattn_tiles(b, m_eff) if cfg[1] is None else cfg[1]
            ms = S.prepare_inputs([img], "svma", b, args.max_num, coverage_tiles=m_eff)[0]
            stream_tag = stream_tag_of(b, cfg)
            output_results["candidates"].setdefault(stream_tag, [])
            adapter = S.InternVLAdapter(tokenizer, model, budget=b)
            _peak_reset()
            answers, all_stats, timing = S.run_online_kv_with_memory_bank(
                model, adapter, [ms], [q],
                budget=b,
                qattn_tiles=s_eff, qattn_grid_frac=(cfg[2] if len(cfg) > 2 else None),
                max_new_tokens=args.max_new_tokens, dtype=dtype,
            )
            output_results["candidates"][stream_tag].append(answers[0])
            pk = timing.get("_peak_mem_gb", {})
            (stream_rows[stream_tag] if keep else []).append({
                "raw_tiles": n_tiles,
                "ttft_s": timing.get("TTFT"),
                "peak_alloc_gb": pk.get("OVERALL", _peak_alloc_gb()),
                "peak_resv_gb": _peak_resv_gb(),
                "vision_s": timing.get("Vision+OnlineKV"),
                "kv_cache_mb": timing.get("_kv_cache_mb") or (all_stats[0].get("kv_cache_mb") if all_stats else None),
                **_flops(fc, all_stats[0]["vit_encoded_tiles"],
                         FL.svma_segments(fc, tb_q, ta_q, all_stats[0], S.THUMB_ATTN_LAYERS[1], S.QATTN_LAYER)),
                **{f"t_{k}": v for k, v in all_stats[0].get("ttft_breakdown", {}).items()},
            })
            del adapter
            gc.collect()
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- SparseVLM @ each budget（同一組 budget，才能跟 stream_b<K> 公平比較）----
        if args.run_sparsevlm:
            for b in budgets:
                sv_tag = f"sparsevlm_b{b}"
                output_results["candidates"].setdefault(sv_tag, [])

                lam = lam_map.get(b, args.sparsevlm_lam)
                if lam is None:
                    lam, _calib_log = SV.auto_tune_lam(
                        model, tokenizer, pv, q, budget=b,
                        lam_lo=args.sparsevlm_auto_lam_lo, lam_hi=args.sparsevlm_auto_lam_hi,
                        max_evals=args.sparsevlm_auto_lam_max_evals,
                        overshoot_tol=args.sparsevlm_auto_lam_overshoot_tol,
                        vit_batch=args.vit_batch,
                    )

                lam = SV.budget_lam(n_vis, b, lam)     # 原圖 token <= budget 時不剪枝（見 SV.budget_lam）
                # TTFT 不含 auto_tune_lam 校準（那是逐張調參成本；accuracy pass 用的是校準好的
                # 固定 λ，推論延遲看的是給定 λ 之後的一次 forward）。
                _peak_reset()
                _sync()
                t0 = time.time()
                SV.generate_sparsevlm(
                    model, tokenizer, pv, q, budget=b, lam=lam,
                    vit_batch=args.vit_batch, max_new_tokens=1,
                )
                _sync()
                sv_ttft = time.time() - t0

                answer, sv_stats = SV.generate_sparsevlm(
                    model, tokenizer, pv, q, budget=b, lam=lam,
                    vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                )
                output_results["candidates"][sv_tag].append(answer)
                (sparsevlm_rows[b] if keep else []).append({
                    "raw_tiles": n_tiles,
                    "ttft_s": sv_ttft,
                    "peak_alloc_gb": _peak_alloc_gb(),
                    "peak_resv_gb": _peak_resv_gb(),
                    "vision_s": None,
                    "kv_cache_mb": sv_stats["kv_cache_mb"],
                    **_flops(fc, n_tiles, FL.sparsevlm_segments(fc, tb_q, n_vis, ta_q, sv_stats["stage_logs"],
                                                                sv_stats["final_vis_tokens"])),
                })
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- FastV @ each budget（同一組 budget，才能跟 stream_b<K>/sparsevlm_b<K>
        #      公平比較）。budget 直接對應 ATTENTION_RANK，沒有校準步驟。----
        if args.run_fastv:
            for b in budgets:
                fv_tag = f"fastv_b{b}"
                output_results["candidates"].setdefault(fv_tag, [])

                _peak_reset()
                _sync()
                t0 = time.time()
                FV.generate_fastv(
                    model, tokenizer, pv, q, budget=b, agg_layer=args.fastv_agg_layer,
                    vit_batch=args.vit_batch, max_new_tokens=1,
                )
                _sync()
                fv_ttft = time.time() - t0

                answer, fv_stats = FV.generate_fastv(
                    model, tokenizer, pv, q, budget=b, agg_layer=args.fastv_agg_layer,
                    vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                )
                output_results["candidates"][fv_tag].append(answer)
                (fastv_rows[b] if keep else []).append({
                    "raw_tiles": n_tiles,
                    "ttft_s": fv_ttft,
                    "peak_alloc_gb": _peak_alloc_gb(),
                    "peak_resv_gb": _peak_resv_gb(),
                    "vision_s": None,
                    "kv_cache_mb": fv_stats["kv_cache_mb"],
                    **_flops(fc, n_tiles, FL.one_shot_prune_segments(fc, tb_q, n_vis, ta_q, b, args.fastv_agg_layer)),
                })
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # ---- IF-Prune @ each budget（同一組 budget）。小 VLM 只在這段 onload，peak 只算在
        #      IF-Prune 頭上（含常駐的小模型權重）；TTFT 含小 VLM 的一次 forward，跟其他方法
        #      同一個定義（ViT + 方法本身 + prefill + 第一個 token）。kv_cache_mb 只算大模型。----
        if args.run_ifprune:
            ifprune_guide.onload()
            for b in budgets:
                ip_tag = f"ifprune_b{b}"
                output_results["candidates"].setdefault(ip_tag, [])

                _peak_reset()
                _sync()
                t0 = time.time()
                IP.generate_ifprune(
                    model, tokenizer, pv, q, ifprune_guide, budget=b, prune_layer=args.ifprune_prune_layer,
                    vit_batch=args.vit_batch, max_new_tokens=1,
                )
                _sync()
                ip_ttft = time.time() - t0

                answer, ip_stats = IP.generate_ifprune(
                    model, tokenizer, pv, q, ifprune_guide, budget=b, prune_layer=args.ifprune_prune_layer,
                    vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                )
                output_results["candidates"][ip_tag].append(answer)
                (ifprune_rows[b] if keep else []).append({
                    "raw_tiles": n_tiles,
                    "ttft_s": ip_ttft,
                    "peak_alloc_gb": _peak_alloc_gb(),
                    "peak_resv_gb": _peak_resv_gb(),
                    "vision_s": None,
                    "kv_cache_mb": ip_stats["kv_cache_mb"],
                    # 小 VLM（guide）：自己的 ViT 跑全部 tile + 自己的 LLM 跑完整序列（prompt 長度近似用大模型的）
                    **_flops(fc, n_tiles, FL.one_shot_prune_segments(fc, tb_q, n_vis, ta_q, b, args.ifprune_prune_layer),
                             extra_tflops=(fc_guide.vit(n_tiles) + fc_guide.llm(
                                 FL.full_segments(fc_guide, tb_q, n_vis, ta_q))) / 1e12),
                })
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
            ifprune_guide.offload()

        # ---- VisionZip @ each budget（同一組 budget）。TTFT 用 max_new_tokens=1 另外量，
        #      peak 讀兩次呼叫之間沒有 reset 的最大值（TTFT 那次就是 prefill 峰值）。----
        if args.run_visionzip:
            for b in budgets:
                vz_tag = f"visionzip_b{b}"
                output_results["candidates"].setdefault(vz_tag, [])

                _peak_reset()
                _sync()
                t0 = time.time()
                VZ.generate_visionzip(
                    model, tokenizer, pv, q, budget=b, contextual_ratio=args.visionzip_contextual_ratio,
                    vit_batch=args.vit_batch, max_new_tokens=1,
                )
                _sync()
                vz_ttft = time.time() - t0

                answer, vz_stats = VZ.generate_visionzip(
                    model, tokenizer, pv, q, budget=b, contextual_ratio=args.visionzip_contextual_ratio,
                    vit_batch=args.vit_batch, max_new_tokens=args.max_new_tokens,
                )
                output_results["candidates"][vz_tag].append(answer)
                (vz_rows[b] if keep else []).append({
                    "raw_tiles": n_tiles,
                    "ttft_s": vz_ttft,
                    "peak_alloc_gb": _peak_alloc_gb(),
                    "peak_resv_gb": _peak_resv_gb(),
                    "vision_s": None,
                    "kv_cache_mb": vz_stats["kv_cache_mb"],
                    **_flops(fc, n_tiles, FL.full_segments(fc, tb_q, vz_stats["final_vis_tokens"], ta_q)),
                })
                gc.collect()
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

        if (i + 1) % 5 == 0 or i + 1 == len(ds):
            print(f"  [{i + 1}/{len(ds)}] done  (raw_tiles={n_tiles})")

    # ---- 匯總 + 寫 CSV ----
    # kv_cache_mb：analytical KV cache 大小（見 S.estimate_kv_cache_mb()
    # docstring），跟 SparseVLM/H2O/StreamingLLM 等文獻同一種量測方式，
    # 排除掉固定模型權重的稀釋，才看得出壓縮方法本身的效果——peak_alloc_gb
    # 仍然保留，兩者互補（一個是「方法本身省多少」，一個是「實際部署省多少」）。
    num_keys = ["raw_tiles", "ttft_s", "peak_alloc_gb", "peak_resv_gb", "mem_delta_gb", "vision_s", "kv_cache_mb",
                "vit_tflops", "llm_tflops", "aux_tflops", "total_tflops"] + [f"t_{k}" for k in BREAKDOWN_KEYS]
    n_measured = max(0, len(ds) - args.warmup_images)
    out_rows = []
    tag = f"model={args.model_name}"
    ds_tag = args.dataset + (f":{S.HRBENCH_SPLIT}" if args.dataset == "hrbench" else "")
    method_note = "svma m=floor(COVERAGE_FRAC*K), S=default_qattn_tiles(b, m)"

    def _row(method, agg, extra=""):
        r = {"model": args.model_name, "dataset": ds_tag, "method": method,
             "n_images": n_measured, "model_load_peak_gb": round(model_load_peak, 3),
             "ttft_s_median": None if agg.get("ttft_s_median") is None else round(agg["ttft_s_median"], 4),
             "note": (extra or method_note)}
        for k in num_keys:
            m, s = agg[k]
            r[k + "_mean"] = None if m is None else round(m, 4)
            r[k + "_std"] = None if s is None else round(s, 4)
        return r

    if unc_rows:
        out_rows.append(_row("uncompressed", _agg(unc_rows, num_keys), extra="max_num tiles, budget-independent"))
    for b in budgets:
        for cfg in svma_ms(b):
            t = stream_tag_of(b, cfg)
            if stream_rows[t]:
                m_eff = S.default_coverage_tiles(b) if cfg[0] is None else cfg[0]
                s_eff = S.default_qattn_tiles(b, m_eff) if cfg[1] is None else cfg[1]
                out_rows.append(_row(t, _agg(stream_rows[t], num_keys), extra=f"svma m={m_eff}, S={s_eff}"))
        if args.run_sparsevlm:
            lb = lam_map.get(b, args.sparsevlm_lam)
            sv_note = "auto_lam" if lb is None else f"lam={lb:g}"
            out_rows.append(_row(f"sparsevlm_b{b}", _agg(sparsevlm_rows[b], num_keys), extra=sv_note))
        if args.run_fastv:
            out_rows.append(_row(f"fastv_b{b}", _agg(fastv_rows[b], num_keys),
                                  extra=f"agg_layer={args.fastv_agg_layer}"))
        if args.run_ifprune:
            out_rows.append(_row(f"ifprune_b{b}", _agg(ifprune_rows[b], num_keys),
                                  extra=f"L={args.ifprune_prune_layer},head={args.ifprune_head_dtype}"))
        if args.run_visionzip:
            out_rows.append(_row(f"visionzip_b{b}", _agg(vz_rows[b], num_keys),
                                  extra=f"ctx_ratio={args.visionzip_contextual_ratio:.4g}"))

    fields = ["model", "dataset", "method", "n_images", "model_load_peak_gb", "note", "ttft_s_median"]
    fields += [k + "_mean" for k in num_keys] + [k + "_std" for k in num_keys]
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(out_rows)

    if args.save:
        if ds and "answer" in ds[0]:
            output_results["meta"] = [
                {"answer": d.get("answer", ""), "options": d.get("options", []),
                 "question_type": d.get("question_type", "open"), "id": d.get("id", str(k)),
                 "answer_text": d.get("answer_text", ""), "category": d.get("category", ""),
                 "answers": d.get("answers", [])}
                for k, d in enumerate(ds)
            ]
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(output_results, f, indent=2, ensure_ascii=False)
        print(f"[saved] {args.out_json}  (references={len(output_results['references'])}, "
              f"candidates={list(output_results['candidates'].keys())})")
        print(f"  接著跑 accuracy，例如：python evaluate_{args.dataset}.py "
              f"--input_json {args.out_json} ...（看你 dataset 對應哪支 evaluate_*.py）")

    # stdout 表
    print(f"\n=== metrics  {tag}  {ds_tag}  ({len(ds)} imgs, {method_note}) ===")
    print(f"{'method':<20}{'raw_tiles':>10}{'TTFT med s':>12}{'Δmem GB':>10}"
          f"{'KV cache MB':>13}{'ViT TF':>9}{'LLM TF':>9}{'total TF':>10}")
    print("-" * 93)
    for r in out_rows:
        def g(k):
            v = r.get(k + "_mean")
            return "   —" if v is None else f"{v:.3f}"
        med = r.get("ttft_s_median")
        print(f"{r['method']:<20}{g('raw_tiles'):>10}{('   —' if med is None else f'{med:.3f}'):>12}"
              f"{g('mem_delta_gb'):>10}{g('kv_cache_mb'):>13}{g('vit_tflops'):>9}{g('llm_tflops'):>9}"
              f"{g('total_tflops'):>10}")
        if r.get("t_detail_vit_mean") is not None:
            print("      TTFT 分段：" + "  ".join(f"{k}={r.get(f't_{k}_mean') or 0:.3f}" for k in BREAKDOWN_KEYS))
    if unc_rows:
        u = out_rows[0]
        for r in out_rows[1:]:
            du = u.get("peak_alloc_gb_mean") or 0
            dr = r.get("peak_alloc_gb_mean") or 0
            ku = u.get("kv_cache_mb_mean") or 0
            kr = r.get("kv_cache_mb_mean") or 0
            if ku:
                print(f"  {r['method']:<14} Δkv_cache={kr - ku:+.1f} MB  ({(kr - ku) / ku * 100:+.1f}%)")
            tu = u.get("ttft_s_mean")
            tr = r.get("ttft_s_mean")
            if du:
                line = (f"  {r['method']:<14} Δpeak_alloc={dr - du:+.2f} GB  "
                        f"({(dr - du) / du * 100:+.1f}%)")
                if tu and tr is not None:    # 該方法沒量 TTFT（None）就不印，不要當 0 算成 -100%
                    line += f"   ΔTTFT={tr - tu:+.3f} s ({(tr - tu) / tu * 100:+.1f}%)"
                print(line)
    print(f"\n[saved] {args.out_csv}")


if __name__ == "__main__":
    main()
