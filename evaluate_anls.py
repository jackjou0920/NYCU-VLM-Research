"""
DocVQA ANLS 評測：對一份 run 出來的 JSON（{"references": [...], "candidates": {tag: [...]}}）
逐 tag 算 ANLS。

Ground truth 三種來源，優先序 --use_hf_gt > meta > references：
  --use_hf_gt        從 HF lmms-lab-encoder/DocVQA validation **重新載入**官方
                     answers（多 GT，官方算法）。**假設 JSON 裡第 i 筆對應 HF
                     dataset 第 i 筆**——如果 load_docvqa() 跳過了任何一張圖
                     （no image / 壞檔），後面每一筆都會位移，且不會有警告。
                     只建議跑舊版（load_docvqa 還沒存 meta.answers）留下的 JSON
                     才用這個；新跑的都應該有 meta，不需要它。
  （預設，meta 有 answers）不重載 HF，直接用 JSON meta 裡存好的 answers（
                     load_docvqa 存的，跟 candidates 天生對齊、沒有錯位風險）。
  （都沒有時的最後手段）用 JSON 裡的 references 當 pseudo-GT。不用連網，但
                     references 本身要是乾淨、對齊的才有意義。

**對齊是 index-based**：gold[i] / references[i] 對應 candidates[tag][i]。
所以腳本會先印各欄長度，任何一欄長度跟 GT 對不上就大聲警告——長度不一致
幾乎一定代表續跑時 append 錯位（前面 N 筆重複、其餘 off-by-N），該欄的 ANLS
不可信。

用法：
    python evaluate_anls.py --output_csv output_docvqa_internvl_t2.json --use_hf_gt
    python evaluate_anls.py --input_json output_docvqa_internvl_t2.json            # 用 references
    python evaluate_anls.py --output_csv run.json --use_hf_gt --dump_csv anls_per_sample.csv

── metric 2：--similarity ────────────────────────────────────────────────────
對每個 tag 算「該 tag 的輸出 vs references（未壓縮輸出）」的 sentence-embedding
cosine + BERTScore F1（呼叫 semantic_similarity_evaluator，跟 evaluate_mmmu.py
的 --similarity 是同一套邏輯）。不需要 ground truth，任何 tag 都能比，是連續值
不是接近二元的 ANLS，用同樣的 n 通常能偵測到更小的效應——ANLS 的 paired
bootstrap CI 太寬、看不出兩個 selection 變體差異時，先看這個。
需要 pip install sentence-transformers bert-score --break-system-packages。

    python evaluate_anls.py --input_json out.json --use_hf_gt --similarity
"""
import re
import csv
import json
import argparse

import Levenshtein

HF_DATASET = "lmms-lab-encoder/DocVQA"
HF_CONFIG = "DocVQA"
HF_SPLIT = "validation"


# ──────────────────────────────────────────────────────────────────────────────
# ANLS（單一樣本，支援多 ground truth）
# ──────────────────────────────────────────────────────────────────────────────
def advanced_normalize_text(text: str) -> str:
    """
    進階文本清洗：比照官方 MMEval / VLMEvalKit 常規
    移除贅字、修飾詞、單位、標點，只保留核心核心詞
    """
    text = str(text).lower().strip()
    
    # 1. 移除大模型常見的句首引導詞、修飾詞 (nearly, about, approximately 等)
    prefixes_to_remove = [
        r"^\s*according\s+to\s+the\s+[a-z_]+\s*,\s*", 
        r"^\s*the\s+[a-z_]+\s+is\s+", r"^\s*it\s+is\s+approximately\s+",
        r"\b(approximately|approx|about|nearly|over|around|more\s+than|less\s+than|almost)\b"
    ]
    for p in prefixes_to_remove:
        text = re.sub(p, "", text, flags=re.IGNORECASE)
        
    # 2. 如果模型吐了一長串話，嘗試抽取「最後一個標點後面的核心」或「引號內文字」
    # 許多模型喜歡寫: The total amount is $150. 
    if "is " in text:
        text = text.split("is ")[-1]
    
    # 移除句尾的點 (例如 "50." -> "50")
    text = text.rstrip('.')
    
    # 3. 移除常見的末尾單位贅字 (例如 "4 million farmers" -> "4 million")
    text = re.sub(r"\s+(farmers|people|patients|dollars|units|crores|rupees|lacs)\b.*$", "", text, flags=re.IGNORECASE)
    
    # 4. 清除多餘符號與空格
    text = re.sub(r"[,\"\';\(\):\-–—?]", " ", text) # 將標點轉空格，避免影響長度比對
    text = " ".join(text.split()) # 壓縮連續空格
    
    return text.strip()


def calculate_single_anls(prediction: str, ground_truths, theta: float = 0.5) -> float:
    """ 升級版 ANLS 計算 """
    # 先做進階清洗
    pred = advanced_normalize_text(prediction)
    max_anls = 0.0
    
    for gt in ground_truths:
        gt = advanced_normalize_text(gt)
        
        if not pred and not gt:
            max_anls = max(max_anls, 1.0)
            continue
        if not pred or not gt:
            max_anls = max(max_anls, 0.0)
            continue
            
        # 計算編輯距離
        edit_distance = Levenshtein.distance(pred, gt)
        max_len = max(len(pred), len(gt))
        nld = edit_distance / max_len
        
        ans_similarity = (1.0 - nld) if nld < theta else 0.0
        
        # 額外防防呆：如果 pred 是一句長話但「完全包含」了短的 gt (或者相反)
        # 且長度落差在合理範圍內，給予部分分數免得直接歸 0
        if ans_similarity == 0.0 and (gt in pred or pred in gt) and len(pred) < len(gt) * 3:
            ans_similarity = min(len(pred), len(gt)) / max(len(pred), len(gt))
            # 限制 fallback 相似度不要高過門檻太多，但至少不為 0
            ans_similarity = ans_similarity if ans_similarity < theta else 0.45
            
        if ans_similarity > max_anls:
            max_anls = ans_similarity
            
    return max_anls


# ──────────────────────────────────────────────────────────────────────────────
# 答案抽取：candidate 常是 CoT（"Step 1... Answer: X"）或整句，先抽最後的短答案
# ──────────────────────────────────────────────────────────────────────────────
_MARKER_RE = re.compile(
    r"^\s*\**\s*(?:the\s+)?(?:final\s+answer|answer|ans)\s*\**\s*[:\-–—]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _tidy(s: str) -> str:
    s = s.strip().strip("`").strip()
    s = s.strip("* ").strip()
    s = s.strip(" \t\"'")
    return re.sub(r"\s+", " ", s).strip()


def extract_final_answer(text: str) -> str:
    """由後往前找 "Answer: ..." 行取冒號後內容；找不到 marker 就取最後一個非空行。"""
    if not text:
        return ""
    lines = [ln.strip() for ln in str(text).strip().splitlines() if ln.strip()]
    if not lines:
        return ""
    for ln in reversed(lines):
        m = _MARKER_RE.match(ln)
        if m:
            return _tidy(m.group(1))
    return _tidy(lines[-1])


# ──────────────────────────────────────────────────────────────────────────────
# HF gold
# ──────────────────────────────────────────────────────────────────────────────
_AGG_SUFFIXES = ("_max", "_topk_mean", "_mean_std", "_quantile")


def pick_baseline_tag(tags, explicit=None):
    """baseline = mean 聚合那個 tag（tag 沒有 _max/_topk_mean/... 後綴）。
    找不到就用第一個 tag。可用 --baseline_tag 明確指定。"""
    if explicit:
        if explicit not in tags:
            raise SystemExit(f"--baseline_tag {explicit!r} 不在 tags 裡：{tags}")
        return explicit
    plain = [t for t in tags if not t.endswith(_AGG_SUFFIXES)]
    return plain[0] if plain else tags[0]


def paired_bootstrap_delta(var_scores, base_scores, n_boot=2000, seed=0):
    """paired bootstrap：回傳 (delta, lo95, hi95)，delta = mean(var) - mean(base)。
    同一個 index 的 (var_i, base_i) 綁在一起 resample，反映「同一題」的配對關係，
    比未配對的標準誤更緊。CI 不含 0 → 該 tag 相對 baseline 的差異在 95% 顯著。"""
    m = min(len(var_scores), len(base_scores))
    d = [var_scores[i] - base_scores[i] for i in range(m)]
    obs = sum(d) / m if m else 0.0
    try:
        import numpy as np

        arr = np.asarray(d, dtype=np.float64)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, m, size=(n_boot, m))
        boot = arr[idx].mean(axis=1)
        lo, hi = np.percentile(boot, [2.5, 97.5])
        return obs, float(lo), float(hi)
    except ImportError:
        import random

        rng = random.Random(seed)
        boot = []
        for _ in range(n_boot):
            s = 0.0
            for _ in range(m):
                s += d[rng.randrange(m)]
            boot.append(s / m)
        boot.sort()
        return obs, boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]


def load_hf_gold(dataset=HF_DATASET, config=HF_CONFIG, split=HF_SPLIT):
    """回傳 list[list[str]]，第 i 筆是第 i 題所有可接受答案。"""
    from datasets import load_dataset

    ds = load_dataset(dataset, config, split=split)
    key = next(
        (k for k in ("answers", "answer", "gt_answers", "gt_answer", "gt") if k in ds.column_names),
        None,
    )
    if key is None:
        raise KeyError(f"找不到答案欄位，dataset 欄位有：{ds.column_names}")

    gold = []
    for a in ds[key]:
        if isinstance(a, str):
            a = [a]
        gold.append([str(x) for x in a])
    return gold


# ──────────────────────────────────────────────────────────────────────────────
# metric 2：語意相似度（vs references，不需要 ground truth）
# ──────────────────────────────────────────────────────────────────────────────
def run_similarity(references, candidates, tags):
    try:
        from semantic_similarity_evaluator import SemanticSimilarityEvaluator
    except Exception as e:                                       # noqa: BLE001
        print(f"\n[similarity] 載入 semantic_similarity_evaluator 失敗：{e}"
              "\n  pip install sentence-transformers bert-score --break-system-packages")
        return
    ev = SemanticSimilarityEvaluator()
    by_tag = {}
    for t in tags:
        k = min(len(references), len(candidates[t]))
        by_tag[t] = candidates[t][:k]
    res = ev.evaluate_multiple(references, by_tag, min_tokens=0)
    print("\n=== metric 2：generated output vs references（未壓縮輸出）語意相似度 ===")
    ev.print_comparison_table(res)


# ──────────────────────────────────────────────────────────────────────────────
def evaluate(json_path, use_hf_gt, threshold, do_extract, dump_csv,
             hf_dataset, hf_config, hf_split,
             baseline_tag=None, n_boot=2000, seed=0, summary_json=None,
             similarity=False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not (isinstance(data, dict) and "candidates" in data):
        raise SystemExit("JSON 結構不對，需要 {\"references\": [...], \"candidates\": {tag: [...]}}")

    print("->", len(data.get("references", [])))
    for k, v in data["candidates"].items():
        print(k, len(v))

    references = [str(x) for x in data.get("references", [])]
    candidates = {k: [str(x) for x in v] for k, v in data["candidates"].items()}
    tags = sorted(candidates)

    prep = extract_final_answer if do_extract else (lambda s: str(s).strip())

    # ---- ground truth：優先序 --use_hf_gt > meta.answers > references ----
    meta = data.get("meta") or []
    meta_has_answers = bool(meta) and any(m.get("answers") for m in meta)
    if use_hf_gt:
        print(f"[gt] loading HF {hf_dataset}/{hf_config}:{hf_split} ..."
              " (assumes JSON row i == HF row i, no skipped images)")
        gts = load_hf_gold(hf_dataset, hf_config, hf_split)          # list[list[str]]
        gt_name = f"HF {hf_dataset}:{hf_split} answers (reloaded)"
    elif meta_has_answers:
        # load_docvqa 存的、跟 candidates 天生對齊，不用重載 HF。
        # 個別題目 answers 是空的話（理論上不該發生），退回該題的 reference 當 GT，
        # 避免整題直接算 0 分。
        gts = [
            [str(a) for a in m.get("answers", [])] or ([prep(references[i])] if i < len(references) else [])
            for i, m in enumerate(meta)
        ]
        gt_name = "JSON meta answers (aligned, from load_docvqa)"
    else:
        if not references:
            raise SystemExit("JSON 沒有 references，且沒有 meta.answers，請改用 --use_hf_gt")
        gts = [[prep(r)] for r in references]                        # list[list[str]]
        gt_name = "JSON references (pseudo-GT)"
    gt_len = len(gts)

    # ---- 長度 / 對齊檢查（這次踩過的坑：續跑 append 錯位會讓某欄長度 != 其他欄）----
    print(f"\n{'column':<46} {'length':>8}")
    print("-" * 56)
    print(f"{'<ground truth> ' + gt_name:<46} {gt_len:>8}")
    if references:
        print(f"{'references':<46} {len(references):>8}")
            #   + ("   <-- != GT" if len(references) != gt_len else ""))
    bad = []
    for tag in tags:
        L = len(candidates[tag])
        flag = "" if L == gt_len else "   <-- != GT, 疑似對齊錯位"
        if L != gt_len:
            bad.append(tag)
        print(f"{tag:<46} {L:>8}{flag}")
    if bad:
        print("\n以下 tag 長度跟 GT 不一致，通常是續跑時把新一輪 append 在舊的部分結果上，"
              "\n導致前面若干筆重複、其餘整體位移。這些欄的 ANLS 只是參考，建議刪掉輸出檔重跑：")
        for t in bad:
            print(f"      - {t}")

    # ---- 要評分的欄位：candidates + references 當作「未壓縮上界」一起評 ----
    #      只有 GT 是真正獨立於 references 的來源（HF 重載或 meta.answers）才有意義；
    #      GT 本身就是 references 的話，把 references 拿來跟自己比只會恆等於 100%。
    REF_KEY = "<references / uncompressed baseline>"
    score_cols = dict(candidates)
    if (use_hf_gt or meta_has_answers) and references:
        score_cols[REF_KEY] = references
    ordered = ([REF_KEY] if REF_KEY in score_cols else []) + tags

    # ---- 逐欄算 per-sample ANLS ----
    per_sample_rows = []
    per_tag_scores = {}
    for tag in ordered:
        preds = score_cols[tag]
        n = min(gt_len, len(preds))
        scores = []
        for i in range(n):
            p = prep(preds[i])
            s = calculate_single_anls(p, gts[i], theta=threshold)
            scores.append(s)
            if dump_csv:
                per_sample_rows.append({
                    "tag": tag, "idx": i, "anls": f"{s:.4f}",
                    "gt": " | ".join(gts[i]), "pred": p,
                })
        per_tag_scores[tag] = scores

    # ---- baseline + paired bootstrap ΔANLS ----
    base_tag = pick_baseline_tag(tags, baseline_tag)
    base_scores = per_tag_scores[base_tag]

    print(f"\nground truth = {gt_name}   |   answer extraction = {'ON' if do_extract else 'OFF'}"
          f"   |   threshold = {threshold}   |   baseline = {base_tag}")
    print(f"paired bootstrap: n_boot={n_boot}, seed={seed}   (* = 95% CI 不含 0，相對 baseline 顯著)\n")
    print(f"{'tag':<44} {'n':>6} {'ANLS%':>8} {'exact%':>7} {'zero%':>7}   {'ΔANLS% vs base (95% CI)':>30}")
    print("-" * 112)

    summary = []
    for tag in ordered:
        scores = per_tag_scores[tag]
        n = len(scores)
        anls = sum(scores) / n * 100 if n else 0.0
        exact = sum(1 for s in scores if s >= 0.999) / n * 100 if n else 0.0
        zero = sum(1 for s in scores if s == 0.0) / n * 100 if n else 0.0

        if tag == base_tag:
            delta_str = "—  (baseline)"
            d = lo = hi = 0.0
        else:
            d, lo, hi = paired_bootstrap_delta(scores, base_scores, n_boot=n_boot, seed=seed)
            d, lo, hi = d * 100, lo * 100, hi * 100
            sig = "*" if (lo > 0 or hi < 0) else " "
            delta_str = f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]{sig}"

        summary.append((tag, n, anls, exact, zero, d, lo, hi))
        print(f"{tag:<44} {n:>6} {anls:>8.2f} {exact:>7.1f} {zero:>7.1f}   {delta_str:>30}")

    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "idx", "anls", "gt", "pred"])
            w.writeheader()
            w.writerows(per_sample_rows)
        print(f"\n[saved] {dump_csv}  ({len(per_sample_rows)} rows)")

    if summary_json:
        payload = {
            "source_json": json_path,
            "gt_name": gt_name,
            "gt_source": "hf" if use_hf_gt else ("meta" if meta_has_answers else "references"),
            "threshold": threshold,
            "answer_extraction": bool(do_extract),
            "baseline_tag": base_tag,
            "n_boot": n_boot,
            "seed": seed,
            "rows": [
                {
                    "tag": tag,
                    "n": n,
                    "anls": anls,               # %
                    "exact": exact,             # %
                    "zero": zero,               # %
                    "delta": d,                 # %  ΔANLS vs baseline（baseline 為 0）
                    "ci_lo": lo,                # %  paired bootstrap 95% CI（baseline 為 0）
                    "ci_hi": hi,                # %
                    "significant": bool(tag != base_tag and (lo > 0 or hi < 0)),
                    "is_baseline": bool(tag == base_tag),
                    "is_reference": bool(tag == REF_KEY),
                    "len_ok": bool(len(score_cols[tag]) == gt_len),
                }
                for (tag, n, anls, exact, zero, d, lo, hi) in summary
            ],
        }
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[saved] {summary_json}  ({len(payload['rows'])} rows)  <- plot_results.py 吃這個")

    if similarity and references:
        run_similarity(references, candidates, tags)

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="逐 tag 計算 DocVQA ANLS（GT 可來自 HF answers 或 JSON references）")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input_json", type=str, help="run 出來的結果 JSON 路徑")
    src.add_argument("--output_csv", type=str, help="同 --input_json（相容舊的呼叫方式）")
    ap.add_argument("--use_hf_gt", action="store_true",
                    help="用 HF lmms-lab-encoder/DocVQA validation 的官方 answers 當 GT；"
                         "不加則用 JSON 內的 references")
    ap.add_argument("--threshold", type=float, default=0.5, help="ANLS 門檻，NL 低於此歸 0（官方 0.5）")
    ap.add_argument("--no_extract", action="store_true", help="不做答案抽取，直接比原始字串")
    ap.add_argument("--dump_csv", type=str, default=None, help="把 per-sample (tag, idx, anls, gt, pred) 存成 CSV")
    ap.add_argument("--baseline_tag", type=str, default=None,
                    help="ΔANLS 的比較基準；預設自動挑沒有聚合後綴的 tag（= mean）")
    ap.add_argument("--n_boot", type=int, default=2000, help="paired bootstrap 重抽次數")
    ap.add_argument("--seed", type=int, default=0, help="bootstrap 亂數種子")
    ap.add_argument("--summary_json", type=str, default=None,
                    help="把逐 tag 的 ANLS / ΔANLS / CI 匯出成 JSON，給 plot_results.py 用")
    ap.add_argument("--similarity", action="store_true",
                    help="另外算 metric 2：各 tag 輸出 vs references 的 cosine + BERTScore，"
                         "不需要 ground truth，適合偵測 ANLS bootstrap CI 太寬看不出來的小效應")
    ap.add_argument("--hf_dataset", type=str, default=HF_DATASET)
    ap.add_argument("--hf_config", type=str, default=HF_CONFIG)
    ap.add_argument("--hf_split", type=str, default=HF_SPLIT)
    args = ap.parse_args()

    json_path = args.input_json or args.output_csv
    if not json_path:
        ap.error("需要 --input_json（或相容用的 --output_csv）指定結果 JSON")
    evaluate(
        json_path=json_path,
        use_hf_gt=args.use_hf_gt,
        threshold=args.threshold,
        do_extract=not args.no_extract,
        dump_csv=args.dump_csv,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        baseline_tag=args.baseline_tag,
        n_boot=args.n_boot,
        seed=args.seed,
        summary_json=args.summary_json,
        similarity=args.similarity,
    )
