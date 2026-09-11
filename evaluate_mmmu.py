"""
MMMU 評測（吃 internvl_svm.py 存的 JSON：{"references":[...], "candidates":{tag:[...]}, "meta":[...]}）

meta[i] = {"answer":.., "options":[..], "question_type":"multiple-choice"|"open", "id":..}，
順序對齊 references[i] / candidates[tag][i]，所以這支腳本不用重載 dataset、沒有對齊風險。

── metric 1：accuracy ─────────────────────────────────────────────────────────
  multiple-choice：把輸出解析成選項字母（(A) / A. / " A " / 選項全文反推），跟
                   meta 的 gold 字母 exact-match（移植官方 MMMU parse_multi_choice）。
  open           ：數值題容忍相對誤差；字串題正規化後相等 / 包含。
  references（未壓縮輸出）也當一欄一起評 = accuracy 上界。
  逐 tag 印 acc% / mc% / open% / 對未壓縮答案的一致率 matchRef% / 解析失敗率 unparsed%，
  以及 Δacc vs --baseline_tag + paired bootstrap 95% CI。

── metric 2：--similarity ────────────────────────────────────────────────────
  對每個 tag 算「該 tag 的輸出 vs references（未壓縮輸出）」的
  sentence-embedding cosine + BERTScore F1（呼叫 semantic_similarity_evaluator）。
  需要 pip install sentence-transformers bert-score。

用法：
  python evaluate_mmmu.py --input_json out_mmmu_1024.json \
      --baseline_tag budget=1024_evict_info_density --dump_csv mmmu_per_sample.csv
  python evaluate_mmmu.py --input_json out_mmmu_1024.json --similarity
"""
import re
import csv
import json
import string
import argparse


# ── 答案抽取（同 evaluate_anls.py）：candidate 常是 CoT，先抽最後的短答案 ──
_MARKER_RE = re.compile(
    r"^\s*\**\s*(?:the\s+)?(?:final\s+answer|answer|ans|option|choice)\s*\**\s*[:\-–—]\s*(.+?)\s*$",
    re.IGNORECASE,
)


def _tidy(s):
    s = str(s).strip().strip("`").strip().strip("* ").strip().strip(" \t\"'")
    return re.sub(r"\s+", " ", s).strip()


def extract_final_answer(text):
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


# ── MMMU metric ──────────────────────────────────────────────────────────────
def _norm_str(s):
    s = str(s).strip().lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return " ".join(s.split())


def _numbers(s):
    return re.findall(r"-?\d+\.?\d*", str(s).replace(",", ""))


def _index2ans(options):
    L = string.ascii_uppercase
    return {L[i]: str(o) for i, o in enumerate(options)}


def parse_multi_choice(response, index2ans):
    """自由文字 -> 選項字母，解析不到回傳 None。移植官方 MMMU parse_multi_choice_response 精神。"""
    if response is None:
        return None
    choices = list(index2ans.keys())
    resp = str(response)
    padded = " " + resp + " "
    for ch in [",", ".", "!", "?", ";", ":", "'", '"', ")", "("]:
        padded = padded.replace(ch, " ")
    padded = " " + " ".join(padded.split()) + " "

    cands = []
    for c in choices:                                    # 1) 明確字母標記
        if f"({c})" in resp or f"{c})" in resp or f"{c}." in resp or f" {c} " in padded:
            cands.append(c)
    if not cands:                                        # 2) 選項全文反推
        low = resp.lower()
        for c, ans in index2ans.items():
            a = str(ans).strip().lower()
            if a and a in low:
                cands.append(c)
    if not cands:
        return None
    if len(cands) == 1:
        return cands[0]
    low = resp.lower()                                   # 多個 -> 取最後出現的
    best, best_pos = cands[0], -1
    for c in cands:
        pos = max(low.rfind(f"({c.lower()})"), low.rfind(f"{c.lower()})"),
                  low.rfind(f" {c.lower()} "), low.rfind(str(index2ans[c]).strip().lower()))
        if pos >= best_pos:
            best, best_pos = c, pos
    return best


def eval_open(gold, pred):
    """open：數值題容忍相對誤差；字串題正規化後相等 / gold 被 pred 包含。回傳 0/1。"""
    if pred is None:
        return 0.0
    gn = _numbers(gold)
    if gn:
        try:
            gv = float(gn[0])
        except ValueError:
            gv = None
        if gv is not None:
            for x in _numbers(pred):
                try:
                    if abs(float(x) - gv) <= 1e-4 * max(1.0, abs(gv)):
                        return 1.0
                except ValueError:
                    pass
            return 0.0
    g, p = _norm_str(gold), _norm_str(pred)
    if not g:
        return 0.0
    return 1.0 if (g == p or g in p) else 0.0


def gold_letter(answer, options):
    a = str(answer).strip()
    letters = string.ascii_uppercase[:len(options)]
    if len(a) == 1 and a.upper() in letters:
        return a.upper()
    an = _norm_str(a)
    for i, o in enumerate(options):
        if _norm_str(o) == an:
            return string.ascii_uppercase[i]
    return a.upper()[:1] if a else None


def canon_pred(text, m):
    """輸出 -> 可比對標準形：MC -> 字母 or None；open -> 抽出的短答案字串。"""
    short = extract_final_answer(text)
    if m["question_type"] == "multiple-choice":
        idx2 = _index2ans(m["options"])
        return parse_multi_choice(short, idx2) or parse_multi_choice(str(text), idx2)
    return short or str(text)


def score_pred(pred_canon, m):
    """回傳 (correct 0/1, unparsed_bool)。"""
    if m["question_type"] == "multiple-choice":
        gl = gold_letter(m["answer"], m["options"])
        if pred_canon is None:
            return 0.0, True
        return (1.0 if pred_canon == gl else 0.0), False
    return eval_open(str(m["answer"]), str(pred_canon) if pred_canon is not None else ""), False


def match_ref(pred_canon, ref_canon, m):
    """該 tag 的答案跟未壓縮輸出的答案是否一致（= 壓縮有沒有改到答案）。"""
    if m["question_type"] == "multiple-choice":
        return 1.0 if (pred_canon is not None and pred_canon == ref_canon) else 0.0
    a, b = str(pred_canon or ""), str(ref_canon or "")
    if not _norm_str(a) and not _norm_str(b):
        return 1.0
    return 1.0 if (eval_open(b, a) or eval_open(a, b)) else 0.0


# ── 共用小工具（copy 自 evaluate_anls.py）────────────────────────────────────
_AGG_SUFFIXES = ("_max", "_topk_mean", "_mean_std", "_quantile")


def pick_baseline_tag(tags, explicit=None):
    if explicit:
        if explicit not in tags:
            raise SystemExit(f"--baseline_tag {explicit!r} 不在 tags 裡：{tags}")
        return explicit
    plain = [t for t in tags if not t.endswith(_AGG_SUFFIXES)]
    return plain[0] if plain else tags[0]


def paired_bootstrap_delta(var_scores, base_scores, n_boot=2000, seed=0):
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
        boot = sorted(sum(d[rng.randrange(m)] for _ in range(m)) / m for _ in range(n_boot))
        return obs, boot[int(0.025 * n_boot)], boot[int(0.975 * n_boot)]


# ── metric 2：語意相似度（呼叫既有的 semantic_similarity_evaluator）──────────
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


# ── main ────────────────────────────────────────────────────────────────────
def evaluate(json_path, dump_csv, baseline_tag=None, n_boot=2000, seed=0,
             summary_json=None, similarity=False):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "candidates" not in data:
        raise SystemExit('JSON 需要 {"references":[...], "candidates":{tag:[...]}, "meta":[...]}')
    if "meta" not in data or not data["meta"]:
        raise SystemExit("JSON 沒有 meta（answer/options/question_type）。"
                         "請用更新後的 internvl_svm.py 重跑（它現在會存 meta）。")

    references = [str(x) for x in data.get("references", [])]
    candidates = {k: [str(x) for x in v] for k, v in data["candidates"].items()}
    meta = data["meta"]
    for m in meta:                                            # 補防呆欄位
        m.setdefault("question_type", "multiple-choice" if m.get("options") else "open")
        m.setdefault("options", [])
        m.setdefault("answer", "")
    tags = sorted(candidates)
    gt_len = len(meta)

    # ---- 長度 / 對齊檢查 ----
    print(f"\n{'column':<52} {'length':>8}")
    print("-" * 62)
    print(f"{'meta (gold answers)':<52} {gt_len:>8}")
    if references:
        print(f"{'references (uncompressed output)':<52} {len(references):>8}"
              + ("   <-- != meta" if len(references) != gt_len else ""))
    for t in tags:
        L = len(candidates[t])
        print(f"{t:<52} {L:>8}" + ("" if L == gt_len else "   <-- != meta, 疑似對齊錯位"))

    ref_canon = [canon_pred(references[i], meta[i]) for i in range(min(gt_len, len(references)))] if references else []

    REF_KEY = "<references / uncompressed>"
    cols = dict(candidates)
    if references:
        cols[REF_KEY] = references
    ordered = ([REF_KEY] if REF_KEY in cols else []) + tags

    rows_csv = []
    per_tag = {}
    for tag in ordered:
        preds = cols[tag]
        n = min(gt_len, len(preds))
        correct, mc_c, mc_n, op_c, op_n, mref, unpar = [], 0, 0, 0, 0, [], 0
        for i in range(n):
            pc = canon_pred(preds[i], meta[i])
            s, up = score_pred(pc, meta[i])
            correct.append(s)
            unpar += int(up)
            is_mc = meta[i]["question_type"] == "multiple-choice"
            if is_mc:
                mc_n += 1
                mc_c += s
            else:
                op_n += 1
                op_c += s
            if tag != REF_KEY and ref_canon:
                mref.append(match_ref(pc, ref_canon[i], meta[i]))
            if dump_csv:
                rows_csv.append({
                    "tag": tag, "idx": i, "correct": int(s >= 0.999),
                    "qtype": meta[i]["question_type"],
                    "gold": gold_letter(meta[i]["answer"], meta[i]["options"])
                            if is_mc else meta[i]["answer"],
                    "pred_parsed": pc,
                    "match_ref": "" if tag == REF_KEY else int((mref[-1] if mref else 0) >= 0.999),
                    "pred_raw": _tidy(preds[i])[:200],
                })
        per_tag[tag] = dict(
            n=n, acc=sum(correct) / n * 100 if n else 0.0,
            mc=mc_c / mc_n * 100 if mc_n else float("nan"),
            op=op_c / op_n * 100 if op_n else float("nan"),
            mref=sum(mref) / len(mref) * 100 if mref else float("nan"),
            unpar=unpar / n * 100 if n else 0.0,
            scores=correct,
        )

    base_tag = pick_baseline_tag(tags, baseline_tag)
    base_scores = per_tag[base_tag]["scores"]

    n_mc = sum(1 for m in meta[:gt_len] if m["question_type"] == "multiple-choice")
    print(f"\n=== metric 1：accuracy ===   #MC={n_mc}  #open={gt_len - n_mc}   baseline = {base_tag}")
    print(f"paired bootstrap n_boot={n_boot}  (* = 95% CI 不含 0，相對 baseline 顯著)\n")
    hdr = f"{'tag':<46} {'n':>5} {'acc%':>7} {'mc%':>7} {'open%':>7} {'matchRef%':>10} {'unpars%':>8}   {'Δacc% vs base (95% CI)':>28}"
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for tag in ordered:
        r = per_tag[tag]
        if tag == base_tag:
            dstr, d, lo, hi = "—  (baseline)", 0.0, 0.0, 0.0
        elif tag == REF_KEY:
            d, lo, hi = paired_bootstrap_delta(r["scores"], base_scores, n_boot, seed)
            d, lo, hi = d * 100, lo * 100, hi * 100
            dstr = f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"
        else:
            d, lo, hi = paired_bootstrap_delta(r["scores"], base_scores, n_boot, seed)
            d, lo, hi = d * 100, lo * 100, hi * 100
            dstr = f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"
        mref_s = "   —" if tag == REF_KEY else f"{r['mref']:.1f}"
        print(f"{tag:<46} {r['n']:>5} {r['acc']:>7.2f} {r['mc']:>7.2f} {r['op']:>7.2f} "
              f"{mref_s:>10} {r['unpar']:>8.1f}   {dstr:>28}")
        summary.append(dict(tag=tag, n=r["n"], acc=r["acc"], mc=r["mc"], open=r["op"],
                            match_ref=r["mref"], unparsed=r["unpar"],
                            delta=d, ci_lo=lo, ci_hi=hi,
                            significant=bool(tag not in (base_tag,) and (lo > 0 or hi < 0)),
                            is_baseline=(tag == base_tag), is_reference=(tag == REF_KEY),
                            len_ok=(len(cols[tag]) == gt_len)))

    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "idx", "correct", "qtype", "gold",
                                              "pred_parsed", "match_ref", "pred_raw"])
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\n[saved] {dump_csv}  ({len(rows_csv)} rows)")

    if summary_json:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({"source_json": json_path, "metric": "mmmu_accuracy",
                       "baseline_tag": base_tag, "n_boot": n_boot, "seed": seed,
                       "rows": summary}, f, indent=2, ensure_ascii=False)
        print(f"[saved] {summary_json}")

    if similarity and references:
        run_similarity(references, candidates, tags)

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="MMMU validation 評測（accuracy + 可選語意相似度）")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input_json", type=str)
    src.add_argument("--output_csv", type=str, help="同 --input_json（相容舊呼叫）")
    ap.add_argument("--baseline_tag", type=str, default=None,
                    help="Δacc 的比較基準；預設自動挑沒有聚合後綴的第一個 tag")
    ap.add_argument("--dump_csv", type=str, default=None)
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--similarity", action="store_true",
                    help="另外算 metric 2：各 tag 輸出 vs references 的 cosine + BERTScore")
    args = ap.parse_args()

    json_path = args.input_json or args.output_csv
    if not json_path:
        ap.error("需要 --input_json 指定結果 JSON")
    evaluate(json_path, args.dump_csv, args.baseline_tag, args.n_boot, args.seed,
             args.summary_json, args.similarity)
