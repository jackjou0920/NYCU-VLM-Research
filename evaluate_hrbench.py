"""
HR-Bench 評測（吃 internvl_stream_v2.py 存的 JSON：
    {"references": [...], "candidates": {tag: [...]}, "meta": [...]}）

meta[i] = {"answer": "A", "answer_text": "27B", "options": ["27B","37B","27D","27E"],
           "question_type": "multiple-choice", "category": "single"|"cross", "id": ...}
順序對齊 references[i] / candidates[tag][i]，所以這支腳本不用重載 dataset、沒有對齊風險。

HR-Bench 是純選擇題 benchmark，metric 只有 accuracy（答對率）。兩種跑法都用這支評：

  --prompt_mode letter （load_hrbench 的預設）
        prompt 已經帶四個選項並要求「只輸出選項字母」，模型輸出通常是 "A" / "(B)" /
        "The answer is C" 之類。解析成字母後跟 meta 的 gold 字母 exact-match。

  --prompt_mode open
        prompt 只有題目，模型自由生成一句話。除了照樣試著抽字母，還會用「選項全文
        是否出現在輸出裡」反推選中的選項（parse_multi_choice 第 2 段），最後再退一步
        用正規化後的包含關係去比對 answer_text。

逐 tag 印：
    acc%            整體答對率
    per-category    每個 category（single / cross ...）各自的答對率
    matchRef%       跟未壓縮輸出（references）答案一致的比例 = 壓縮有沒有改到答案
    unpars%         完全解析不出任何選項的比例
    Δacc% vs base   相對 --baseline_tag 的差，附 paired bootstrap 95% CI（* = 不含 0）
references（未壓縮輸出）也當成一欄一起評 = accuracy 上界。

用法：
    python evaluate_hrbench.py --input_json out_block_a_hrbench_hrbench_4k_letter_b1024.json \
        --prompt_mode letter --baseline_tag budget=1024_info_density_delay0 \
        --dump_csv hrbench_per_sample.csv --summary_json summary_hrbench_b1024.json

summary_json 的 rows 欄位跟 evaluate_mmmu.py 一致（tag/n/acc/delta/ci_lo/ci_hi/...），
plot_block_a.py --metric_field acc 直接吃。
"""
import re
import csv
import json
import string
import argparse
from collections import OrderedDict


# ── 答案抽取（同 evaluate_mmmu.py）：candidate 可能是 CoT，先抽最後的短答案 ──
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


def _norm_str(s):
    s = str(s).strip().lower()
    s = re.sub(r"[%s]" % re.escape(string.punctuation), " ", s)
    return " ".join(s.split())


# ── 選項字母 <-> 選項文字 ────────────────────────────────────────────────────
def _index2ans(options):
    L = string.ascii_uppercase
    return OrderedDict((L[i], str(o)) for i, o in enumerate(options))


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
    for c in choices:                                    # 1) 明確字母標記： (A) / A) / A. / " A "
        if f"({c})" in resp or f"{c})" in resp or f"{c}." in resp or f" {c} " in padded:
            cands.append(c)
    if not cands:                                        # 2) 選項全文出現在輸出裡 -> 反推
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


def loose_option_match(text, index2ans):
    """open 模式最後一道退路：正規化後看哪個選項文字被輸出包含（或反過來）。
    只在 parse_multi_choice 完全失敗時用，回傳字母或 None。"""
    t = _norm_str(text)
    if not t:
        return None
    hits = []
    for c, ans in index2ans.items():
        a = _norm_str(ans)
        if a and (a in t or (len(a) >= 3 and t in a)):
            hits.append((c, len(a)))
    if not hits:
        return None
    hits.sort(key=lambda x: x[1], reverse=True)          # 命中最長的選項最可信
    return hits[0][0]


def gold_letter(m):
    """meta -> gold 選項字母。HR-Bench 的 answer 本來就是字母，這裡只是多幾層防呆。"""
    opts = m.get("options", []) or []
    letters = string.ascii_uppercase[:len(opts)] or "ABCD"
    a = str(m.get("answer", "") or "").strip()
    if len(a) == 1 and a.upper() in letters:
        return a.upper()
    idx2 = _index2ans(opts)
    for src in (a, m.get("answer_text", "")):
        an = _norm_str(src)
        if not an:
            continue
        for c, o in idx2.items():
            if _norm_str(o) == an:
                return c
    return a.upper()[:1] if a else None


def canon_pred(text, m, prompt_mode):
    """輸出 -> 選項字母（或 None）。"""
    idx2 = _index2ans(m.get("options", []) or [])
    short = extract_final_answer(text)
    letter = parse_multi_choice(short, idx2) or parse_multi_choice(str(text), idx2)
    if letter is None and prompt_mode == "open":
        letter = loose_option_match(short, idx2) or loose_option_match(str(text), idx2)
    return letter


# ── baseline 選擇 + paired bootstrap（copy 自 evaluate_mmmu.py）───────────────
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


# ── main ────────────────────────────────────────────────────────────────────
def evaluate(json_path, prompt_mode, dump_csv, baseline_tag=None,
             n_boot=2000, seed=0, summary_json=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "candidates" not in data:
        raise SystemExit('JSON 需要 {"references":[...], "candidates":{tag:[...]}, "meta":[...]}')
    if not data.get("meta"):
        raise SystemExit("JSON 沒有 meta（answer/options/category）。"
                         "請用更新後的 internvl_stream_v2.py 重跑（--dataset hrbench 會存 meta）。")

    references = [str(x) for x in data.get("references", [])]
    candidates = {k: [str(x) for x in v] for k, v in data["candidates"].items()}
    meta = data["meta"]
    for m in meta:                                            # 補防呆欄位
        m.setdefault("options", [])
        m.setdefault("answer", "")
        m.setdefault("answer_text", "")
        m.setdefault("category", "")
    tags = sorted(candidates)
    gt_len = len(meta)
    categories = sorted({(m.get("category") or "uncat") for m in meta})

    # ---- 長度 / 對齊檢查 ----
    print(f"\n{'column':<52} {'length':>8}")
    print("-" * 62)
    print(f"{'meta (gold answers)':<52} {gt_len:>8}   categories={categories}")
    if references:
        print(f"{'references (uncompressed output)':<52} {len(references):>8}"
              + ("   <-- != meta" if len(references) != gt_len else ""))
    for t in tags:
        L = len(candidates[t])
        print(f"{t:<52} {L:>8}" + ("" if L == gt_len else "   <-- != meta, 疑似對齊錯位"))

    golds = [gold_letter(m) for m in meta]
    ref_canon = ([canon_pred(references[i], meta[i], prompt_mode)
                  for i in range(min(gt_len, len(references)))] if references else [])

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
        correct, mref, unpar = [], [], 0
        cat_c = {c: 0 for c in categories}
        cat_n = {c: 0 for c in categories}
        for i in range(n):
            pc = canon_pred(preds[i], meta[i], prompt_mode)
            s = 1.0 if (pc is not None and pc == golds[i]) else 0.0
            correct.append(s)
            unpar += int(pc is None)
            c = meta[i].get("category") or "uncat"
            cat_n[c] += 1
            cat_c[c] += s
            if tag != REF_KEY and ref_canon:
                mref.append(1.0 if (pc is not None and pc == ref_canon[i]) else 0.0)
            if dump_csv:
                rows_csv.append({
                    "tag": tag, "idx": i, "id": meta[i].get("id", i), "category": c,
                    "correct": int(s >= 0.999), "gold": golds[i],
                    "pred_parsed": pc if pc is not None else "",
                    "match_ref": "" if tag == REF_KEY else int((mref[-1] if mref else 0) >= 0.999),
                    "pred_raw": _tidy(preds[i])[:200],
                })
        per_tag[tag] = dict(
            n=n,
            acc=sum(correct) / n * 100 if n else 0.0,
            by_category={c: (cat_c[c] / cat_n[c] * 100 if cat_n[c] else float("nan"))
                         for c in categories},
            mref=sum(mref) / len(mref) * 100 if mref else float("nan"),
            unpar=unpar / n * 100 if n else 0.0,
            scores=correct,
        )

    base_tag = pick_baseline_tag(tags, baseline_tag)
    base_scores = per_tag[base_tag]["scores"]

    print(f"\n=== HR-Bench accuracy ===   prompt_mode={prompt_mode}   n={gt_len}   baseline = {base_tag}")
    print(f"paired bootstrap n_boot={n_boot}  (* = 95% CI 不含 0，相對 baseline 顯著)\n")
    cat_hdr = "".join(f"{c[:10]+'%':>12}" for c in categories)
    hdr = (f"{'tag':<46} {'n':>5} {'acc%':>7}{cat_hdr} {'matchRef%':>10} {'unpars%':>8}"
           f"   {'Δacc% vs base (95% CI)':>26}")
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for tag in ordered:
        r = per_tag[tag]
        if tag == base_tag:
            dstr, d, lo, hi = "—  (baseline)", 0.0, 0.0, 0.0
        else:
            d, lo, hi = paired_bootstrap_delta(r["scores"], base_scores, n_boot, seed)
            d, lo, hi = d * 100, lo * 100, hi * 100
            dstr = f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"
        mref_s = "   —" if tag == REF_KEY else f"{r['mref']:.1f}"
        cat_cells = "".join(f"{r['by_category'][c]:>12.2f}" for c in categories)
        print(f"{tag:<46} {r['n']:>5} {r['acc']:>7.2f}{cat_cells} "
              f"{mref_s:>10} {r['unpar']:>8.1f}   {dstr:>26}")
        summary.append(dict(
            tag=tag, n=r["n"], acc=r["acc"], by_category=r["by_category"],
            match_ref=r["mref"], unparsed=r["unpar"],
            delta=d, ci_lo=lo, ci_hi=hi,
            significant=bool(tag not in (base_tag,) and (lo > 0 or hi < 0)),
            is_baseline=(tag == base_tag), is_reference=(tag == REF_KEY),
            len_ok=(len(cols[tag]) == gt_len),
        ))

    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "idx", "id", "category", "correct",
                                              "gold", "pred_parsed", "match_ref", "pred_raw"])
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\n[saved] {dump_csv}  ({len(rows_csv)} rows)")

    if summary_json:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({"source_json": json_path, "metric": "hrbench_accuracy",
                       "prompt_mode": prompt_mode, "categories": categories,
                       "baseline_tag": base_tag, "n_boot": n_boot, "seed": seed,
                       "rows": summary}, f, indent=2, ensure_ascii=False)
        print(f"[saved] {summary_json}")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HR-Bench 評測（accuracy + per-category）")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input_json", type=str)
    src.add_argument("--output_csv", type=str, help="同 --input_json（相容舊呼叫）")
    ap.add_argument("--prompt_mode", type=str, default="letter", choices=["letter", "open"],
                    help="跑這份結果時 load_hrbench 用的 prompt 模式（影響解析退路）")
    ap.add_argument("--baseline_tag", type=str, default=None,
                    help="Δacc 的比較基準；預設自動挑沒有聚合後綴的第一個 tag")
    ap.add_argument("--dump_csv", type=str, default=None)
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    json_path = args.input_json or args.output_csv
    if not json_path:
        ap.error("需要 --input_json 指定結果 JSON")
    evaluate(json_path, args.prompt_mode, args.dump_csv, args.baseline_tag,
             args.n_boot, args.seed, args.summary_json)
