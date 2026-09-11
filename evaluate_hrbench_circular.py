"""
HR-Bench 官方 CircularEval 計分。

HR-Bench 每題有 4 個「選項輪換」版本（同圖同問句，A/B/C/D 順序輪轉）。官方協定：
一組 4 個全部答對，該題才算對（vanilla per-sample acc 會高估，因為模型只要位置
偏好就能矇到 1~3 個）。這支腳本吃 internvl_stream_v2.py 存的 run JSON
（{"references":[...], "candidates":{tag:[...]}, "meta":[...]}），把 200 筆（= 50 組
× 4 輪換）攤平的結果重新按組計分。

分組方式（--group_by）：
  id      （預設）：以 meta 裡的題目 id 分組（int(id) // 4）。HR-Bench 官方排列每題
                    4 個選項輪換、id 連號（0..3 / 4..7 / ...），所以 id // 4 就是題號。
                    認 id 不認位置 —— 續跑錯位／去重都不影響，且唯一。
  options          ：以「category + 排序後的 4 個選項字串」當 key。看似穩健，但 HR-Bench
                    有多題共用同一個選項集合（顏色／數量題，如 {Red,Green,Blue,Black}），
                    會被併成一大組 → 系統性低估 circ%。除非 meta 沒有乾淨的 id 否則別用。
  index4          ：直接每 4 筆連續一組（i // 4）。跟資料集實際排列一致時最快，
                    但續跑錯位就會壞。
三種都會檢查每組剛好 4 筆，不是就大聲警告。

顯著性：paired bootstrap 的重抽單位是「組」（n = 組數 ≈ 50），不是攤平的 200 筆，
這才是 CircularEval 正確的統計單位（攤平會讓 CI 窄約 2 倍）。

用法：
  python evaluate_hrbench_circular.py --input_json out_block_a_hrbench_b1024.json \
      --baseline_tag budget=1024_info_density_delay0 \
      --dump_csv hrbench_circ_b1024.csv --summary_json summary_circ_b1024.json
"""
import csv
import json
import argparse
from collections import OrderedDict, defaultdict

from evaluate_hrbench import (
    canon_pred, gold_letter, pick_baseline_tag, paired_bootstrap_delta, _tidy,
)


def _group_key(m, mode):
    if mode == "index4":
        return None  # handled by caller via running index
    opts = m.get("options", []) or []
    return (str(m.get("category") or ""), tuple(sorted(str(o) for o in opts)))


def build_groups(meta, mode):
    """回傳 [[idx, idx, ...], ...]，每個內層 list 是一組輪換的 sample index。"""
    if mode == "index4":
        return [list(range(i, min(i + 4, len(meta)))) for i in range(0, len(meta), 4)]

    if mode == "id":
        ids = []
        for m in meta:
            raw = str(m.get("id", "")).strip()
            if not raw.isdigit():
                print(f"[circular] ⚠ meta.id 不是乾淨的整數（見 {raw!r}），"
                      f"group_by=id 退回 index4")
                return build_groups(meta, "index4")
            ids.append(int(raw))
        buckets = OrderedDict()
        for i, gid in enumerate(ids):
            buckets.setdefault(gid // 4, []).append(i)
        return list(buckets.values())

    buckets = OrderedDict()
    for i, m in enumerate(meta):
        buckets.setdefault(_group_key(m, mode), []).append(i)
    return list(buckets.values())


def evaluate(json_path, prompt_mode, group_by, dump_csv, baseline_tag=None,
             n_boot=2000, seed=0, summary_json=None):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "candidates" not in data or not data.get("meta"):
        raise SystemExit('JSON 需要 {"references":[...], "candidates":{tag:[...]}, "meta":[...]}')

    references = [str(x) for x in data.get("references", [])]
    candidates = {k: [str(x) for x in v] for k, v in data["candidates"].items()}
    meta = data["meta"]
    for m in meta:
        m.setdefault("options", [])
        m.setdefault("answer", "")
        m.setdefault("answer_text", "")
        m.setdefault("category", "")
    gt_len = len(meta)
    tags = sorted(candidates)

    groups = build_groups(meta, group_by)
    bad = [g for g in groups if len(g) != 4]
    print(f"[circular] group_by={group_by}  {len(groups)} 組 "
          f"({sum(len(g) for g in groups)} 筆)"
          + (f"  ⚠ {len(bad)} 組不是 4 筆" if bad else ""))
    if bad:
        for g in bad[:5]:
            print(f"    e.g. group idx={g}  members={len(g)}")

    golds = [gold_letter(m) for m in meta]
    categories = sorted({(m.get("category") or "uncat") for m in meta})

    # ---- 每欄：先算 per-sample 對錯，再按組聚合 ----
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
        per_sample = [0] * n
        for i in range(n):
            pc = canon_pred(preds[i], meta[i], prompt_mode)
            per_sample[i] = 1 if (pc is not None and pc == golds[i]) else 0

        grp_correct, cat_c, cat_n = [], defaultdict(int), defaultdict(int)
        for g in groups:
            g = [i for i in g if i < n]
            if not g:
                continue
            allc = 1.0 if all(per_sample[i] for i in g) else 0.0
            grp_correct.append(allc)
            c = meta[g[0]].get("category") or "uncat"
            cat_n[c] += 1
            cat_c[c] += allc
            if dump_csv:
                rows_csv.append({
                    "tag": tag, "group_size": len(g),
                    "category": c, "all_correct": int(allc),
                    "per_member": "".join(str(per_sample[i]) for i in g),
                    "gold": "".join(golds[i] or "?" for i in g),
                    "sample_idx": ",".join(str(i) for i in g),
                })
        ng = len(grp_correct)
        per_tag[tag] = dict(
            n_groups=ng,
            circ_acc=sum(grp_correct) / ng * 100 if ng else 0.0,
            vanilla_acc=sum(per_sample) / n * 100 if n else 0.0,
            by_category={c: (cat_c[c] / cat_n[c] * 100 if cat_n[c] else float("nan"))
                         for c in categories},
            grp_scores=grp_correct,
        )

    base_tag = pick_baseline_tag(tags, baseline_tag)
    base_scores = per_tag[base_tag]["grp_scores"]

    print(f"\n=== HR-Bench CircularEval ===   prompt_mode={prompt_mode}   "
          f"groups={len(base_scores)}   baseline = {base_tag}")
    print(f"paired bootstrap over GROUPS  n_boot={n_boot}  (* = 95% CI 不含 0)\n")
    cat_hdr = "".join(f"{c[:10]+'%':>12}" for c in categories)
    hdr = (f"{'tag':<46} {'grp':>4} {'circ%':>7} {'vanilla%':>9}{cat_hdr}"
           f"   {'Δcirc% vs base (95% CI)':>26}")
    print(hdr)
    print("-" * len(hdr))
    summary = []
    for tag in ordered:
        r = per_tag[tag]
        if tag == base_tag:
            dstr, d, lo, hi = "—  (baseline)", 0.0, 0.0, 0.0
        else:
            d, lo, hi = paired_bootstrap_delta(r["grp_scores"], base_scores, n_boot, seed)
            d, lo, hi = d * 100, lo * 100, hi * 100
            dstr = f"{d:+.2f}  [{lo:+.2f}, {hi:+.2f}]{'*' if (lo > 0 or hi < 0) else ' '}"
        cat_cells = "".join(f"{r['by_category'][c]:>12.2f}" for c in categories)
        print(f"{tag:<46} {r['n_groups']:>4} {r['circ_acc']:>7.2f} "
              f"{r['vanilla_acc']:>9.2f}{cat_cells}   {dstr:>26}")
        summary.append(dict(
            tag=tag, n_groups=r["n_groups"], circ_acc=r["circ_acc"],
            vanilla_acc=r["vanilla_acc"], by_category=r["by_category"],
            delta=d, ci_lo=lo, ci_hi=hi,
            significant=bool(tag not in (base_tag,) and (lo > 0 or hi < 0)),
            is_baseline=(tag == base_tag), is_reference=(tag == REF_KEY),
        ))

    if dump_csv:
        with open(dump_csv, "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["tag", "group_size", "category",
                                              "all_correct", "per_member", "gold", "sample_idx"])
            w.writeheader()
            w.writerows(rows_csv)
        print(f"\n[saved] {dump_csv}  ({len(rows_csv)} rows)")

    if summary_json:
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump({"source_json": json_path, "metric": "hrbench_circular_accuracy",
                       "prompt_mode": prompt_mode, "group_by": group_by,
                       "categories": categories, "baseline_tag": base_tag,
                       "n_boot": n_boot, "seed": seed, "rows": summary}, f,
                      indent=2, ensure_ascii=False)
        print(f"[saved] {summary_json}")

    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HR-Bench 官方 CircularEval 計分")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--input_json", type=str)
    src.add_argument("--output_csv", type=str, help="同 --input_json（相容舊呼叫）")
    ap.add_argument("--prompt_mode", type=str, default="letter", choices=["letter", "open"])
    ap.add_argument("--group_by", type=str, default="id", choices=["id", "options", "index4"])
    ap.add_argument("--baseline_tag", type=str, default=None)
    ap.add_argument("--dump_csv", type=str, default=None)
    ap.add_argument("--summary_json", type=str, default=None)
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    json_path = args.input_json or args.output_csv
    if not json_path:
        ap.error("需要 --input_json 指定結果 JSON")
    evaluate(json_path, args.prompt_mode, args.group_by, args.dump_csv,
             args.baseline_tag, args.n_boot, args.seed, args.summary_json)
