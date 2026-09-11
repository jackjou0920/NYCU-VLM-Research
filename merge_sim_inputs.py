"""把多個 budget 的 stream-output JSON 併成一份 semantic_similarity_evaluator.py 的輸入。

run_block.sh 每個 budget 各寫一份 out_block_*_b<budget>.json，內含：
    references        : uncompressed（--run_standard）的自由生成輸出
    candidates[tag]   : 該 budget 壓縮後（--run_stream）的輸出，tag 帶 budget

open 模式下 CircularEval（四選項輪換）不適用，改比「每個 budget 的輸出 vs
uncompressed reference」的語意相似度。這支腳本把 N 份 per-budget JSON 攤平成一份
    {"references": [...], "candidates": {budget1_tag: [...], budget2_tag: [...], ...}}
讓 semantic_similarity_evaluator.py 一次跑完、輸出跨 budget 的比較表（省下每個
budget 重載 sentence model + BERTScorer 的成本）。

references 取第一份有 references 的檔；其餘檔的 references 只用來檢查長度是否對齊
（num_images / dataset / dedup 不一致會錯位）。

用法：
    python merge_sim_inputs.py --out merged.json a_b1024.json a_b2048.json a_b4096.json
"""
import json
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="輸出的合併 JSON 路徑")
    ap.add_argument("inputs", nargs="+", help="各 budget 的 stream-output JSON")
    args = ap.parse_args()

    references = None
    ref_src = None
    candidates = {}

    for path in args.inputs:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            print(f"[skip] 找不到 {path}")
            continue

        refs = data.get("references") or []
        if refs:
            if references is None:
                references, ref_src = refs, path
            elif len(refs) != len(references):
                print(f"[warn] {path} references 長度 {len(refs)} != {ref_src} 的 "
                      f"{len(references)}，兩邊 num_images / dataset / dedup 可能不一致，"
                      f"相似度會錯位")

        for tag, outs in (data.get("candidates") or {}).items():
            if tag in candidates:
                print(f"[warn] tag {tag!r} 在多份檔案出現，用 {path} 的覆蓋前一份")
            candidates[tag] = outs

    if references is None:
        raise SystemExit("沒有任何輸入 JSON 帶 references，請先對它們跑 --run_standard")
    if not candidates:
        raise SystemExit("沒有任何 candidates，無法比對")

    for tag, outs in candidates.items():
        if len(outs) != len(references):
            print(f"[warn] candidate {tag!r} 長度 {len(outs)} != references {len(references)}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"references": references, "candidates": candidates}, f,
                  indent=2, ensure_ascii=False)
    print(f"[merged] {len(candidates)} 個 tag、references={len(references)} 筆（取自 "
          f"{ref_src}） -> {args.out}")


if __name__ == "__main__":
    main()
