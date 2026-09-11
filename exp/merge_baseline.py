"""把另一份 output JSON 的 references 併進主 JSON 當一個 candidate tag。

用途：Block A 的「直接少切」baseline（--max_num=K 直接跑、不經壓縮）只會寫進
--run_standard 的 references 欄，不是 candidates[tag]。要跟其他方法一起被
evaluate_anls.py / evaluate_mmmu.py 評分，得先把它搬進主 JSON 的 candidates。

用法：
    python merge_baseline.py --main_json out_block_a_docvqa_b1024.json \
        --baseline_json out_block_a_docvqa_fixed_b1024.json --tag "fixed_tiles=4"
"""
import json
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--main_json", required=True, help="要併入的主結果 JSON（會被原地覆寫）")
    ap.add_argument("--baseline_json", required=True, help="來源 JSON，讀它的 references")
    ap.add_argument("--tag", required=True, help="併入後的 candidate tag 名稱")
    args = ap.parse_args()

    with open(args.main_json, encoding="utf-8") as f:
        main_data = json.load(f)
    with open(args.baseline_json, encoding="utf-8") as f:
        base_data = json.load(f)

    refs = base_data.get("references", [])
    if not refs:
        raise SystemExit(f"{args.baseline_json} 沒有 references，請先對它跑 --run_standard")

    main_refs = main_data.get("references", [])
    if main_refs and len(refs) != len(main_refs):
        print(f"[warn] baseline references 長度 {len(refs)} != 主 JSON references 長度 "
              f"{len(main_refs)}，請確認兩邊 --num_images / --dataset 一致，否則對齊會錯位")

    main_data.setdefault("candidates", {})[args.tag] = refs
    with open(args.main_json, "w", encoding="utf-8") as f:
        json.dump(main_data, f, indent=2, ensure_ascii=False)
    print(f"[merged] candidates[{args.tag!r}] <- {len(refs)} rows from {args.baseline_json}")


if __name__ == "__main__":
    main()
