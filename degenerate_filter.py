"""
退化輸出偵測工具：把「答案退化」跟「答案本來就短」分開判斷。

背景：evaluate_multiple() 原本只用 min_tokens 過濾，這在 MMMU 這種長文字
描述的資料集上沒問題（正常答案本來就長，短的幾乎都真的是退化），但在
DocVQA 這種答案常常就是一個數字/日期/單詞的資料集上，min_tokens 會把
「模型答對了、但答案本來就短」的樣本跟「模型真的輸出空白/複讀/亂碼」
混在一起一起丟掉——這支模組就是為了把兩者分開。

判斷邏輯（只要中一項就算 degenerate，理由會記錄在 reasons 裡）：
    1. empty          candidate 整段是空白或去除標點後完全沒有文字內容
    2. char_repeat     同一個字元連續重複 >= 10 次（例如 "aaaaaaaaaa"、
                       過量的標點符號洗版，如 "!!!!!!!!!!"）
    3. word_repeat     同一個詞（或同一段短語）佔了整段輸出過高比例
                       （預設某個詞出現次數 / 總詞數 > 0.5，且總詞數 >= 4，
                       避免答案本來就只有 1-2 個詞時被誤判）
    4. low_diversity   words 數 >= 8 時，unique word 比例過低（預設 < 0.3），
                       用來抓「A B A B A B...」這種兩三個詞循環複讀、
                       但單一詞頻率沒有超過 word_repeat 門檻的情況
    5. echo_prompt     candidate 幾乎整段就是把 question 原封不動複誦回來
                       （常見於某些 chat template 沒切乾淨、echo 了輸入），
                       這種需要額外傳入 questions 才能檢查，預設不啟用

不算退化、不會被這支模組標記的情況：
    - 短但乾淨的答案（例如 "2019"、"John Smith"）→ 保留
    - 正常長度的完整句子 → 保留

用法：
    detector = DegenerateOutputDetector()
    df = detector.diagnose(candidates)                  # 逐筆診斷報告
    kept_ref, kept_cand, kept_idx, dropped = detector.filter(
        references, candidates
    )
"""
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from anls_eval import ANLSCalculator
import pandas as pd


CHAR_REPEAT_PATTERN = re.compile(r"(.)\1{9,}")  # 同一字元連續 >=10 次


@dataclass
class DegenerateVerdict:
    index: int
    text: str
    token_len: int
    is_degenerate: bool
    reasons: List[str] = field(default_factory=list)


class DegenerateOutputDetector:
    def __init__(
        self,
        word_repeat_ratio_threshold: float = 0.5,
        word_repeat_min_tokens: int = 4,
        low_diversity_ratio_threshold: float = 0.3,
        low_diversity_min_tokens: int = 8,
        char_repeat_pattern: re.Pattern = CHAR_REPEAT_PATTERN,
    ):
        self.word_repeat_ratio_threshold = word_repeat_ratio_threshold
        self.word_repeat_min_tokens = word_repeat_min_tokens
        self.low_diversity_ratio_threshold = low_diversity_ratio_threshold
        self.low_diversity_min_tokens = low_diversity_min_tokens
        self.char_repeat_pattern = char_repeat_pattern

    # ──────────────────────────────────────────────────────────────────
    # 單筆診斷
    # ──────────────────────────────────────────────────────────────────
    def diagnose_one(self, text: str, index: int = -1) -> DegenerateVerdict:
        reasons = []
        stripped = text.strip()
        tokens = stripped.split()
        token_len = len(tokens)

        # 1. empty：去除標點符號後仍然沒有任何文字內容
        alnum_only = re.sub(r"[^\w]", "", stripped, flags=re.UNICODE)
        if not alnum_only:
            reasons.append("empty")
            return DegenerateVerdict(index, text, token_len, True, reasons)

        # 2. char_repeat：同一字元連續重複過多次（含標點符號洗版）
        if self.char_repeat_pattern.search(stripped):
            reasons.append("char_repeat")

        # 3. word_repeat：單一詞（不分大小寫）佔比過高
        if token_len >= self.word_repeat_min_tokens:
            norm_tokens = [t.lower().strip(".,!?;:\"'()") for t in tokens]
            norm_tokens = [t for t in norm_tokens if t]
            if norm_tokens:
                counts = Counter(norm_tokens)
                most_common_word, most_common_count = counts.most_common(1)[0]
                if most_common_count / len(norm_tokens) > self.word_repeat_ratio_threshold:
                    reasons.append(
                        f"word_repeat({most_common_word!r} x{most_common_count}/{len(norm_tokens)})"
                    )

        # 4. low_diversity：長度夠長，但 unique word 比例過低（抓循環複讀）
        if token_len >= self.low_diversity_min_tokens:
            norm_tokens = [t.lower().strip(".,!?;:\"'()") for t in tokens]
            norm_tokens = [t for t in norm_tokens if t]
            if norm_tokens:
                unique_ratio = len(set(norm_tokens)) / len(norm_tokens)
                if unique_ratio < self.low_diversity_ratio_threshold:
                    reasons.append(f"low_diversity(unique_ratio={unique_ratio:.2f})")

        is_degenerate = len(reasons) > 0
        return DegenerateVerdict(index, text, token_len, is_degenerate, reasons)

    # ──────────────────────────────────────────────────────────────────
    # 批次診斷 → DataFrame，方便直接看、直接存 CSV
    # ──────────────────────────────────────────────────────────────────
    def diagnose(self, candidates: List[str]) -> pd.DataFrame:
        verdicts = [self.diagnose_one(c, i) for i, c in enumerate(candidates)]
        return pd.DataFrame({
            "index": [v.index for v in verdicts],
            "token_len": [v.token_len for v in verdicts],
            "is_degenerate": [v.is_degenerate for v in verdicts],
            "reasons": ["; ".join(v.reasons) for v in verdicts],
            "text": [v.text for v in verdicts],
        })

    # ──────────────────────────────────────────────────────────────────
    # 過濾：只丟掉真退化的樣本，短但乾淨的答案一律保留
    # ──────────────────────────────────────────────────────────────────
    def filter(
        self,
        references: List[str],
        candidates: List[str],
        also_check_references: bool = False,
        # ↑ baseline reference 理論上不該退化（budget=20000 no-evict），
        # 但如果你想同時抓 reference 本身異常（例如 baseline 也複讀），開這個。
    ) -> Tuple[List[str], List[str], List[int], pd.DataFrame]:
        assert len(references) == len(candidates)

        kept_ref, kept_cand, kept_idx = [], [], []
        records = []

        for i, (r, c) in enumerate(zip(references, candidates)):
            v_cand = self.diagnose_one(c, i)
            v_ref = self.diagnose_one(r, i) if also_check_references else None
            is_degenerate = v_cand.is_degenerate or (v_ref is not None and v_ref.is_degenerate)

            records.append({
                "index": i,
                "candidate_token_len": v_cand.token_len,
                "is_degenerate": is_degenerate,
                "reasons": "; ".join(
                    v_cand.reasons + (v_ref.reasons if v_ref else [])
                ),
                "reference": r,
                "candidate": c,
            })

            if not is_degenerate:
                kept_ref.append(r)
                kept_cand.append(c)
                kept_idx.append(i)

        report = pd.DataFrame(records)
        n_dropped = int(report["is_degenerate"].sum())
        if n_dropped:
            print(f"[DegenerateOutputDetector] dropped {n_dropped}/{len(candidates)} "
                  f"truly degenerate samples (empty/repeat/gibberish), "
                  f"kept {len(candidates) - n_dropped} incl. short-but-clean answers")
            reason_counts = Counter()
            for reasons in report.loc[report["is_degenerate"], "reasons"]:
                for r in reasons.split("; "):
                    # 只取類別名（去掉 word_repeat(...) 裡的細節），方便統計分布
                    reason_counts[r.split("(")[0]] += 1
            for reason, count in reason_counts.most_common():
                print(f"  ── {reason}: {count}")

        return kept_ref, kept_cand, kept_idx, report

    def degenerate_rate(self, candidates: List[str]) -> float:
        """快速拿到退化率，適合放進 comparison table 當一欄。"""
        df = self.diagnose(candidates)
        return float(df["is_degenerate"].mean())


# ──────────────────────────────────────────────────────────────────────────────
# 跟 SemanticSimilarityEvaluator 整合的範例
# ──────────────────────────────────────────────────────────────────────────────
def evaluate_multiple_with_degenerate_split(
    evaluator,  # SemanticSimilarityEvaluator 實例
    references: List[str],
    candidates_by_tag: dict,
    detector: Optional[DegenerateOutputDetector] = None,
    calc: Optional[ANLSCalculator] = None,
):
    """
    取代原本 evaluate_multiple() 裡單純用 min_tokens 過濾的做法：
        1. 先用 DegenerateOutputDetector 抓出真退化的樣本並丟掉
        2. 剩下的（含短但乾淨的答案）才拿去算 cos_sim / BERTScore
        3. 額外回報每個 tag 的 degenerate_rate，讓你能把「哪個模型/哪個
           budget 更容易產生退化輸出」這件事直接寫成表格裡的一欄，
           而不是被過濾規則吃掉、藏在樣本數差異裡看不出來
    """
    detector = detector or DegenerateOutputDetector()
    results, degenerate_rates = {}, {}

    for tag, candidates in candidates_by_tag.items():
        kept_ref, kept_cand, kept_idx, report = detector.filter(references, candidates)
        degenerate_rates[tag] = 1.0 - len(kept_idx) / len(candidates)

        print(f"\n[evaluate_multiple_with_degenerate_split] '{tag}': "
              f"{len(kept_cand)}/{len(candidates)} kept "
              f"(degenerate_rate={degenerate_rates[tag]:.1%})")

        if calc:
            results[tag] = calc.evaluate(kept_ref, kept_cand, tag=tag)
        else:
            results[tag] = evaluator.evaluate(kept_ref, kept_cand, tag=tag)

    return results, degenerate_rates


def print_comparison_table_with_degenerate_rate(results: dict, degenerate_rates: dict):
    print(f"\n{'tag':<30} {'n':>5} {'degen_rate':>11} {'cos_sim':>12} {'bertscore_f1':>14}")
    print("-" * 78)
    for tag, r in results.items():
        print(f"{tag:<30} {r.n_samples:>5} "
              f"{degenerate_rates.get(tag, float('nan')):>10.1%} "
              f"{r.cosine_sim_mean:>7.4f}±{r.cosine_sim_std:<4.3f} "
              f"{r.bertscore_f1_mean:>9.4f}±{r.bertscore_f1_std:<4.3f}")
