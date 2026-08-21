"""
ANLS (Average Normalized Levenshtein Similarity) —— DocVQA 官方標準指標。

為什麼需要這個，不能只用 cos_sim：
    DocVQA 這類任務的答案是具體事實（數字/人名/日期/公司名），本質上是
    「對」或「錯」的二元結果，沒有「部分正確」這種中間態。但 cosine
    similarity 是連續值，硬套在二元任務上會讓分數分佈變成雙峰（一群接近
    1、一群接近 0.2-0.4），平均值受雙峰比例影響，不太能直觀反映「模型
    表現好不好」，也不是 DocVQA 這個 benchmark 的標準評測方式。

ANLS 的設計精神：
    - 用 edit distance（Levenshtein distance）正規化成 [0,1] 的相似度，
      對「幾乎打對但有一兩個字元誤差」（OCR 常見情況，例如 "5177" vs
      "5177." 或 "John Smith" vs "Jon Smith"）給予部分分數，比 exact
      match 更寬容合理。
    - 但相似度低於門檻值（官方預設 0.5）時，直接視為 0 分，避免「完全
      答錯」跟「答對一半」被賦予相近的分數，這正是 cos_sim 在這個場景
      失真的地方——ANLS 用一個硬門檻把「答錯」跟「答對但有誤差」明確
      切開，不會被連續值的中間地帶模糊掉。

公式（單一樣本，單一 ground truth）：
    NL(gt, pred) = 1 - EditDistance(gt, pred) / max(len(gt), len(pred))
    ANLS_i = NL(gt, pred)          if NL(gt, pred) >= threshold
           = 0                      otherwise

若一題有多個可接受的 ground truth（DocVQA 官方標註常常一題有多個同義
答案），該題分數取所有 ground truth 中的最大值。

整個資料集的 ANLS = 所有樣本分數的平均值。

用法：
    calculator = ANLSCalculator()
    result = calculator.evaluate(references, candidates)
    calculator.print_report(result)
"""
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import pandas as pd


@dataclass
class ANLSResult:
    tag: str
    n_samples: int
    anls_per_sample: List[float]
    anls_mean: float
    exact_match_rate: float          # NL == 1.0（完全一致，正規化後）的比例
    below_threshold_rate: float      # 被判定為 0 分（低於門檻）的比例
    references: List[str] = field(repr=False)
    candidates: List[str] = field(repr=False)


def levenshtein_distance(a: str, b: str) -> int:
    """標準動態規劃版 edit distance，O(len(a) * len(b))。

    DocVQA 答案通常很短（幾個字到幾十個字元），這個複雜度完全夠用；
    如果你的 candidate 是長句子（例如拿 ANLS 套在非 DocVQA 的長文字
    輸出上），這裡的計算量會隨長度平方成長，會變慢，屆時建議只在短答案
    型任務上使用 ANLS，長描述型任務維持用 cos_sim/BERTScore。
    """
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la

    # 只保留前一列，滾動更新，省記憶體
    prev_row = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        curr_row = [i] + [0] * lb
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr_row[j] = min(
                prev_row[j] + 1,        # 刪除
                curr_row[j - 1] + 1,    # 插入
                prev_row[j - 1] + cost,  # 替換（相同字元 cost=0）
            )
        prev_row = curr_row
    return prev_row[lb]


class ANLSCalculator:
    def __init__(
        self,
        threshold: float = 0.5,
        case_sensitive: bool = False,
        strip_whitespace: bool = True,
        normalize_punctuation: bool = True,
    ):
        """
        threshold: 官方 DocVQA 預設 0.5，低於此正規化相似度直接算 0 分。
        case_sensitive: 官方評測是大小寫不敏感的，預設 False。
        strip_whitespace: 去除頭尾空白，避免多一個空格造成不必要的懲罰。
        normalize_punctuation: 去除答案兩端常見的標點符號（句號、逗號、
            引號等），OCR/生成文字常見的格式雜訊，不應該影響語意層級的
            正確性判斷。只去頭尾，不動字串中間的標點（例如 "6112-0202/202"
            這種本身就含標點的答案不受影響）。
        """
        self.threshold = threshold
        self.case_sensitive = case_sensitive
        self.strip_whitespace = strip_whitespace
        self.normalize_punctuation = normalize_punctuation

    def _normalize(self, s: str) -> str:
        s = s.strip() if self.strip_whitespace else s
        if not self.case_sensitive:
            s = s.lower()
        if self.normalize_punctuation:
            s = s.strip(".,!?;:\"'()[]{}<>")
        return s

    def normalized_similarity(self, gt: str, pred: str) -> float:
        """單一 pair 的 NL(gt, pred)，尚未套用門檻值。"""
        gt_n, pred_n = self._normalize(gt), self._normalize(pred)
        max_len = max(len(gt_n), len(pred_n))
        if max_len == 0:
            return 1.0  # 兩邊正規化後都是空字串，視為一致
        dist = levenshtein_distance(gt_n, pred_n)
        return 1.0 - dist / max_len

    def score_one(
        self,
        ground_truths: Union[str, Sequence[str]],
        prediction: str,
    ) -> float:
        """單一樣本的 ANLS 分數。ground_truths 可以是單一字串，或多個
        可接受答案的 list（分數取最大值）。"""
        if isinstance(ground_truths, str):
            ground_truths = [ground_truths]

        best_nl = max(self.normalized_similarity(gt, prediction) for gt in ground_truths)
        return best_nl if best_nl >= self.threshold else 0.0

    # ──────────────────────────────────────────────────────────────────
    # 批次評測，介面盡量跟 SemanticSimilarityEvaluator.evaluate() 對稱，
    # 方便兩者搭配著在同一份 pipeline 裡跑
    # ──────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        references: Union[List[str], List[Sequence[str]]],
        candidates: List[str],
        tag: str = "unnamed",
    ) -> ANLSResult:
        """
        references: 通常是你的 baseline（budget=20000 no-evict）輸出，
            當作 pseudo-ground-truth。也支援每筆多個可接受答案
            （references[i] 傳 list[str] 而不是 str）。
        """
        assert len(references) == len(candidates), (
            f"references ({len(references)}) 和 candidates ({len(candidates)}) 長度不一致"
        )

        scores = [
            self.score_one(gt, pred) for gt, pred in zip(references, candidates)
        ]

        exact_match = sum(1 for s in scores if s == 1.0) / len(scores)
        below_threshold = sum(1 for s in scores if s == 0.0) / len(scores)

        # 為了跟 SemanticSimilarityEvaluator 統一介面，references/candidates
        # 存進結果時一律轉成 str（多 ground truth 的情況取第一個代表）
        ref_repr = [r if isinstance(r, str) else r[0] for r in references]

        return ANLSResult(
            tag=tag,
            n_samples=len(scores),
            anls_per_sample=scores,
            anls_mean=sum(scores) / len(scores),
            exact_match_rate=exact_match,
            below_threshold_rate=below_threshold,
            references=ref_repr,
            candidates=candidates,
        )

    def evaluate_multiple(
        self,
        references: Union[List[str], List[Sequence[str]]],
        candidates_by_tag: dict,
    ) -> dict:
        return {
            tag: self.evaluate(references, candidates, tag=tag)
            for tag, candidates in candidates_by_tag.items()
        }

    # ──────────────────────────────────────────────────────────────────
    # 輸出
    # ──────────────────────────────────────────────────────────────────
    def print_report(self, result: ANLSResult, show_worst_k: int = 5):
        print(f"\n{'='*70}")
        print(f"[{result.tag}]  n_samples={result.n_samples}")
        print(f"{'='*70}")
        print(f"  ANLS               : {result.anls_mean:.4f}")
        print(f"  Exact match rate   : {result.exact_match_rate:.1%}")
        print(f"  Below threshold(=0): {result.below_threshold_rate:.1%}")

        if show_worst_k > 0:
            worst_idx = sorted(
                range(result.n_samples), key=lambda i: result.anls_per_sample[i]
            )[:show_worst_k]
            print(f"\n  The worst {show_worst_k}:")
            for i in worst_idx:
                print(f"  ── sample {i}  anls={result.anls_per_sample[i]:.4f}")
                print(f"     reference : {result.references[i][:100]}")
                print(f"     candidate : {result.candidates[i][:100]}")

    def print_comparison_table(self, results: dict):
        print(f"\n{'tag':<30} {'n':>5} {'ANLS':>8} {'exact%':>8} {'zero%':>8}")
        print("-" * 65)
        for tag, r in results.items():
            print(f"{tag:<30} {r.n_samples:>5} {r.anls_mean:>8.4f} "
                  f"{r.exact_match_rate:>7.1%} {r.below_threshold_rate:>7.1%}")

    def to_dataframe(self, result: ANLSResult) -> pd.DataFrame:
        return pd.DataFrame({
            "tag": result.tag,
            "sample_idx": range(result.n_samples),
            "anls": result.anls_per_sample,
            "reference": result.references,
            "candidate": result.candidates,
        })
