"""
語意相似度評測工具：Sentence-Embedding Cosine Similarity + BERTScore

用法：
    evaluator = SemanticSimilarityEvaluator()
    results = evaluator.evaluate(baseline_answers, compressed_answers)
    evaluator.print_report(results)
    evaluator.to_dataframe(results).to_csv("eval_results.csv", index=False)

安裝：
    pip install sentence-transformers bert-score --break-system-packages
"""
import torch
import json
import argparse
import pandas as pd
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from bert_score import BERTScorer
from sentence_transformers import util, SentenceTransformer
from anls_eval import ANLSCalculator
from degenerate_filter import DegenerateOutputDetector
from degenerate_filter import evaluate_multiple_with_degenerate_split, print_comparison_table_with_degenerate_rate
from stream_adapters import DEVICE


@dataclass
class EvalResult:
    """單一組比較（例如某個 budget/score_fn/merge_mode 設定）的完整評測結果。"""
    tag: str                                    # 這組結果的標籤，例如 "budget=1024_evict_l2norm"
    n_samples: int
    cosine_sim_per_sample: List[float]
    cosine_sim_mean: float
    cosine_sim_std: float
    bertscore_precision_per_sample: List[float]
    bertscore_recall_per_sample: List[float]
    bertscore_f1_per_sample: List[float]
    bertscore_f1_mean: float
    bertscore_f1_std: float
    references: List[str] = field(repr=False)
    candidates: List[str] = field(repr=False)


class SemanticSimilarityEvaluator:
    """
    同時計算：
      1. Sentence-embedding cosine similarity —— 快、適合大量消融實驗先篩選
      2. BERTScore                              —— 較精細，適合最終報告的數字

    模型只在第一次用到時才載入（lazy loading），避免你只想用其中一種指標時
    也要付兩份模型載入的時間/顯存成本。
    """

    def __init__(
        self,
        sentence_model_name: str = "all-mpnet-base-v2",
        bertscore_model_type: str = "microsoft/deberta-xlarge-mnli",
        device: Optional[str] = None,
        bertscore_lang: str = "en",
    ):
        self.sentence_model_name = sentence_model_name
        self.bertscore_model_type = bertscore_model_type
        self.bertscore_lang = bertscore_lang
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._sentence_model = None    # lazy init
        self._bertscore_scorer = None  # lazy init

    # ──────────────────────────────────────────────────────────────────
    # Lazy loaders
    # ──────────────────────────────────────────────────────────────────
    @property
    def sentence_model(self):
        if self._sentence_model is None:
            print(f"[SemanticSimilarityEvaluator] Loading sentence-transformer "
                  f"'{self.sentence_model_name}' ...")
            self._sentence_model = SentenceTransformer(
                self.sentence_model_name, device=self.device
            )
        return self._sentence_model

    @property
    def bertscore_scorer(self):
        if self._bertscore_scorer is None:
            print(f"[SemanticSimilarityEvaluator] Loading BERTScore model "
                  f"'{self.bertscore_model_type}' ...")
            self._bertscore_scorer = BERTScorer(
                model_type=self.bertscore_model_type,
                lang=self.bertscore_lang,
                device=self.device,
                rescale_with_baseline=False,
            )
        return self._bertscore_scorer

    # ──────────────────────────────────────────────────────────────────
    # 個別指標（也可以單獨呼叫，不一定要透過 evaluate() 一次跑兩個）
    # ──────────────────────────────────────────────────────────────────
    def compute_cosine_similarity(
        self, references: List[str], candidates: List[str]
    ) -> List[float]:
        """references[i] 對應 candidates[i]，逐筆算 cosine similarity。"""
        assert len(references) == len(candidates), "references 和 candidates 長度必須一致"

        emb_ref = self.sentence_model.encode(
            references, convert_to_tensor=True, show_progress_bar=False
        )
        emb_cand = self.sentence_model.encode(
            candidates, convert_to_tensor=True, show_progress_bar=False
        )
        sims = util.cos_sim(emb_ref, emb_cand).diagonal()
        return sims.cpu().tolist()

    def compute_bertscore(
        self, references: List[str], candidates: List[str]
    ) -> Dict[str, List[float]]:
        """回傳逐筆的 precision / recall / f1。"""
        assert len(references) == len(candidates), "references 和 candidates 長度必須一致"

        P, R, F1 = self.bertscore_scorer.score(candidates, references)
        return {
            "precision": P.tolist(),
            "recall": R.tolist(),
            "f1": F1.tolist(),
        }

    # ──────────────────────────────────────────────────────────────────
    # 主要入口：一次跑兩種指標
    # ──────────────────────────────────────────────────────────────────
    def evaluate(
        self,
        references: List[str],     # 通常是 baseline（未壓縮）的輸出
        candidates: List[str],     # 通常是 compressed（memory bank / resampler）的輸出
        tag: str = "unnamed",
    ) -> EvalResult:
        assert len(references) == len(candidates), (
            f"references ({len(references)}) 和 candidates ({len(candidates)}) 長度不一致"
        )

        cos_sims = self.compute_cosine_similarity(references, candidates)
        bert = self.compute_bertscore(references, candidates)

        cos_mean, cos_std = _mean_std(cos_sims)
        f1_mean, f1_std = _mean_std(bert["f1"])

        return EvalResult(
            tag=tag,
            n_samples=len(references),
            cosine_sim_per_sample=cos_sims,
            cosine_sim_mean=cos_mean,
            cosine_sim_std=cos_std,
            bertscore_precision_per_sample=bert["precision"],
            bertscore_recall_per_sample=bert["recall"],
            bertscore_f1_per_sample=bert["f1"],
            bertscore_f1_mean=f1_mean,
            bertscore_f1_std=f1_std,
            references=references,
            candidates=candidates,
        )

    def evaluate_multiple(
        self,
        references: List[str],
        candidates_by_tag: Dict[str, List[str]],
        min_tokens: int = 5,
        max_tokens: Optional[int] = None,
    ) -> Dict[str, EvalResult]:
        """一次比較多組設定，每組先各自濾掉極短（或極長）的樣本，
        對應的 baseline reference 也同筆一起丟掉，例如：
        candidates_by_tag = {
            "budget=512_evict":  [...],
            "budget=512_merge":  [...],
            "budget=1024_evict": [...],
            "budget=1024_merge": [...],
        }
        每一組都跟同一份 references（baseline）比較。
        """
        results = {}
        for tag, candidates in candidates_by_tag.items():
            f_ref, f_cand, kept_idx = self.filter_extreme_length(
                references, candidates, min_tokens=min_tokens, max_tokens=max_tokens
            )
            print(f"\n[evaluate_multiple] Evaluating '{tag}' "
                f"({len(f_cand)}/{len(candidates)} samples after filtering) ...")
            results[tag] = self.evaluate(f_ref, f_cand, tag=tag)
        return results

    # ──────────────────────────────────────────────────────────────────
    # 輸出／報告
    # ──────────────────────────────────────────────────────────────────
    def print_report(self, result: EvalResult, show_worst_k: int = 3):
        print(f"\n{'='*70}")
        print(f"[{result.tag}]  n_samples={result.n_samples}")
        print(f"{'='*70}")
        print(f"  Cosine similarity : mean={result.cosine_sim_mean:.4f}  "
              f"std={result.cosine_sim_std:.4f}")
        print(f"  BERTScore F1      : mean={result.bertscore_f1_mean:.4f}  "
              f"std={result.bertscore_f1_std:.4f}")

        if show_worst_k > 0:
            worst_idx = sorted(
                range(result.n_samples),
                key=lambda i: result.cosine_sim_per_sample[i],
            )[:show_worst_k]
            print(f"\n  The top {show_worst_k} worst:")
            for i in worst_idx:
                print(f"  ── sample {i}  cos_sim={result.cosine_sim_per_sample[i]:.4f}  "
                      f"bertscore_f1={result.bertscore_f1_per_sample[i]:.4f}")
                print(f"     reference : {result.references[i][:150]}...")
                print(f"     candidate : {result.candidates[i][:150]}...")

    def print_comparison_table(self, results: Dict[str, EvalResult]):
        """把 evaluate_multiple() 的結果整理成一張表，方便直接放進論文。"""
        print(f"\n{'tag':<30} {'n':>5} {'cos_sim':>12} {'bertscore_f1':>14}")
        print("-" * 65)
        for tag, r in results.items():
            print(f"{tag:<30} {r.n_samples:>5} "
                  f"{r.cosine_sim_mean:>7.4f}±{r.cosine_sim_std:<4.3f} "
                  f"{r.bertscore_f1_mean:>9.4f}±{r.bertscore_f1_std:<4.3f}")

    def to_dataframe(self, result: EvalResult):
        """轉成 pandas DataFrame，方便存 CSV 或畫圖。"""
        return pd.DataFrame({
            "tag": result.tag,
            "sample_idx": range(result.n_samples),
            "cosine_similarity": result.cosine_sim_per_sample,
            "bertscore_precision": result.bertscore_precision_per_sample,
            "bertscore_recall": result.bertscore_recall_per_sample,
            "bertscore_f1": result.bertscore_f1_per_sample,
            "reference": result.references,
            "candidate": result.candidates,
        })

    def multiple_to_dataframe(self, results: Dict[str, EvalResult]):
        """把 evaluate_multiple() 的多組結果合併成一張長格式 DataFrame，
        方便用 seaborn/matplotlib 畫「budget vs 相似度」的趨勢圖。"""
        frames = [self.to_dataframe(r) for r in results.values()]
        return pd.concat(frames, ignore_index=True)

    def filter_extreme_length(
        self,
        references: List[str],
        candidates: List[str],
        min_tokens: int = 5,
        max_tokens: Optional[int] = None,
    ) -> tuple[List[str], List[str], List[int]]:
        """濾掉 reference 或 candidate 任一方 token 數低於 min_tokens
        （或高於 max_tokens）的樣本。回傳過濾後的 references, candidates,
        以及被保留樣本的原始 index（方便追蹤是哪幾筆被丟掉）。
        """
        kept_ref, kept_cand, kept_idx = [], [], []
        dropped = []
        for i, (r, c) in enumerate(zip(references, candidates)):
            r_len, c_len = len(r.split()), len(c.split())
            too_short = r_len < min_tokens or c_len < min_tokens
            too_long = max_tokens is not None and (r_len > max_tokens or c_len > max_tokens)
            if too_short or too_long:
                dropped.append((i, r_len, c_len))
                continue
            kept_ref.append(r)
            kept_cand.append(c)
            kept_idx.append(i)

        if dropped:
            print(f"[filter_extreme_length] dropped {len(dropped)}/{len(references)} samples "
                f"(min_tokens={min_tokens}, max_tokens={max_tokens})")
            for i, r_len, c_len in dropped:
                print(f"  ── idx={i}  ref_len={r_len}  cand_len={c_len}")

        return kept_ref, kept_cand, kept_idx


def _mean_std(values: List[float]):
    t = torch.tensor(values, dtype=torch.float32)
    return t.mean().item(), t.std().item()


def load_eval_json(json_path: str):
    """
    Load evaluation json.

    Format:
    {
        "references": [...],
        "candidates": {
            "budget=512": [...],
            "budget=1024": [...]
        }
    }
    """
    json_path = Path(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    references = data["references"]
    candidates = data["candidates"]

    return references, candidates


# ──────────────────────────────────────────────────────────────────────────────
# 使用範例
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # baseline_answers = [
    #     "The image shows a red sports car parked on a city street at sunset, "
    #     "with tall glass buildings reflecting orange light in the background.",
    #     "A golden retriever is running across a grassy field, chasing a tennis ball, "
    #     "with a wooden fence visible in the distance.",
    # ]

    # compressed_answers = {
    #     "budget=512_evict": [
    #         "A red car is parked on a street with buildings in the background.",
    #         "A dog is running in a field chasing a ball.",
    #     ],
    #     "budget=1024_merge": [
    #         "The image shows a red sports car parked on a city street at sunset, "
    #         "with glass buildings reflecting light in the background.",
    #         "A golden retriever runs across a grassy field chasing a tennis ball, "
    #         "with a fence visible in the distance.",
    #     ],
    # }

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_json", type=str, required=True, help="evaluation json")
    parser.add_argument("--output_csv", type=str, default="semantic_eval_results.csv")
    args = parser.parse_args()

    # detector = DegenerateOutputDetector()

    references, candidates = load_eval_json(args.input_json)
    evaluator = SemanticSimilarityEvaluator(device=DEVICE)

    # # 拿 budget=1024 那組看看:min_tokens<3 丟掉的 3585 筆裡,
    # # 真正 empty/複讀/亂碼的有多少,vs 短但乾淨的有多少
    # df = detector.diagnose(candidates["budget=1024_evict_info_density"])
    # print(df["is_degenerate"].value_counts())

    # # 具體看幾筆「被舊規則(min_tokens<3)誤殺、但新規則判定為乾淨」的樣本
    # short_but_clean = df[(df["token_len"] < 3) & (~df["is_degenerate"])]
    # print(short_but_clean[["token_len", "text"]].head(20))

    # calc = ANLSCalculator()

    # results, degen_rates = evaluate_multiple_with_degenerate_split(
    #     evaluator, references, candidates, calc=None  # candidates 是四個 budget 的 dict
    # )
    # print_comparison_table_with_degenerate_rate(results, degen_rates)

    # calc.print_comparison_table(results)

    # tag = "budget=1024_none"
    # all_results = evaluator.evaluate(references, candidates[tag], tag)
    # evaluator.print_report(all_results)

    all_results = evaluator.evaluate_multiple(references, candidates, min_tokens=1)
    evaluator.print_comparison_table(all_results)

    # # 多組比較（不同 budget/merge 設定一次跑完，一張表比較）
    # all_results = evaluator.evaluate_multiple(baseline_answers, compressed_answers)
    # evaluator.print_comparison_table(all_results)

    # # 存成 CSV，方便後續畫圖或放進論文附錄
    # df = evaluator.multiple_to_dataframe(all_results)
    # df.to_csv("semantic_eval_results.csv", index=False)
    # print("\nSaved to semantic_eval_results.csv")
