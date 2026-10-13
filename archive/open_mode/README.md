# open prompt 模式（已移除，2026-10-13）

HR-Bench 不帶選項、DocVQA / InfoVQA 不加作答指示的自由生成模式；沒有 GT 可比，改用「輸出 vs uncompressed
reference」的語意相似度評分。最終方法只用官方協定（HR-Bench letter + CircularEval、DocVQA / InfoVQA short + ANLS），
所以 internvl_stream.py 的 dedup_hrbench_open()、baseline 腳本裡的 open 分支與相似度區段都已移除。

- `merge_sim_inputs.py`：把各 budget 的 JSON 合併成一份給 semantic_similarity_evaluator.py（只有 open 模式用）
- 舊的 open 結果：results/svma_v1/*_open_*、results/svma_v1/internvl_svma_hrbench/*open*
- evaluate_hrbench.py 仍保留 --prompt_mode open，舊結果還能重新評分
