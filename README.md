# FoveaTile

**Look Before You Encode: Foveated Tile Streaming for Resolution-Independent VLM Inference**（目標 CVPR 2027）

在 ViT 之前就決定要編碼哪些 tile：m 塊低解析度覆蓋 tile 給全圖，prior（一次短 forward）加上邊界 q_attn 從細節網格挑
K − m 塊高解析度細節 tile，逐 tile 串流寫進 KV。沒被選中的 tile 連 ViT 都不跑，ViT、KV 與記憶體都是 O(K)，跟解析度無關。
方法原名 SVMA（2026-10-24 改名）；凍結版本 v1.0（2026-10-17），超參數見 `foveatile_config.py`。

模型：InternVL3.5-8B。環境：`~/.virtualenvs/vlm/bin/python`。

## 結構

| 路徑 | 內容 |
|---|---|
| `foveatile_internvl.py` | FoveaTile 主程式，也提供 uncompressed / Low-Res-Fill 模式與資料集註冊 |
| `foveatile_config.py` | 超參數：budget → m、S（held-out 校準的結果） |
| `internvl_preprocess.py`、`process_common.py` | 影像切 tile、資料集 loader |
| `fastv_internvl.py`、`sparsevlm_internvl.py`、`visionzip_internvl.py`、`ifprune_internvl.py`（+ `*_core.py`） | baseline，照論文與官方程式碼移植 |
| `evaluate_*.py`、`semantic_similarity_evaluator.py`、`summarize_results.py` | 評分（ANLS、CircularEval、選擇題）與彙整 |
| `bench_metrics.py`、`flops.py` | TTFT、記憶體增量、KV cache、FLOPs |
| `run_test_suite.sh` | 所有方法 × 測試集 × budget 的 accuracy |
| `run_budget_fill.sh` | 補一個 budget 的 accuracy + 效率 |
| `run_efficiency.sh`、`run_scaling.sh` | 效率總表、解析度縮放曲線 |
| `run_calibrate.sh`、`run_calibrate_stageB.sh` | held-out 校準（階段 A：m、階段 B：S） |
| `run_foveatile.sh` | 只跑 FoveaTile / uncompressed / Low-Res-Fill |
| `plot_*.py`、`results/paper/build_main_table.py` | 論文圖表 |
| `docs/` | 實驗規劃、校準流程、相關工作比較 |
| `experiments/` | 開發過程與沒有採用的候選（見 `experiments/README.md`） |

## 結果

| 路徑 | 內容 |
|---|---|
| `results/runs/<dataset>/<method>_b<B>_InternVL3_5-8B.json` | 測試集 accuracy（所有方法；`uncompressed_InternVL3_5-8B.json` 是未壓縮） |
| `results/efficiency/` | TTFT / 記憶體 / FLOPs（`b1536/`、`b2560/` 是補的 budget） |
| `results/scaling/` | 解析度縮放（HR-Bench 8K，7~188 塊 tile） |
| `results/calib/` | held-out 校準集的結果 |
| `results/diagnostics/` | V* tile 選擇診斷、VisionZip / SparseVLM 實作檢查 |
| `results/paper/main_internvl.csv` | 主表（`python results/paper/build_main_table.py`） |
| `results/archive/` | 舊實驗結果（tag 保留舊名 `svma_*` / `stream_b*`） |

## 常用指令

```bash
python foveatile_internvl.py --dataset docvqa --budget 1536          # FoveaTile，結果存 results/runs/docvqa/
python foveatile_config.py                                           # 各 budget 的 m、S
DATASETS="vstar" METHODS="foveatile fastv" BUDGETS="1536" ./run_test_suite.sh
python summarize_results.py --datasets vstar --ref uncompressed
python results/paper/build_main_table.py
```
