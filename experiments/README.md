# experiments

開發過程與沒有採用的實驗，不在最終流程（根目錄）裡。方法原名 SVMA（2026-10-24 改名 FoveaTile），
這裡的 script 與結果檔大多保留舊名（`svma_*`、`stream_b*`、`internvl_stream.py`），是當時的紀錄，沒有改名。

| 資料夾 | 內容 | 結果 |
|---|---|---|
| `candidates_rejected/` | 依事先定的標準判定**不採用**的候選改動：3b 打分窗口隨網格放大（`run_calibrate_stageG.sh`）、v1.1 打分 query / 聚合（`run_calibrate_v11.sh`、`analyze_v11.py`）、prior head 選擇 A1 / 基準校正 A2（`prepare_needle_calib.py`、`diagnose_prior_heads.py`、`analyze_prior_heads.py`）。`internvl_stream.py`、`svma_config.py` 是含這些選項的完整版（2026-10-25 快照），主程式已移除這些選項 | `results/calib/`（tag 含 `_gf` / `_q` / `_agg` / `_ph` / `_pb`）、`results/archive/diagnostics_prior_heads/`；判斷過程見 `docs/04_calibration_protocol.md` §7、§7b、§7c |
| `legacy_runs/` | 被 `run_test_suite.sh` / `run_budget_fill.sh` 取代的舊執行腳本（各 baseline 的 `run_*.sh`、`run_b1536.sh`、`run_metrics_all.sh`），以及 SparseVLM budget 規則的一次性修補（`sparsevlm_small_images.py`、`finalize_sparsevlm.sh`） | `results/archive/baselines*/`（已整合進 `results/runs/`） |
| `svma_v2_calib/` | v0（m = 1）時期在測試集上做的 S 校準，已被 held-out 校準取代 | `results/archive/svma_v2/` |
| `multiscale/` | 覆蓋 + 細節兩種尺度的早期掃描、Low-Res / Low-Res-Fill baseline | `results/archive/multiscale/` |
| `diagnostics/` | oracle / selfscore / layersweep / cascade / thumbprior / q_attn 診斷 | `results/archive/diagnostics_*/` |
| `ablation/` | 小規模 suite、torch.compile 測試 | `results/archive/svma_v2/` |
| `open_mode/` | HR-Bench 開放式作答的嘗試 | — |
| `other_models/` | DeepSeek-VL 版本 | — |
| `prototypes/` | 最早期的原型：`exp/`（LLaVA-OV、Qwen、memory bank 等）、`exp_module/`、`reports/`、`img_outputs/` | — |

## 重跑

這些 script 依賴根目錄的模組，而且很多是用舊名 import（`internvl_stream`、`svma_config`）。要重跑：

- `candidates_rejected/`：在根目錄用 `PYTHONPATH=experiments/candidates_rejected python experiments/candidates_rejected/<script>.py ...`，
  會用到同資料夾裡含候選選項的 `internvl_stream.py` / `svma_config.py`；`.sh` 要先複製到根目錄，並把裡面的
  `internvl_stream.py` 改成 `experiments/candidates_rejected/internvl_stream.py`。
- 其他資料夾：用 `git checkout 1e916ef -- <檔案>` 取回整理前的版本（commit `1e916ef` 是 2026-10-25 整理前的快照）。
