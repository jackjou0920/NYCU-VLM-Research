# 多尺度 SVMA 的實驗腳本（2026-10-10 ~ 10-13，已封存）

這些腳本用的是整理前的 `internvl_stream.py` CLI（`--run_stream` / `--run_standard` / `--coverage_tiles` /
`--detail_scale` / `--lowres_fill` / `--use_ds` …），整理後的版本不再支援，要重跑請 checkout 2026-10-13 之前的 commit。
結果在 `results/multiscale/`、`results/baselines/lowres*/`，結論記在 memory（multiscale-svma-v0-results）。

| 腳本 | 用途 | 結果位置 |
|---|---|---|
| `run_internvl.sh` | 舊版 SVMA（m=1）各 budget + uncompressed | `results/svma_v1/`、`results/svma_v2/` |
| `run_lowres.sh` | Low-Res / Low-Res-Fill baseline（`FILL=1`） | `results/baselines/lowres/`、`results/baselines/lowresfill/` |
| `run_multiscale.sh` | 掃覆蓋 tile 數 m（v0 / v1）、細節網格 `DETAIL_SCALE` | `results/multiscale/v0/`、`results/multiscale/v1/` |
| `analyze_confidence.py` | 依答案信心度切換 m 的離線分析（結論：沒用） | `results/multiscale/confidence/` |

整理後的對應：`run_svma.sh`（`MODES="svma uncompressed lowres_fill"`）+ `summarize_results.py`。
