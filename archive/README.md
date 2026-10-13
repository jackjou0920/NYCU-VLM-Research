# archive

已完成的開發 / 診斷 script，不在目前主比較流程（根目錄 run_*.sh）裡。

- `svma_v2_calib/`：SVMA v2 的 S 校準流程（run_svma_v2.sh、run_svma_srange.sh、calibrate_qattn_tiles.py …）。
  校準結果已寫進根目錄 `svma_config.py` 的 `QATTN_TILES_BY_BUDGET`；明細在
  `results/svma_v2/infovqa/calib_s0n600/svma_calibration.json`。
- `ablation/`：小規模 suite / torch.compile 測試（結果在 `results/svma_v2/{suite_small,compile_small}`）。
- `diagnostics/`：oracle / selfscore / layersweep / cascade / thumbprior / q_attn 診斷（結果在 `results/diagnostics/`）。
- `other_models/`：DeepSeek-VL 版本。

這些 script 依賴根目錄的模組（internvl_stream.py、process_common.py …），而 .sh 用相對路徑呼叫 .py。
要重跑時把該檔搬回根目錄，或在根目錄用 `PYTHONPATH=. python archive/<dir>/<script>.py ...`（.sh 需搬回根目錄）。
