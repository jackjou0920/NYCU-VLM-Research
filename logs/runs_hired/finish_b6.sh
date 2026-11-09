#!/usr/bin/env bash
# B6 剩下的 tile 上限改到 GPU1（GPU0 有別人常駐的服務，永遠達不到 <1 GB）；GPU 要全空才開始，最多等 24 小時
cd /home/jackjou/project/NYCU-VLM-Research
PY=$HOME/.virtualenvs/vlm/bin/python
for n in 25 7 145; do
  f=results/concurrency_hired/n${n}_hired.csv; [ -f $f ] && continue
  s=$(date +%s)
  until [ "$(nvidia-smi -i 1 --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1024 ]; do
    [ $(( $(date +%s) - s )) -ge 86400 ] && { echo "[TIMEOUT] n$n"; continue 2; }; sleep 120; done
  echo "[$(date '+%m-%d %H:%M')] GPU1 conc n$n"
  CUDA_VISIBLE_DEVICES=1 $PY bench_concurrency.py --method hired --max_num $n --out_csv $f > logs/runs_hired/conc_n${n}.log 2>&1 || echo "[error] n$n"
done
echo "[$(date '+%m-%d %H:%M')] B6 done"
