#!/bin/bash

IMAGE="img_datasets/img01.jpg"
BUDGET=4096
SCORE_FN="info_density"
MERGE_MODE="evict"

LOG_FILE="batch_test_$(date +%Y%m%d_%H%M%S).log"

echo "===== Start: $(date) =====" | tee "$LOG_FILE"

for BS in 16 20
do
    echo "" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"
    echo "Running batch_size=$BS" | tee -a "$LOG_FILE"
    echo "========================================" | tee -a "$LOG_FILE"

    python internvl_svm.py \
        --image "$IMAGE" \
        --batch_size "$BS" \
        --budget "$BUDGET" \
        --score_fn "$SCORE_FN" \
        --merge_mode "$MERGE_MODE" \
        --run_stream 2>&1 | tee -a "$LOG_FILE"

    echo "" | tee -a "$LOG_FILE"
done

echo "===== Finished: $(date) =====" | tee -a "$LOG_FILE"