#!/bin/bash

# 範例參數
# SEED=536831999
# INPUT=/home/pp25/pp25s058/hw/hw01/testcases/38.in
# OUTPUT=/home/pp25/pp25s058/hw/hw01/38.out
SEED=536869888
INPUT=/home/pp25/pp25s058/hw/hw01/testcases/33.in
OUTPUT=/home/pp25/pp25s058/hw/hw01/33.out

# 從 1 到 12 個 process 執行
for n in {1..8}
do
    echo "=== Running with $n processes ==="

    # Use 'timeout' to limit each run to 300 seconds
    timeout 300s srun -N 1 -n $n ./hw1_test $SEED $INPUT $OUTPUT

    # Check if the previous command timed out
    if [ $? -eq 124 ]; then
        echo " Run with $n processes timed out after 300 seconds. Skipping..."
    fi

    echo ""
done
