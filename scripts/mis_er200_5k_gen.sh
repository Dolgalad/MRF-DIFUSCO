#!/usr/bin/env bash
set -euo pipefail

ROOT="$HOME/MRF-DIFUSCO"
FRAMEWORK="$ROOT/data/mis-benchmark-framework"
DATASET="/data1/schulz/MRF-DIFUSCO-data/mis_er200_5k"

P=0.15
N=200

mkdir -p \
  "$DATASET/train" \
  "$DATASET/train_labels" \
  "$DATASET/val" \
  "$DATASET/val_labels" \
  "$DATASET/test" \
  "$DATASET/test_labels"

generate_split () {
    split="$1"
    count="$2"

    python -u "$FRAMEWORK/main.py" gendata \
      random \
      None \
      "$DATASET/$split" \
      --model er \
      --min_n "$N" \
      --max_n "$N" \
      --num_graphs "$count" \
      --er_p "$P"
}

label_split () {
    split="$1"

    python -u "$FRAMEWORK/main.py" solve \
      kamis \
      "$DATASET/$split" \
      "$DATASET/${split}_labels" \
      --time_limit 60

}

generate_split train 4000
generate_split val 500
generate_split test 500

label_split train
label_split val
label_split test
