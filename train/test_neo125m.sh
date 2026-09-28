#!/bin/bash
set -e
cd "$(dirname "$0")"

export MODELNAME=EleutherAI/gpt-neo-125m DATASET=wiki-merged EPOCHS=1 GAS=16 TARGET_EPSILON=8.0 WANDB_MODE=${WANDB_MODE:-disabled}
export MODEL_SAVE_DIR=${TEST_DIR:-/scratch/jhu/afield6/kramesh3/dp-fact/privacy-hallucination-tradeoff/test-neo125m}

bash train.sh
EXTRA_ARGS=--dry_test_run NUM_RETURN_SEQ=2 MAX_NEW_TOKENS=32 INFERENCE_DIR="$MODEL_SAVE_DIR/inference" bash inference.sh "$MODEL_SAVE_DIR"/*_to_end
ls "$MODEL_SAVE_DIR" "$MODEL_SAVE_DIR/inference"
