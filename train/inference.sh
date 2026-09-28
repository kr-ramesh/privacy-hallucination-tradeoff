#!/bin/bash
set -e
cd "$(dirname "$0")"

MODELNAME=${MODELNAME:-EleutherAI/gpt-neo-125m}
DATASET=${DATASET:-wiki-merged}
TEMPERATURES=(${TEMPERATURES:-0.3})
NUM_RETURN_SEQ=${NUM_RETURN_SEQ:-5}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
SEQUENCE_LEN=${SEQUENCE_LEN:-512}
inference_paths=${INFERENCE_DIR:-/scratch/jhu/afield6/kramesh3/dp-fact/privacy-hallucination-tradeoff/inference-results/$(date +%Y-%m-%d_%H-%M-%S)}
mkdir -p "$inference_paths"

for lora_weights_path in "$@"; do
    model_name=$(basename "$lora_weights_path")
    [ ${#model_name} -gt 150 ] && model_name="${model_name:0:130}${model_name: -30}"
    for temperature in "${TEMPERATURES[@]}"; do
        echo "Running inference for model: $lora_weights_path with temperature: $temperature"
        python inference.py --model_name "$MODELNAME" --lora_weights_path "$lora_weights_path" --dataset_name "$DATASET" \
            --output_file "${inference_paths}/${model_name}_inference_${DATASET}_temperature_${temperature}.csv" \
            --sequence_len "$SEQUENCE_LEN" --batch_size 2 --num_beams 1 --top_p 0.9 --temperature "$temperature" --repetition_penalty 1.0 \
            --num_return_seq "$NUM_RETURN_SEQ" --max_new_tokens "$MAX_NEW_TOKENS" --min_new_tokens 1 --eval_batch_size 2 --device cuda $EXTRA_ARGS
    done
done
