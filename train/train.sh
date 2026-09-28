#!/bin/bash
set -e
cd "$(dirname "$0")"

DATASET=${DATASET:-wikipedia-large-v2}
MODELNAME=${MODELNAME:-EleutherAI/gpt-neo-125m}
gradient_accumulation_steps=${GAS:-4096}
TARGET_EPSILON=${TARGET_EPSILON:-8.0}
LR=${LR:-1e-4}
EPOCHS=${EPOCHS:-20}
SEQUENCE_LEN=${SEQUENCE_LEN:-512}
lora_r=${LORA_R:-4}
lora_alpha=${LORA_ALPHA:-512}
max_grad_norm=${MAX_GRAD_NORM:-1.0}
PROJECT=${PROJECT:-dp-fact}
model_save_dir=${MODEL_SAVE_DIR:-/scratch/jhu/afield6/kramesh3/dp-fact/privacy-hallucination-tradeoff}
export WANDB_MODE=${WANDB_MODE:-offline}
mkdir -p "$model_save_dir"

model_name="${MODELNAME//\//_}"
BATCHSIZE=${gradient_accumulation_steps}
SAVEDIR="${model_save_dir}/${DATASET}_model_${model_name}_lora_r_${lora_r}_lora_alpha_${lora_alpha}_batchsize_${BATCHSIZE}_epochs_${EPOCHS}_target_epsilon_${TARGET_EPSILON}_max_grad_norm_${max_grad_norm}_sequence_len_${SEQUENCE_LEN}_learning_rate_${LR}_epochs_${EPOCHS}"
master_port=$(shuf -i 10000-20000 -n 1)

echo "Using master port: $master_port"
echo "Training $model_name on $DATASET (epsilon=$TARGET_EPSILON); saving to $SAVEDIR"

python -m torch.distributed.launch --nproc_per_node=1 --master_port "$master_port" train.py \
    --dataset_name "$DATASET" --model_name "$MODELNAME" --path_to_save_model "$SAVEDIR" --sequence_len "$SEQUENCE_LEN" \
    --per_device_train_batch_size "$BATCHSIZE" --gradient_accumulation_steps "$gradient_accumulation_steps" --epochs "$EPOCHS" --lr "$LR" \
    --project_name "$PROJECT" --target_epsilon "$TARGET_EPSILON" --clipping flat --max_grad_norm "$max_grad_norm" \
    --lora_r "$lora_r" --lora_alpha "$lora_alpha" --pld true "$@"

echo "Training completed. Model saved to $SAVEDIR"
