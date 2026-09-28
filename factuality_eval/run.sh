#!/bin/bash
set -e
INPUT_CSV=$(realpath "${1:?usage: run.sh <generations.csv> <knowledge.db> [output_prefix] [extra args]}")
DB_FILE=$(realpath "${2:?usage: run.sh <generations.csv> <knowledge.db> [output_prefix] [extra args]}")
OUTPUT_PREFIX=$(realpath -m "${3:-${INPUT_CSV%.csv}_factscore}")
MODEL=${MODEL:-retrieval+metallama}
KNOWLEDGE_SOURCE=${KNOWLEDGE_SOURCE:-$(basename "${DB_FILE%.db}")}
TOPIC_FIELD=${TOPIC_FIELD:-title}
N_REPEAT=${N_REPEAT:-3}
cd "$(dirname "$0")/.."

python -m factuality_eval.evaluate --path_to_input_csv "$INPUT_CSV" --db_file_path "$DB_FILE" \
    --path_to_output_csv "${OUTPUT_PREFIX}.csv" --path_to_output_pkl "${OUTPUT_PREFIX}.pkl" \
    --model_name "$MODEL" --knowledge_source "$KNOWLEDGE_SOURCE" --topic_field "$TOPIC_FIELD" --n_repeat "$N_REPEAT" "${@:4}"
