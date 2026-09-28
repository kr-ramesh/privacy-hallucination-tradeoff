#!/bin/bash
set -e
cd "$(dirname "$0")"
OUT_DIR=$(realpath -m "${OUT_DIR:-../outputs/factuality_test}")
mkdir -p "$OUT_DIR"
rm -f "$OUT_DIR/dummy.db"

(cd .. && python -m factuality_eval.build_db --data factuality_eval/data/dummy_knowledge.csv --title_field title --text_field text --db_path "$OUT_DIR/dummy.db")
MODEL=${MODEL:-retrieval+mistral} N_REPEAT=1 bash run.sh data/dummy_generations.csv "$OUT_DIR/dummy.db" "$OUT_DIR/dummy_factscore" --cache_dir "$OUT_DIR/cache"
cat "$OUT_DIR/dummy_factscore.csv"
