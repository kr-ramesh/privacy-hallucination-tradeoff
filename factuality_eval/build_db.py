import argparse
import os
import sqlite3

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_from_disk
from transformers import RobertaTokenizer

from .retrieval import SPECIAL_SEPARATOR

MAX_LENGTH = 256


def load_documents(path, title_field, text_field, split="train", min_chars=0):
    if path.endswith(".csv"):
        df = pd.read_csv(path)
    elif path.endswith(".jsonl"):
        df = pd.read_json(path, lines=True)
    else:
        ds = load_from_disk(path)
        df = (ds[split] if isinstance(ds, DatasetDict) else ds).to_pandas()
    df = df[df[text_field].notna() & (df[text_field].str.len() >= min_chars)]
    return df[title_field].tolist(), df[text_field].tolist()


def build_db(db_path, titles, texts):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-large")
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE documents (title PRIMARY KEY, text);")
    seen, rows = set(), []
    for title, text in zip(titles, texts):
        if title in seen:
            continue
        seen.add(title)
        passages = [[]]
        for sent in [text] if isinstance(text, str) else text:
            assert len(sent.strip()) > 0
            tokens = tokenizer(sent)["input_ids"]
            room = MAX_LENGTH - len(passages[-1])
            passages[-1].extend(tokens[:room])
            passages += [tokens[offset:offset + MAX_LENGTH] for offset in range(room, len(tokens), MAX_LENGTH)]
        rows.append((title, SPECIAL_SEPARATOR.join(tokenizer.decode(p) for p in passages if np.sum([t not in [0, 2] for t in p]) > 0)))
    cursor.executemany("INSERT INTO documents VALUES (?,?)", rows)
    conn.commit()
    conn.close()
    print(f"Saved {len(rows)} documents to {db_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--db_path", required=True)
    parser.add_argument("--title_field", default="title")
    parser.add_argument("--text_field", default="content")
    parser.add_argument("--split", default="train")
    parser.add_argument("--min_chars", type=int, default=0)
    args = parser.parse_args()
    if os.path.exists(args.db_path):
        raise FileExistsError(f"{args.db_path} already exists")
    build_db(args.db_path, *load_documents(args.data, args.title_field, args.text_field, args.split, args.min_chars))


if __name__ == "__main__":
    main()
