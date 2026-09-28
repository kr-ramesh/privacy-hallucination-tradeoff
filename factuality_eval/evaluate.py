import logging
import pickle
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from transformers import HfArgumentParser

from .factscorer import FactScorer
from .utils import save_to_pickle, titles_in_db, truncate_to_last_sentence

logging.getLogger("LiteLLM").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


@dataclass
class FactScoreArguments:
    path_to_input_csv: str = "input.csv"
    path_to_output_csv: str = "output.csv"
    path_to_output_pkl: str = "output.pkl"
    db_file_path: Optional[str] = None
    model_name: str = "retrieval+metallama"
    knowledge_source: str = "enwiki-20230401"
    topic_field: str = "title"
    text_field: str = "output_text"
    is_test: bool = False
    n_repeat: int = 3
    gamma: int = 10
    cache_dir: str = ".cache/factscore"
    abstain_detection_type: Optional[str] = None
    use_core: bool = True
    rescore_from: Optional[str] = None


def load_generations(args):
    df = pd.read_csv(args.path_to_input_csv)
    if args.topic_field not in df.columns:
        raise KeyError(f"topic_field '{args.topic_field}' is not a column of {args.path_to_input_csv}: {list(df.columns)}")
    df["topic"] = df[args.topic_field]
    df = df.rename(columns={args.text_field: "text"}).dropna(subset=["text"])
    df["text"] = df["text"].apply(truncate_to_last_sentence)
    print(f"{len(df)} generations loaded")
    df = df[titles_in_db(args.db_file_path, df[args.topic_field].tolist())]
    if args.knowledge_source == "wikipedia-ai":
        df = df[df["category"].isna()]
    if args.knowledge_source == "only-wiki-merged":
        df = df[df["timestamp"].isna()]
    print(f"{len(df)} generations have a matching document in {args.db_file_path}")
    if args.is_test:
        df = df.head(5)
    return pd.DataFrame(np.repeat(df.values, args.n_repeat, axis=0), columns=df.columns)


def load_saved_claims(prefix):
    with open(prefix + ".pkl", "rb") as f:
        decisions = pickle.load(f)["decisions"]
    df = pd.read_csv(prefix + ".csv")
    return df["topics"].tolist(), df["bios"].tolist(), [[d["atom"] for d in claim_set] for claim_set in decisions if claim_set]


def main():
    args = HfArgumentParser(FactScoreArguments).parse_args_into_dataclasses()[0]
    scorer = FactScorer(model_name=args.model_name, cache_dir=args.cache_dir, db_path=args.db_file_path,
                        abstain_detection_type=args.abstain_detection_type, use_core=args.use_core)
    if args.rescore_from:
        topics, bios, claims = load_saved_claims(args.rescore_from)
        out = scorer.score_facts(topics, bios, claims, gamma=args.gamma, knowledge_source=args.knowledge_source)
    else:
        df = load_generations(args)
        topics, bios = df["topic"].tolist(), df["text"].tolist()
        out = scorer.get_score(topics, bios, gamma=args.gamma, knowledge_source=args.knowledge_source)

    rows = zip(topics, bios, out["scores"], out["init_scores"], out["num_facts_per_response"])
    pd.DataFrame(rows, columns=["topics", "bios", "factscore", "factscore_wo_length_penalty", "num_facts_per_response"]).to_csv(args.path_to_output_csv)
    save_to_pickle(out, args.path_to_output_pkl)
    print(f"FActScore = {100 * out['score']:.1f}% | w/o length penalty = {100 * out['init_score']:.1f}% | "
          f"respond ratio = {100 * out['respond_ratio']:.1f}% | facts per response = {out['num_facts_per_response_aggregated']:.1f}")
    print(f"Saved {args.path_to_output_csv} and {args.path_to_output_pkl}")


if __name__ == "__main__":
    main()
