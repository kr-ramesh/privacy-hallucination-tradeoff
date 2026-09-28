import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import ALL_DATASETS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--dataset_name", type=str, required=True, choices=list(ALL_DATASETS.keys()))
    parser.add_argument("--lora_weights_path", type=str)
    parser.add_argument("--path_to_dataset", type=str)
    parser.add_argument("--output_file", type=str, default="results.csv")
    parser.add_argument("--sequence_len", type=int, default=1025)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--dry_test_run", action="store_true")
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--num_return_seq", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_model_and_tokenizer(model_name, lora_weights_path, device="cuda"):
    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if lora_weights_path is not None:
        print(f"Applying LoRA from: {lora_weights_path}")
        model.load_adapter(lora_weights_path)
    model.eval()
    model.to(device)
    return model, tokenizer


def main():
    args = parse_args()
    model, tokenizer = load_model_and_tokenizer(args.model_name, args.lora_weights_path, device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Reading test file: {args.path_to_dataset}")
    dataset = ALL_DATASETS[args.dataset_name](args, tokenizer)
    output_df = dataset.compute_test_metrics(model, tokenizer, args)
    print(f"Saving results to: {args.output_file}")
    output_df.to_csv(args.output_file, index=False)


if __name__ == "__main__":
    main()
