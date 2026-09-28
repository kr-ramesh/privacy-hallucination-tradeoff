import os

import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk

DATASET_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "datasets")
INSTRUCT = "You are a helpful assistant. Write a response that appropriately completes the request.\n\n### Input:\n"
RESPONSE = "\n\n### Response:"
WIKI = dict(text_field="title", label_field="content")
MED_WIKI = dict(load_type="from_hf", dataset_name="gamino/wiki_medical_terms", text_field="page_title", label_field="page_text", max_label_words=1024)
ALL_DATASET_CONFIGS = {
    "tab": dict(text_field="control", label_field="text", prompt_end="\n"),
    "mimic": dict(text_field="LONG_TITLE", label_field="TEXT", prompt_begin="Diagnosis: ", prompt_end=" Summary :"),
    "wiki": dict(text_field="Name", label_field="Text", prompt_begin="Name : ", prompt_end="\nBiography:"),
    "wiki-instruct": dict(text_field="Name", label_field="Text", prompt_begin=INSTRUCT + "Generate a biography about ", prompt_end=RESPONSE),
    "med-wiki": dict(**MED_WIKI, prompt_begin="Medical Term: ", prompt_end="\nDescription: "),
    "med-wiki-instruct": dict(**MED_WIKI, prompt_begin=INSTRUCT + "Generate a description of this medical term: ", prompt_end=RESPONSE, n_test=500),
    "med-wiki-instruct-baseline": dict(**MED_WIKI, prompt_begin=INSTRUCT + "Generate a description of this medical term: ", prompt_end=RESPONSE, n_train=100, n_test=500),
    "med-abst-instruct": dict(load_type="from_hf", dataset_name="TimSchopf/medical_abstracts", text_field="condition_label", label_field="medical_abstract",
                              prompt_begin=INSTRUCT + "Generate a text for the following medical condition: ", prompt_end=RESPONSE),
    "med-summ": dict(text_field="note", label_field="answer", prompt_begin="Summarize the following medical text:\n", prompt_end="\nSummary:"),
    "clinical-notes": dict(text_field="Condition", label_field="full_note", prompt_begin=INSTRUCT + "Generate a SOAP note for the following medical condition: ", prompt_end=RESPONSE),
    "common-pile-news-filtered": dict(text_field="headline", label_field="text"),
    "wikipedia-pretrain-v2": dict(text_field="title", label_field="text"),
    **{name: WIKI for name in ["wiki-science", "wiki-ai", "wikipedia-large", "wiki-merged", "wikipedia-large-v2", "wikipedia-large-v2-only", "wiki-merged-split-150", "wiki-merged-split-200"]},
}


def unwrap(model, attr):
    while not hasattr(model, attr):
        model = model._module if hasattr(model, "_module") else model.module
    return model


class GenericCustomDataset:
    def __init__(self, args, tokenizer, name):
        config = ALL_DATASET_CONFIGS[name]
        self.name, self.tokenizer, self.sequence_len = name, tokenizer, args.sequence_len
        self.text_field, self.label_field = config["text_field"], config["label_field"]
        self.prompt_begin, self.prompt_end = config.get("prompt_begin", ""), config.get("prompt_end", "")
        self.path_to_test_dataset = test_path = getattr(args, "path_to_test_dataset", None)

        if config.get("load_type", "from_disk") == "from_disk":
            self.dataset = load_from_disk(getattr(args, "path_to_dataset", None) or os.path.join(DATASET_DIR, name))
        else:
            self.dataset = load_dataset(config["dataset_name"])
        if isinstance(self.dataset, Dataset):
            self.dataset = DatasetDict({"train": self.dataset})

        if test_path:
            try:
                self.dataset["test"] = load_from_disk(test_path)
            except Exception:
                self.dataset["test"] = self.create_test_dataset(test_path)
        if "test" in self.dataset and len(self.dataset["test"]) > 500:
            print(f"Dataset {name} has more than 500 elements in test set, sampling 200 for evaluation.")
            self.dataset["test"] = self.dataset["test"].shuffle(seed=42).select(range(200))
        if "test" not in self.dataset:
            if len(self.dataset["train"]) > 500:
                print(f"Dataset {name} has more than 500 elements, sampling 200 for test set.")
                self.dataset["test"] = self.dataset["train"].shuffle(seed=42).select(range(200))
            else:
                print(f"Dataset {name} has less than 500 elements, using the entire train set for test.")
                self.dataset["test"] = self.dataset["train"]

        if config.get("max_label_words"):
            self.dataset = self.dataset.filter(lambda x: len(x[self.label_field].split()) <= config["max_label_words"] and len(x[self.label_field]) != 0)
        self.dataset = self.dataset.filter(lambda x: x[self.label_field] is not None and len(x[self.label_field]) > 0)
        for split in ["train", "test"]:
            if config.get(f"n_{split}"):
                self.dataset[split] = self.dataset[split].select(range(config[f"n_{split}"]))

    def create_test_dataset(self, path_to_dataset):
        output_dir = path_to_dataset.split(".csv")[0]
        os.makedirs(output_dir, exist_ok=True)
        dataset = DatasetDict({"test": Dataset.from_pandas(pd.read_csv(path_to_dataset))})
        dataset.save_to_disk(output_dir)
        return dataset["test"]

    def preprocess_function(self, examples):
        tok, L, pad = self.tokenizer, self.sequence_len, self.tokenizer.pad_token_id
        prompts = tok([self.prompt_begin + str(x) + self.prompt_end for x in examples[self.text_field]])["input_ids"]
        targets = tok([str(x) for x in examples[self.label_field]])["input_ids"]
        out = {"input_ids": [], "attention_mask": [], "labels": []}
        for p, t in zip(prompts, targets):
            t = t[1:] if t[0] == tok.bos_token_id else t
            n = max(0, L - len(p) - len(t))
            out["input_ids"].append(torch.tensor(([pad] * n + p + t)[:L]))
            out["attention_mask"].append(torch.tensor(([0] * n + [1] * (len(p) + len(t)))[:L]))
            out["labels"].append(torch.tensor(([-100] * (n + len(p)) + t)[:L]))
        return out

    def compute_test_metrics(self, model, tokenizer, args):
        n = args.num_return_seq
        print(f"Testing for the entire dataset. Number of generations  per prompt: {n}")
        if self.path_to_test_dataset is None:
            test_dataset = Dataset.from_pandas(self.dataset["test"].to_pandas().drop_duplicates(subset=self.text_field, keep="first"))
            print(test_dataset)
        else:
            print("Loading custom test dataset...")
            df = pd.read_csv(self.path_to_test_dataset)
            test_dataset = Dataset.from_pandas(df[df[self.text_field].notna()])
        if args.dry_test_run:
            print("Test run...")
            test_dataset = test_dataset.select(range(5))

        output = {c: [v for v in test_dataset[c] for _ in range(n)] for c in test_dataset.column_names if c != self.label_field}
        queries = tokenizer([self.prompt_begin + str(x) + self.prompt_end for x in test_dataset[self.text_field]], padding=False)["input_ids"]
        print("Length of test data", len(queries))

        model = unwrap(model, "generate")
        model.eval()
        tokenizer.padding_side = "left"
        gen = dict(min_new_tokens=args.min_new_tokens, max_new_tokens=args.max_new_tokens, num_return_sequences=n,
                   eos_token_id=tokenizer.eos_token_id, bad_words_ids=[[1, 4768, 5275]], repetition_penalty=args.repetition_penalty)
        gen.update(dict(do_sample=False, num_beams=5) if args.temperature == 0.0 else dict(do_sample=True, top_k=args.top_k, top_p=args.top_p, temperature=args.temperature))
        responses, batch_size = [], min(len(queries), args.eval_batch_size)
        for i in range(0, len(queries), batch_size):
            inputs = tokenizer.pad({"input_ids": queries[i:i + batch_size]}, padding=True, return_tensors="pt").to(args.device)
            if args.temperature == 0.0:
                print("Greedy decoding...")
            with torch.no_grad():
                generations = model.generate(**inputs, **gen)
            responses += [tokenizer.decode(g, skip_special_tokens=True) for g in generations[:, inputs["input_ids"].shape[1]:]]

        output["input_prompt"] = [tokenizer.decode(q, skip_special_tokens=True) for q in queries for _ in range(n)]
        output["output_text"] = responses
        return pd.DataFrame(output)


ALL_DATASETS = {name: (lambda args, tokenizer, name=name: GenericCustomDataset(args, tokenizer, name)) for name in ALL_DATASET_CONFIGS}
