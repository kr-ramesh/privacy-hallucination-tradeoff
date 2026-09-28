import math
import os
import pickle
import random
import time
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from opacus import PrivacyEngine
from opacus.grad_sample.utils import register_grad_sampler
from opacus.utils.batch_memory_manager import BatchMemoryManager
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

try:
    from transformers.pytorch_utils import Conv1D as HFConv1D
except ImportError:
    HFConv1D = None

HF_MODEL = "gpt2"
COMPARE_INIT = True
HF_KEEP_TIED_FROZEN = False

D_MODEL = 64
N_HEADS = 4
N_LAYERS = 3
D_FF = 128

FACTS_PER_FREQ = 5
FREQUENCIES = [1, 3, 7, 15, 30, 60]
NOISE_LEVELS = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
CLIP_NORM = 1.0
EPSILON_BY_SIGMA = {0.0: math.inf, 0.1: 1310.535, 0.3: 108.122, 0.5: 29.847, 0.7: 13.433, 1.0: 6.397}

NUM_EPOCHS = 15
LR = 5e-3
LR_HF = 5e-3
LR_LIST_PRETRAINED = [LR_HF]
LR_LIST_RANDOM = [LR]
THRESHOLD = 0.5
NUM_RUNS = 1
BASE_SEED = 42
DEVICE = "cuda"
BATCH_SIZE = 64
MAX_PHYSICAL_BATCH_SIZE = 4

RESULTS_DIR = "results_cache"
COLORBLIND_PALETTE = ["#4e79a7", "#f28e2b", "#e15759", "#76b7b2", "#59a14f", "#edc949", "#af7aa1", "#ff9da7", "#9c755f", "#bab0ab"]

SUBJECTS = [
    "zephyria", "kaldor", "belvane", "thornwick", "maldren", "corvath", "selenix", "dravion", "arcthos", "velrune",
    "pyraxis", "glenmoor", "obsidyn", "halcyon", "nexara", "stratholm", "verdania", "cryosten", "luminex", "dawnridge",
    "ironvale", "novaheim", "solheim", "temporia", "crystara", "emberfell", "frostholm", "goldenreach", "shadowmere", "titanforge",
]
OBJECTS = [
    "mordath", "ventris", "draxil", "seraph", "luxon", "kestrel", "zircon", "thalis", "silvane", "fenwick",
    "caldris", "orinath", "vexel", "aldric", "casciel", "delmar", "evander", "faelorn", "gareth", "ithral",
    "jarenth", "kelwyn", "lorenth", "maelis", "norvin", "pellarn", "quillon", "raveth", "stellan", "thandor",
]
TEMPLATES = [
    (["the", "capital", "of"], ["is"]),
    (["the", "leader", "of"], ["is"]),
    (["the", "currency", "of"], ["is"]),
    (["the", "main", "export", "of"], ["is"]),
    (["the", "official", "language", "of"], ["is"]),
    (["the", "founder", "of"], ["was"]),
]


def eps_label(sigma):
    eps = EPSILON_BY_SIGMA[sigma]
    return r"$\epsilon=\infty$" if math.isinf(eps) else rf"$\epsilon$={round(eps)}"


def lr_list_for(init_label, use_hf):
    return LR_LIST_PRETRAINED if use_hf and init_label.lower().startswith("pretrained") else LR_LIST_RANDOM


def configure_plot_style():
    matplotlib.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica", "sans-serif"],
        "font.weight": "normal", "font.size": 20, "axes.titleweight": "normal", "axes.labelweight": "normal",
        "axes.titlesize": 24, "axes.labelsize": 22, "xtick.labelsize": 18, "ytick.labelsize": 18,
        "legend.frameon": False, "legend.fontsize": 18, "figure.titlesize": 26,
        "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
    })


def save_figure(fig, stem):
    for ext, kwargs in [("png", {"dpi": 150}), ("pdf", {})]:
        fig.savefig(f"{stem}.{ext}", bbox_inches="tight", facecolor="white", **kwargs)
        print(f"Saved -> {stem}.{ext}")


def save_results(**payload):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"tf_train_opacus_{time.strftime('%Y%m%d_%H%M%S')}.pkl")
    for p in [path, os.path.join(RESULTS_DIR, "latest.pkl")]:
        with open(p, "wb") as f:
            pickle.dump(payload, f)
    return path


class WordTokenizer:
    def __init__(self):
        self.w2i = {"<pad>": 0, "<bos>": 1}
        self.i2w = {0: "<pad>", 1: "<bos>"}

    def add(self, word):
        if word not in self.w2i:
            self.i2w[len(self.w2i)] = word
            self.w2i[word] = len(self.w2i)

    def encode(self, words):
        return [self.w2i[w] for w in words]

    def decode_id(self, i):
        return self.i2w.get(i, "??")

    @property
    def size(self):
        return len(self.w2i)


def shuffled_entities(seed):
    rng = random.Random(seed)
    subjs, objs = list(SUBJECTS), list(OBJECTS)
    rng.shuffle(subjs)
    rng.shuffle(objs)
    return subjs, objs


def generate_facts_custom(facts_per_freq, frequencies, seed=0):
    tok = WordTokenizer()
    for w in [w for prefix, bridge in TEMPLATES for w in prefix + bridge] + SUBJECTS + OBJECTS:
        tok.add(w)
    subjs, objs = shuffled_entities(seed)
    facts_by_freq, idx = {}, 0
    for freq in frequencies:
        facts_by_freq[freq] = []
        for i in range(facts_per_freq):
            prefix, bridge = TEMPLATES[(idx + i) % len(TEMPLATES)]
            words = ["<bos>"] + prefix + [subjs[idx]] + bridge + [objs[idx]]
            ids = tok.encode(words)
            facts_by_freq[freq].append(dict(token_ids=ids, prompt_ids=ids[:-1], target_id=ids[-1], subject=subjs[idx],
                                            object=objs[idx], frequency=freq, text=" ".join(words[1:])))
            idx += 1
    return facts_by_freq, tok


def generate_facts_hf(hf_tok, facts_per_freq, frequencies, seed=0):
    subjs, objs = shuffled_entities(seed)
    facts_by_freq, idx = {}, 0
    for freq in frequencies:
        facts_by_freq[freq] = []
        for i in range(facts_per_freq):
            prefix, bridge = TEMPLATES[(idx + i) % len(TEMPLATES)]
            prompt_text = " ".join(prefix + [subjs[idx]] + bridge)
            prompt_ids, obj_ids = hf_tok.encode(prompt_text), hf_tok.encode(" " + objs[idx])
            facts_by_freq[freq].append(dict(token_ids=prompt_ids + obj_ids, prompt_ids=prompt_ids, target_id=obj_ids[0], subject=subjs[idx],
                                            object=objs[idx], frequency=freq, text=prompt_text + " " + objs[idx]))
            idx += 1
    return facts_by_freq


def build_sequences(facts_by_freq, rng):
    seqs = [f["token_ids"] for facts in facts_by_freq.values() for f in facts for _ in range(f["frequency"])]
    rng.shuffle(seqs)
    return seqs


class SeqDataset(Dataset):
    def __init__(self, sequences, pad_id):
        width = max(len(s) for s in sequences) - 1
        self.x = torch.tensor([s[:-1] + [pad_id] * (width - len(s) + 1) for s in sequences], dtype=torch.long)
        self.y = torch.tensor([s[1:] + [-100] * (width - len(s) + 1) for s in sequences], dtype=torch.long)

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=20):
        super().__init__()
        self.d_model = d_model
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=0.0, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        T = x.shape[1]
        h = self.tok_emb(x) * math.sqrt(self.d_model) + self.pos_emb(torch.arange(T, device=x.device).unsqueeze(0))
        mask = torch.triu(torch.full((T, T), float("-inf"), device=x.device), diagonal=1)
        return self.head(self.encoder(h, mask=mask))


if HFConv1D is not None:
    @register_grad_sampler(HFConv1D)
    def compute_hf_conv1d_grad_sample(layer, activations, backprops):
        activations = activations[0]
        ret = {}
        if layer.weight.requires_grad:
            ret[layer.weight] = torch.einsum("n...i,n...j->nij", activations, backprops)
        if layer.bias is not None and layer.bias.requires_grad:
            ret[layer.bias] = torch.einsum("n...k->nk", backprops)
        return ret


def make_model(use_hf, random_init, vocab_size, device, keep_tied_frozen=False):
    if not use_hf:
        model = MiniGPT(vocab_size, D_MODEL, N_HEADS, N_LAYERS, D_FF).to(device)
        if random_init:
            for p in model.parameters():
                nn.init.normal_(p, 0, 0.02)
        print(f"    MiniGPT ({'RANDOM' if random_init else 'default'} init)")
        model.tok_emb.weight.requires_grad_(False)
        print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
        return model

    from transformers import AutoConfig, AutoModelForCausalLM
    if random_init:
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(HF_MODEL))
        print(f"    Loaded {HF_MODEL} architecture (RANDOM init)")
    else:
        model = AutoModelForCausalLM.from_pretrained(HF_MODEL)
        print(f"    Loaded {HF_MODEL} (PRETRAINED weights)")
    model = model.to(device)
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte") and hasattr(model, "lm_head"):
        if not keep_tied_frozen:
            lm_head = nn.Linear(model.config.n_embd, model.config.vocab_size, bias=False)
            lm_head.weight.data.copy_(model.transformer.wte.weight.detach().clone())
            model.lm_head = lm_head.to(device)
            model.lm_head.weight.requires_grad_(False)
        model.transformer.wte.weight.requires_grad_(False)
        model.config.tie_word_embeddings = keep_tied_frozen
    if hasattr(model, "transformer") and hasattr(model.transformer, "wpe"):
        model.transformer.wpe.weight.requires_grad_(False)
    print(f"    Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model


def logits_of(model, x):
    out = model(x)
    return out.logits if hasattr(out, "logits") else out


def train_epoch(model, optimizer, loader, epoch):
    model.train()
    total_loss, n_batches = 0.0, 0
    iterator = tqdm(loader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
    for x, y in iterator:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = logits_of(model, x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
        total_loss += loss.detach()
        n_batches += 1
        iterator.set_postfix(loss=(total_loss / n_batches).item())
    print(f"\rEpoch {epoch}/{NUM_EPOCHS}  Loss: {(total_loss / n_batches):.4f}", end="")
    return (total_loss / max(n_batches, 1)).item()


@torch.no_grad()
def evaluate(model, facts_by_freq, tokenizer, device, use_hf=False):
    model.eval()
    results, predictions = {}, {}
    for freq, facts in facts_by_freq.items():
        results[freq], predictions[freq] = [], []
        for f in facts:
            dist = F.softmax(logits_of(model, torch.tensor([f["prompt_ids"]], device=device))[0, -1].float(), dim=0)
            prob, pred_id = dist[f["target_id"]].item(), dist.argmax().item()
            pred_word = tokenizer.decode(pred_id).strip() if use_hf else tokenizer.decode_id(pred_id)
            results[freq].append(prob)
            predictions[freq].append(dict(true=f["object"], predicted=pred_word, prob=prob, text=f["text"]))
    return results, predictions


def run_one(seed, sigma, use_hf, random_init, facts_by_freq, tokenizer, vocab_size, lr, keep_tied_frozen=False):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    device = torch.device(DEVICE)

    sequences = build_sequences(facts_by_freq, random.Random(seed))
    dataset = SeqDataset(sequences, pad_id=tokenizer.eos_token_id if use_hf else 0)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, drop_last=False)

    model = make_model(use_hf, random_init, vocab_size, device, keep_tied_frozen=keep_tied_frozen)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    privacy_engine = PrivacyEngine(accountant="rdp")
    model, optimizer, loader = privacy_engine.make_private(module=model, optimizer=optimizer, data_loader=loader, noise_multiplier=float(sigma),
                                                           max_grad_norm=CLIP_NORM, poisson_sampling=True, grad_sample_mode="hooks")

    eval_model = model._module if hasattr(model, "_module") else model
    curves = {fr: [] for fr in FREQUENCIES}
    for epoch in range(1, NUM_EPOCHS + 1):
        with BatchMemoryManager(data_loader=loader, max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, optimizer=optimizer) as memory_safe_loader:
            train_epoch(model, optimizer, memory_safe_loader, epoch)
        res, _ = evaluate(eval_model, facts_by_freq, tokenizer, device, use_hf)
        for fr in FREQUENCIES:
            curves[fr].append((epoch, np.mean(res[fr])))

    res, preds = evaluate(eval_model, facts_by_freq, tokenizer, device, use_hf)
    return res, preds, curves


def main():
    configure_plot_style()
    use_hf = HF_MODEL is not None
    if use_hf:
        print(f"Model     : {HF_MODEL} (HuggingFace)")
        print(f"Compare   : {'pretrained vs random' if COMPARE_INIT else 'pretrained only'}")
    else:
        print(f"Model     : MiniGPT  {N_LAYERS}L/{N_HEADS}H/d={D_MODEL}")
    print(f"Facts     : {FACTS_PER_FREQ}/freq, freqs = {FREQUENCIES}")
    print(f"DP-SGD    : clip={CLIP_NORM}, batch={BATCH_SIZE}, sigma in {NOISE_LEVELS}")
    print(f"Training  : {NUM_EPOCHS} epochs, lr={LR_HF if use_hf else LR}")
    print(f"Device    : {DEVICE}")
    print(f"Runs      : {NUM_RUNS}\n")

    if use_hf:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(HF_MODEL)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        preview = generate_facts_hf(tokenizer, FACTS_PER_FREQ, FREQUENCIES, BASE_SEED)
        vocab_size = tokenizer.vocab_size
    else:
        preview, tokenizer = generate_facts_custom(FACTS_PER_FREQ, FREQUENCIES, BASE_SEED)
        vocab_size = tokenizer.size

    print("Example facts:")
    for freq in FREQUENCIES[:3]:
        for f in preview[freq][:2]:
            print(f"  freq={freq:2d}  \"{f['text']}\"")
    print()

    if COMPARE_INIT:
        init_modes = [("pretrained" if use_hf else "default_init", False), ("random_init", True)]
    else:
        init_modes = [("default", False)]

    all_results, all_preds, all_curves = {}, {}, {}
    for init_label, random_init in init_modes:
        lr_list = lr_list_for(init_label, use_hf)
        all_results[init_label] = {s: {lr: defaultdict(list) for lr in lr_list} for s in NOISE_LEVELS}
        all_preds[init_label] = {s: {lr: [] for lr in lr_list} for s in NOISE_LEVELS}
        all_curves[init_label] = {s: {lr: [] for lr in lr_list} for s in NOISE_LEVELS}

        print(f"\n{'#' * 60}\n# Init mode: {init_label.upper()}\n{'#' * 60}")
        t_mode, count, total = time.time(), 0, len(NOISE_LEVELS) * NUM_RUNS * max(1, len(lr_list))
        for sigma in NOISE_LEVELS:
            print(f"\n  --- sigma = {sigma} ---")
            for lr in lr_list:
                print(f"    lr = {lr}")
                for run in range(NUM_RUNS):
                    count += 1
                    seed = BASE_SEED + run * 1000 + int(sigma * 100)
                    print(f"    run {run + 1}/{NUM_RUNS} ({count}/{total})", flush=True)
                    t1 = time.time()
                    facts_by_freq = generate_facts_hf(tokenizer, FACTS_PER_FREQ, FREQUENCIES, seed) if use_hf else generate_facts_custom(FACTS_PER_FREQ, FREQUENCIES, seed)[0]
                    res, preds, curves = run_one(seed, sigma, use_hf, random_init, facts_by_freq, tokenizer, vocab_size, lr, keep_tied_frozen=HF_KEEP_TIED_FROZEN)
                    for fr, ps in res.items():
                        all_results[init_label][sigma][lr][fr].append(np.mean(ps))
                    all_preds[init_label][sigma][lr].append(preds)
                    all_curves[init_label][sigma][lr].append(curves)
                    print(f"      ({time.time() - t1:.0f}s)  " + " | ".join(f"f={f}:{np.mean(p):.2f}" for f, p in sorted(res.items())))
        print(f"\n  Mode time: {time.time() - t_mode:.0f}s")

    freq_arr = np.array(sorted(FREQUENCIES))
    stats, thresholds = {}, {}
    for init_label in all_results:
        stats[init_label] = {s: {} for s in NOISE_LEVELS}
        thresholds[init_label] = {s: {} for s in NOISE_LEVELS}
        for sigma in NOISE_LEVELS:
            for lr in lr_list_for(init_label, use_hf):
                m = np.array([np.mean(all_results[init_label][sigma][lr][f]) for f in freq_arr])
                s = np.array([np.std(all_results[init_label][sigma][lr][f]) for f in freq_arr])
                stats[init_label][sigma][lr] = (m, s)
                thresholds[init_label][sigma][lr] = next((f for i, f in enumerate(freq_arr) if m[i] >= THRESHOLD), None)

    print(f"\n{'=' * 60}\nFrequency thresholds (P >= {THRESHOLD}):\n{'=' * 60}")
    for init_label in thresholds:
        print(f"\n  [{init_label}]")
        for sigma in NOISE_LEVELS:
            parts = [f"lr={lr}: " + (f"freq >= {t}" if t else f"> {max(FREQUENCIES)}") for lr, t in thresholds[init_label][sigma].items()]
            print(f"    sigma={sigma:<4}  ->  " + (" | ".join(parts) if parts else "no runs"))

    print(f"\n{'=' * 60}\nExample completions:\n{'=' * 60}")
    for init_label in all_preds:
        print(f"\n  [{init_label}]")
        for sigma in [0.0, NOISE_LEVELS[-1]]:
            for lr in lr_list_for(init_label, use_hf):
                runs_list = all_preds[init_label].get(sigma, {}).get(lr, [])
                if not runs_list:
                    print(f"    sigma={sigma} lr={lr}:  (no runs)")
                    continue
                print(f"    sigma={sigma} lr={lr}:")
                for freq in [min(FREQUENCIES), max(FREQUENCIES)]:
                    for p in runs_list[0].get(freq, [])[:2]:
                        print(f"      [{'Y' if p['true'] == p['predicted'] else 'N'}] f={freq} \"{p['text']}\"")
                        print(f"           -> \"{p['predicted']}\" (P={p['prob']:.3f})")

    results_path = save_results(stats=stats, thresholds=thresholds, all_curves=all_curves, all_preds=all_preds, freq_arr=freq_arr,
                                init_modes=init_modes, use_hf=use_hf, num_runs=NUM_RUNS, noise_levels=NOISE_LEVELS, epsilon_by_sigma=EPSILON_BY_SIGMA)
    print(f"\nSaved raw results -> {results_path}")
    make_plots(stats, thresholds, all_curves, freq_arr, init_modes, use_hf, NUM_RUNS)
    print("\nDone.")


def make_plots(stats, thresholds, all_curves, freq_arr, init_modes, use_hf, num_runs):
    colors = COLORBLIND_PALETTE[:len(NOISE_LEVELS)]
    n_modes = len(init_modes)
    default_lr = {init_label: lr_list_for(init_label, use_hf)[0] for init_label, _ in init_modes}
    title = lambda label: label.replace("_", " ").title()

    fig, axes = plt.subplots(1, n_modes, figsize=(10 * n_modes, 7.5), sharey=True, squeeze=False)
    for ax, (init_label, _) in zip(axes[0], init_modes):
        for i, sigma in enumerate(NOISE_LEVELS):
            m, s = stats[init_label][sigma][default_lr[init_label]]
            ax.plot(freq_arr, m, "o-", color=colors[i], label=eps_label(sigma), lw=2.5, ms=10, zorder=3)
            ax.fill_between(freq_arr, m - s, np.minimum(m + s, 1), alpha=0.12, color=colors[i])
        ax.axhline(THRESHOLD, color="0.35", ls="--", alpha=0.6, lw=2)
        ax.set_xscale("log")
        ax.set_xticks(freq_arr)
        ax.set_xticklabels(freq_arr)
        ax.set_xlabel("Fact Frequency", fontsize=22)
        ax.set_title(title(init_label), fontsize=24)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(True, alpha=0.22)
        ax.legend(loc="lower right", fontsize=26)
        ax.tick_params(axis="both", which="major", labelsize=24)
    axes[0][0].set_ylabel("P(correct completion)", fontsize=22)
    plt.tight_layout(pad=2)
    save_figure(fig, "dp_results_comparison")

    if n_modes == 2:
        fig2, ax = plt.subplots(figsize=(13, 7))
        x_pos, width, max_f = np.arange(len(NOISE_LEVELS)), 0.35, max(FREQUENCIES)
        for j, (init_label, _) in enumerate(init_modes):
            ts = [thresholds[init_label].get(s, {}).get(default_lr[init_label]) for s in NOISE_LEVELS]
            bars = ax.bar(x_pos - width / 2 + j * width, [t if t else max_f * 1.5 for t in ts], width, label=init_label,
                          color=colors[j % len(colors)], edgecolor="0.35", lw=1.0)
            for bar, t in zip(bars, ts):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5, str(t) if t else f">{max_f}", ha="center", fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([eps_label(s) for s in NOISE_LEVELS])
        ax.set_ylabel(r"Min frequency for P $\geq$ 0.5", fontsize=22)
        ax.set_title("Frequency Threshold: pretrained vs random init", fontsize=24)
        ax.legend(fontsize=18)
        ax.grid(True, axis="y", alpha=0.22)
        ax.tick_params(axis="both", which="major", labelsize=18)
        plt.tight_layout()
        save_figure(fig2, "dp_threshold_comparison")

    fig3, axes3 = plt.subplots(1, n_modes, figsize=(9 * n_modes, 6), squeeze=False)
    for ax, (init_label, _) in zip(axes3[0], init_modes):
        hmap = np.array([stats[init_label][s][default_lr[init_label]][0] for s in NOISE_LEVELS])
        im = ax.imshow(hmap, aspect="auto", cmap="cividis", vmin=0, vmax=1)
        ax.set_xticks(range(len(freq_arr)))
        ax.set_xticklabels(freq_arr)
        ax.set_yticks(range(len(NOISE_LEVELS)))
        ax.set_yticklabels([eps_label(s) for s in NOISE_LEVELS])
        ax.set_xlabel("Fact Frequency", fontsize=22)
        ax.set_ylabel(r"Privacy budget ($\epsilon$)", fontsize=22)
        ax.set_title(title(init_label), fontsize=24)
        ax.tick_params(axis="both", which="major", labelsize=18)
        for i in range(len(NOISE_LEVELS)):
            for j in range(len(freq_arr)):
                v = hmap[i, j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=16, color="white" if v < 0.3 or v > 0.7 else "black")
        plt.colorbar(im, ax=ax, shrink=0.8).ax.tick_params(labelsize=16)
    fig3.suptitle("P(correct) heatmap", fontsize=26)
    plt.tight_layout()
    save_figure(fig3, "dp_heatmap_comparison")

    fig4, axes4 = plt.subplots(n_modes, len(NOISE_LEVELS), figsize=(6 * len(NOISE_LEVELS), 5.5 * n_modes), sharey=True, squeeze=False)
    for row, (init_label, _) in enumerate(init_modes):
        runs_by_sigma = all_curves[init_label]
        for col, sigma in enumerate(NOISE_LEVELS):
            ax = axes4[row][col]
            runs = runs_by_sigma[sigma][default_lr[init_label]]
            for freq in FREQUENCIES:
                epochs = [e for e, _ in runs[0][freq]]
                avg = [np.mean([runs[r][freq][ei][1] for r in range(num_runs)]) for ei in range(len(epochs))]
                ax.plot(epochs, avg, "o-", label=f"f={freq}", ms=7, lw=2.2)
            ax.axhline(THRESHOLD, color="0.35", ls="--", alpha=0.5, lw=2)
            ax.set_ylim(-0.02, 1.05)
            ax.grid(True, alpha=0.22)
            ax.tick_params(axis="both", which="major", labelsize=18)
            if row == 0:
                ax.set_title(eps_label(sigma), fontsize=24)
            if col == 0:
                ax.set_ylabel(f"{title(init_label)}\nP(correct)", fontsize=22)
            if row == n_modes - 1:
                ax.set_xlabel("Epoch", fontsize=22)
    axes4[0][-1].legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18)
    fig4.suptitle("Learning curves", fontsize=26)
    plt.tight_layout()
    save_figure(fig4, "dp_learning_curves")


if __name__ == "__main__":
    main()
