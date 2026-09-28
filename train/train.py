import gc
import os
import re
from types import MethodType

import torch
import wandb
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager
from opacus.validators import ModuleValidator
from peft import LoraConfig, TaskType, get_peft_model
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser

from args import DataArguments, ModelArguments, PrivacyArguments, TrainArguments
from data import ALL_DATASETS, unwrap
from pld_accounting import compute_noise_multiplier_with_pld


def freeze_outside_layer_range(model, layer_start, layer_end):
    for name, param in model.named_parameters():
        match = re.search(r"\.(\d+)\.", name)
        if param.requires_grad and match:
            idx = int(match.group(1))
            if not (idx >= layer_start and (layer_end is None or idx <= layer_end)):
                param.requires_grad = False


def enable_dp_layer_metric_capture(dp_optimizer, model):
    param_name_by_id = {id(p): n for n, p in model.named_parameters()}
    original_add_noise = dp_optimizer.add_noise

    def add_noise_with_metrics(self):
        original_add_noise()
        self._dp_layer_metrics = {}
        for index, p in enumerate(self.params):
            if getattr(p, "summed_grad", None) is None or p.grad is None:
                continue
            signal, noise = p.summed_grad.detach().norm(2).item(), (p.grad.detach() - p.summed_grad.detach()).norm(2).item()
            self._dp_layer_metrics[param_name_by_id.get(id(p), f"param_{index}")] = {"signal": signal, "noise": noise, "snr": signal / (noise + 1e-12)}

    dp_optimizer.add_noise = MethodType(add_noise_with_metrics, dp_optimizer)
    dp_optimizer._dp_layer_metrics = {}


def default_collate(batch):
    return {k: torch.stack([torch.as_tensor(sample[k]) for sample in batch]) for k in batch[0]}


def get_dataloader(dataset, batch_size, world_size=1, rank=0, shuffle=True):
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle) if world_size > 1 else None
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=shuffle and sampler is None,
                      collate_fn=default_collate, drop_last=True, pin_memory=True, num_workers=4)


def save(model, path):
    unwrap(model, "save_pretrained").save_pretrained(path)
    print(f"Model saved to {path}")


def prepare_model(model, target_modules, train_args, model_args, device, local_rank, world_size, freeze):
    lora_config = LoraConfig(r=train_args.lora_r, lora_alpha=train_args.lora_alpha, target_modules=target_modules, lora_dropout=0, bias="none", task_type=TaskType.CAUSAL_LM)
    print(f"Using LoRA config: {lora_config}")
    model = get_peft_model(model, lora_config)
    if freeze:
        freeze_outside_layer_range(model, model_args.layer_start, model_args.layer_end)
    model.print_trainable_parameters()
    print("Unfrozen (trainable) parameters:\n" + "\n".join(f"  {n}  shape={list(p.shape)}" for n, p in model.named_parameters() if p.requires_grad))
    model = ModuleValidator.fix(model).to(device)
    return DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) if world_size > 1 else model


def wandb_config(train_args, model_args, opacus_config, data_args):
    return {f"{prefix}_{k}": v for prefix, a in [("train", train_args), ("model", model_args), ("opacus", opacus_config), ("data", data_args)] for k, v in vars(a).items()}


def validation_loss(model, val_loader, device):
    model.eval()
    with torch.no_grad():
        losses = [model(**{k: v.to(device) for k, v in batch.items()}).loss.item() for batch in val_loader]
    model.train()
    return sum(losses) / len(losses)


def setup(train_args, dataset, cuda, local_rank):
    device = torch.device(f"cuda:{local_rank}" if cuda else "cpu")
    if cuda:
        torch.cuda.set_device(local_rank)
    train_set = dataset.dataset["train"]
    train_args.data_size = len(train_set)
    train_set.set_format(type="torch")
    return device, local_rank == 0, train_set


def train(model, path_to_save_model, dataset, train_args, opacus_config, model_args, data_args, cuda=True, local_rank=0, world_size=1):
    device, is_main, train_set = setup(train_args, dataset, cuda, local_rank)
    model = prepare_model(model, ["q_proj", "v_proj"], train_args, model_args, device, local_rank, world_size,
                          model_args.layer_start is not None or model_args.layer_end is not None)
    splits = dataset.dataset
    val_set = splits["validation"] if "validation" in splits else splits["test"] if "test" in splits else train_set.select(range(50))
    train_loader = get_dataloader(train_set, train_args.batch_size, world_size, local_rank)
    val_loader = get_dataloader(val_set, 8, world_size, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)

    privacy_tag = f"target_epsilon_{opacus_config.target_epsilon}" if opacus_config.target_epsilon is not None else f"noise_multiplier_{opacus_config.noise_multiplier}"
    run_name = (f"{model_args.model_name}_dataset_{data_args.dataset_name}_seq_len_{model_args.sequence_len}_lr_{train_args.lr}_batch_size_{train_args.batch_size}"
                f"_gas_{train_args.gradient_accumulation_steps}_{privacy_tag}_lora_{train_args.lora_r}_alpha_{train_args.lora_alpha}_max_grad_norm_{opacus_config.max_grad_norm}")
    wandb.init(project=train_args.project_name, name=run_name, config=wandb_config(train_args, model_args, opacus_config, data_args))

    model.train()
    opacus_config.target_delta = 1 / (len(train_set) ** 1.1)
    privacy_engine = PrivacyEngine()
    private = dict(module=model, optimizer=optimizer, data_loader=train_loader, max_grad_norm=opacus_config.max_grad_norm, clipping=opacus_config.clipping)
    if opacus_config.pld and opacus_config.target_epsilon is not None:
        opacus_config.noise_multiplier = compute_noise_multiplier_with_pld(num_examples=train_args.data_size, batch_size=train_args.batch_size, epochs=train_args.epochs,
                                                                           delta=opacus_config.target_delta, target_epsilon=opacus_config.target_epsilon)
        print(f"Using PLD accountant with noise_multiplier={opacus_config.noise_multiplier} for target_epsilon={opacus_config.target_epsilon} and target_delta={opacus_config.target_delta}")
    if opacus_config.target_epsilon is not None and not opacus_config.pld:
        model, optimizer, train_loader = privacy_engine.make_private_with_epsilon(epochs=train_args.epochs, target_epsilon=opacus_config.target_epsilon,
                                                                                  target_delta=opacus_config.target_delta, **private)
        print(f"OPACUS: using target_epsilon={opacus_config.target_epsilon} (noise_multiplier found automatically).")
    else:
        model, optimizer, train_loader = privacy_engine.make_private(
            noise_multiplier=opacus_config.noise_multiplier, target_delta=opacus_config.target_delta, clipbound_learning_rate=opacus_config.clipbound_learning_rate,
            target_unclipped_quantile=opacus_config.target_unclipped_quantile, min_clipbound=opacus_config.min_clipbound,
            max_clipbound=opacus_config.max_clipbound, unclipped_num_std=opacus_config.unclipped_num_std, **private)
    optimizer.attach_step_hook(privacy_engine.accountant.get_optimizer_hook_fn(sample_rate=1 / len(train_loader) * train_args.gradient_accumulation_steps))

    if is_main:
        enable_dp_layer_metric_capture(optimizer, model)
    print(train_args)
    print(f"Learning rate: {train_args.lr}, Noise multiplier: {opacus_config.noise_multiplier}, Clipping norm: {opacus_config.max_grad_norm}")
    print("Model validated." if not ModuleValidator.validate(model, strict=True) else "Model validation failed. Please check your model configuration.")
    print(f"Privacy parameters: noise_multiplier={opacus_config.noise_multiplier}, target_delta={opacus_config.target_delta}, clipping={opacus_config.clipping}")

    global_step = 0
    model.train()
    for epoch in range(train_args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        losses = []
        optimizer.zero_grad()
        with BatchMemoryManager(data_loader=train_loader, max_physical_batch_size=2, optimizer=optimizer) as new_data_loader:
            for step, batch in enumerate(tqdm(new_data_loader, desc=f"Epoch {epoch + 1}/{train_args.epochs}", disable=not is_main)):
                batch = {k: v.to(device) for k, v in batch.items()}
                if len(batch["input_ids"]) == 0:
                    print("Skipping empty batch at step", step)
                    continue
                loss = model(**batch).loss
                loss.backward()
                optimizer.step()
                global_step += 1
                layer_metrics = getattr(optimizer, "_dp_layer_metrics", None)
                if is_main and layer_metrics and global_step % 10 == 0:
                    wandb.log({"global_step": global_step, **{f"dp_layers/{name.replace('.', '/')}/{k}": v for name, stats in layer_metrics.items() for k, v in stats.items()}})
                optimizer.zero_grad()
                losses.append(loss.item())
                torch.cuda.empty_cache()
                gc.collect()

        train_loss = sum(losses) / len(losses)
        if is_main:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss, "max_grad_norm": optimizer.max_grad_norm, "noise_multiplier": optimizer.noise_multiplier})
            print(f"Epoch {epoch + 1} Train loss: {train_loss:.4f}")
        if val_loader and is_main:
            val_loss = validation_loss(model, val_loader, device)
            print(f"Epoch {epoch + 1} Val loss: {val_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
        torch.cuda.empty_cache()
        if (epoch + 1) % 5 == 0 and is_main:
            save(model, f"{path_to_save_model}/ckpt_epoch_{epoch + 1}")

    return (model, privacy_engine) if is_main else (None, None)


def train_no_dp(model, path_to_save_model, dataset, train_args, opacus_config, model_args, data_args, cuda=True, local_rank=0, world_size=1):
    device, is_main, train_set = setup(train_args, dataset, cuda, local_rank)
    model = prepare_model(model, ["query_key_value"], train_args, model_args, device, local_rank, world_size,
                          model_args.layer_start is not None and model_args.layer_end is not None)
    train_loader = get_dataloader(train_set, train_args.batch_size, world_size, local_rank)
    val_loader = get_dataloader(train_set.select(range(50)), 1, world_size, local_rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_args.lr)
    run_name = (f"{model_args.model_name}_dataset_{data_args.dataset_name}_seq_len_{model_args.sequence_len}_lr_{train_args.lr}_batch_size_{train_args.batch_size}"
                f"_gas_{train_args.gradient_accumulation_steps}_noise_multiplier_inf_lora_{train_args.lora_r}_alpha_{train_args.lora_alpha}")
    wandb.init(project=train_args.project_name, name=run_name, config=wandb_config(train_args, model_args, opacus_config, data_args))

    model.train()
    if is_main:
        print("Starting training ...")
    for epoch in range(train_args.epochs):
        if world_size > 1:
            train_loader.sampler.set_epoch(epoch)
        losses = []
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{train_args.epochs}", disable=not is_main)):
            batch = {k: v.to(device) for k, v in batch.items()}
            if len(batch["input_ids"]) == 0:
                continue
            loss = model(**batch).loss / train_args.gradient_accumulation_steps
            loss.backward()
            if (step + 1) % train_args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                torch.cuda.empty_cache()
                gc.collect()
            losses.append(loss.item())

        train_loss = sum(losses) / len(losses)
        if is_main:
            wandb.log({"epoch": epoch + 1, "train_loss": train_loss})
            print(f"Epoch {epoch + 1} Train loss: {train_loss:.4f}")
        if val_loader and is_main:
            val_loss = validation_loss(model, val_loader, device)
            print(f"Epoch {epoch + 1} Val loss: {val_loss:.4f}")
            wandb.log({"epoch": epoch + 1, "val_loss": val_loss})
        if is_main:
            save(model, f"{path_to_save_model}/ckpt_epoch_{epoch + 1}")
    return (model, None) if is_main else (None, None)


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainArguments, PrivacyArguments))
    model_args, data_args, training_args, privacy_args, _ = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    local_rank, world_size = int(os.environ.get("LOCAL_RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    print(f"Local rank: {local_rank}, World size: {world_size}, CUDA available: {torch.cuda.is_available()}")
    if world_size > 1:
        torch.distributed.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name, torch_dtype=torch.float16)
    tokenizer.pad_token = tokenizer.eos_token

    data_args.sequence_len = model_args.sequence_len
    dataset = ALL_DATASETS[data_args.dataset_name](data_args, tokenizer)
    layer_end_str = str(model_args.layer_end) if model_args.layer_end is not None else "end"
    model_args.path_to_save_model = f"{model_args.path_to_save_model}_layers_{model_args.layer_start}_to_{layer_end_str}"

    training_args.batch_size = training_args.per_device_train_batch_size
    dataset.dataset = dataset.dataset.map(dataset.preprocess_function, batched=True, num_proc=8, desc="tokenizing dataset",
                                          remove_columns=dataset.dataset.column_names["train"], load_from_cache_file=False)
    print(dataset.dataset)

    args = (model_args.path_to_save_model, dataset, training_args, privacy_args, model_args, data_args)
    if (privacy_args.target_epsilon is None or privacy_args.target_epsilon == 0) and privacy_args.noise_multiplier is None:
        print("Training without Differential Privacy")
        model, privacy_engine = train_no_dp(model, *args, cuda=True, local_rank=local_rank, world_size=world_size)
    else:
        print("Training with Differential Privacy")
        model, privacy_engine = train(model, *args, cuda=True, local_rank=local_rank, world_size=world_size)

    if local_rank == 0 and privacy_engine is not None and model is not None:
        privacy_engine.save_checkpoint(path=model_args.path_to_save_model + "_pvt", module=model, optimizer=None)
        save(model, model_args.path_to_save_model)


if __name__ == "__main__":
    main()
