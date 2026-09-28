from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class ModelArguments:
    model_name: str = "gpt2"
    path_to_save_model: str = "my_privacy_gpt2_model"
    sequence_len: int = 512
    layer_start: Optional[int] = None
    layer_end: Optional[int] = None


@dataclass
class DataArguments:
    dataset_name: str = "med-summ"
    ds_config: Optional[Dict[str, Any]] = field(default_factory=dict)
    path_to_dataset: Optional[str] = None
    path_to_test_dataset: Optional[str] = None
    control_field: str = "label"
    text_field: str = "text"
    label_field: str = "text"
    prompt_begin: str = ""
    prompt_end: str = ""

    def __post_init__(self):
        if self.path_to_test_dataset == "None":
            self.path_to_test_dataset = None


@dataclass
class TrainArguments:
    epochs: int = 1
    lr: float = 5e-5
    per_device_train_batch_size: int = 4
    gradient_accumulation_steps: float = 128
    lora_r: int = 8
    lora_alpha: int = 32
    project_name: str = "dp-fact"

    def __post_init__(self):
        for name in ["epochs", "lr", "gradient_accumulation_steps"]:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be greater than 0.")


@dataclass
class PrivacyArguments:
    target_epsilon: float = None
    max_grad_norm: float = 0.5
    target_delta: float = 1e-5
    clipping: str = "per_layer"
    noise_multiplier: float = None
    clipbound_learning_rate: float = 0.2
    target_unclipped_quantile: float = 0.5
    min_clipbound: float = 0.1
    max_clipbound: float = 1.0
    unclipped_num_std: float = 0.1
    pld: bool = False

