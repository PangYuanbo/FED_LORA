from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class LoraConfig:
    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")


@dataclass
class TrainingConfig:
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
    dataset_path: str = "data/sample.jsonl"
    output_dir: str = "outputs/qwen-lora"
    max_train_samples: int | None = None
    max_dataset_files: int | None = None
    max_train_samples: int | None = None
    max_steps: int = 0
    num_epochs: int = 3
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    logging_steps: int = 20
    save_steps: int = 100
    save_total_limit: int = 3
    max_seq_length: int = 2048
    gradient_checkpointing: bool = True
    use_8bit: bool = False
    use_4bit: bool = True
    bf16: bool = True
    fp16: bool = False
    seed: int = 42
    dataset_field: str = "text"
    learning_rate_scheduler_type: str = "cosine"
    resume_from_checkpoint: str | None = None
    trust_remote_code: bool = True
    push_to_hub: bool = False
    hub_model_id: str | None = None
    hub_private_repo: bool = False
    gradient_clipping: float = 0.3
    dataloader_num_workers: int = 2
    target_dtype: str = "bfloat16"
    compile_model: bool = False
    lora: LoraConfig = field(default_factory=LoraConfig)


def _merge_dict(defaults: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in defaults.items():
        result[key] = value
    for key, value in overrides.items():
        if (
            key in result
            and isinstance(result[key], dict)
            and isinstance(value, dict)
        ):
            result[key] = _merge_dict(result[key], value)
        else:
            result[key] = value
    return result


def load_config(path: str) -> TrainingConfig:
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data = yaml.safe_load(cfg_path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping.")
    defaults = TrainingConfig().__dict__
    overrides = _merge_dict(defaults, data)
    lora_data = overrides.get("lora", {})
    if isinstance(lora_data, dict):
        overrides["lora"] = LoraConfig(**lora_data)
    else:
        overrides["lora"] = TrainingConfig().lora
    overrides.pop("__dict__", None)
    return TrainingConfig(**overrides)
