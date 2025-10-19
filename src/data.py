from pathlib import Path
from typing import Any, Dict

from datasets import Dataset, load_dataset

from .config import TrainingConfig


def _resolve_data_files(dataset_path: str) -> Dict[str, Any]:
    path = Path(dataset_path)
    if path.is_file():
        return {"train": str(path)}
    if path.is_dir():
        files = list(path.glob("*.jsonl")) + list(path.glob("*.json")) + list(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No supported files found in {path}")
        return {"train": [str(f) for f in files]}
    return {}


def load_training_dataset(config: TrainingConfig) -> Dataset:
    data_files = _resolve_data_files(config.dataset_path)
    if data_files:
        train_files = data_files.get("train")
        if isinstance(train_files, list) and config.max_dataset_files:
            data_files["train"] = train_files[: config.max_dataset_files]
        if any(file.endswith(".parquet") for file in data_files.get("train", [])):
            dataset = load_dataset("parquet", data_files=data_files)
        else:
            dataset = load_dataset("json", data_files=data_files)
    else:
        dataset = load_dataset(config.dataset_path)
    if "train" not in dataset:
        raise ValueError("Dataset must contain a 'train' split.")
    train_dataset = dataset["train"]
    if config.max_train_samples:
        train_dataset = train_dataset.select(range(min(config.max_train_samples, len(train_dataset))))
    return train_dataset
