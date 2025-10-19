import platform
from pathlib import Path
import time
from typing import Dict, List

import torch
from datasets import Dataset
from peft import LoraConfig as PeftLoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_callback import TrainerCallback

from .config import TrainingConfig
from .data import load_training_dataset


def _get_torch_dtype(name: str) -> torch.dtype:
    mapping: Dict[str, torch.dtype] = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    normalized = name.lower()
    if normalized not in mapping:
        raise ValueError(f"Unsupported target dtype: {name}")
    return mapping[normalized]


def _build_quantization_config(config: TrainingConfig) -> BitsAndBytesConfig | None:
    if not (config.use_4bit or config.use_8bit):
        return None
    if platform.system().lower().startswith("win"):
        raise ValueError("bitsandbytes quantization is not supported on Windows. Set use_4bit=False and use_8bit=False.")
    load_in_4bit = config.use_4bit
    load_in_8bit = config.use_8bit and not load_in_4bit
    return BitsAndBytesConfig(
        load_in_4bit=load_in_4bit,
        load_in_8bit=load_in_8bit,
        bnb_4bit_compute_dtype=torch.bfloat16 if config.bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )


def _prepare_model(config: TrainingConfig, tokenizer: AutoTokenizer) -> AutoModelForCausalLM:
    quantization_config = _build_quantization_config(config)
    torch_dtype = _get_torch_dtype(config.target_dtype)
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=config.trust_remote_code,
        quantization_config=quantization_config,
    )
    if quantization_config is not None:
        model = prepare_model_for_kbit_training(model)
    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    lora_config = PeftLoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=list(config.lora.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def _tokenize_dataset(dataset: Dataset, tokenizer: AutoTokenizer, config: TrainingConfig) -> Dataset:
    column = config.dataset_field
    if column not in dataset.column_names:
        raise KeyError(f"Dataset column '{column}' not found. Available columns: {dataset.column_names}")

    def tokenize_function(examples):
        texts = examples[column]
        return tokenizer(
            texts,
            max_length=config.max_seq_length,
            truncation=True,
            padding="longest",
        )

    tokenized = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
    )
    return tokenized


def _build_training_arguments(config: TrainingConfig) -> TrainingArguments:
    return TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        max_steps=config.max_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.learning_rate_scheduler_type,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        fp16=config.fp16,
        bf16=config.bf16,
        seed=config.seed,
        gradient_checkpointing=config.gradient_checkpointing,
        optim="paged_adamw_32bit" if config.use_4bit or config.use_8bit else "adamw_torch",
        report_to="none",
        torch_compile=config.compile_model,
        dataloader_num_workers=config.dataloader_num_workers,
        max_grad_norm=config.gradient_clipping,
        push_to_hub=config.push_to_hub,
        hub_model_id=config.hub_model_id,
        hub_private_repo=config.hub_private_repo,
    )


def train(config: TrainingConfig) -> None:
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        use_fast=True,
        trust_remote_code=config.trust_remote_code,
    )
    dataset = load_training_dataset(config)
    tokenized_dataset = _tokenize_dataset(dataset, tokenizer, config)
    model = _prepare_model(config, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = _build_training_arguments(config)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.add_callback(_ThroughputCallback(total_steps=training_args.max_steps))

    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)

    trainer.save_model(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)


class _ThroughputCallback(TrainerCallback):
    def __init__(self, total_steps: int | None = None) -> None:
        self.total_steps = total_steps
        self.step_start: float | None = None
        self.step_times: List[float] = []
        self.losses: List[float] = []

    def on_step_begin(self, args, state, control, **kwargs):
        self.step_start = time.perf_counter()

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_start is not None:
            elapsed = time.perf_counter() - self.step_start
            self.step_times.append(elapsed)
        logs = kwargs.get("logs") or {}
        loss = logs.get("loss")
        if loss is not None:
            self.losses.append(float(loss))

    def on_train_end(self, args, state, control, **kwargs):
        if not self.step_times:
            return
        avg_step_time = sum(self.step_times) / len(self.step_times)
        steps_per_sec = 1.0 / avg_step_time if avg_step_time > 0 else 0.0
        avg_loss = sum(self.losses) / len(self.losses) if self.losses else float("nan")
        final_loss = self.losses[-1] if self.losses else float("nan")
        print(
            f"[Throughput] steps={len(self.step_times)} "
            f"avg_step={avg_step_time:.3f}s "
            f"steps_per_sec={steps_per_sec:.3f} "
            f"avg_loss={avg_loss:.4f} "
            f"final_loss={final_loss:.4f}"
        )
