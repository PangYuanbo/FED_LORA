# FED_LORA

## Modal Assets Archive
- Modal profile: `ybpang-1`
- Volume `fed-lora-models` stores `qwen2.5-7b-instruct` downloaded via `scripts/download_qwen.py`
- Volume `fed-lora-traindatasets` hosts FineWeb snapshots populated via `scripts/download_fineweb.py`

Run `modal run scripts/download_qwen.py::download` to refresh the snapshot if weights update. Ensure a valid Hugging Face token is configured through Modal secrets when required.

## Environment Setup
- Create an isolated environment: `python -m venv .venv && source .venv/bin/activate`
- Install dependencies: `pip install -r requirements.txt`
- Windows users must install a CUDA-enabled `torch` wheel manually before the requirements step.
- All source files must remain ASCII-only; never commit Chinese characters or other non-ASCII code points.

## Local LoRA Training (Windows RTX 5090)
- Update `configs/qwen_lora.yaml` with your dataset path (default `data/sample.jsonl`).
- Disable quantization on Windows by setting `use_4bit: false` and `use_8bit: false`.
- Launch training: `python scripts/train_lora.py --config configs/qwen_lora.yaml`
- Outputs (adapter weights, tokenizer) are written to `outputs/qwen-2.5-lora`.

## Modal H100 Workflow
- Ensure the `fed-lora-models` and `fed-lora-traindatasets` volumes contain weights and data.
- Submit a GPU job: `modal run scripts/modal_train_lora.py::train --config configs/qwen_lora.yaml --dataset-path /datasets/CC-MAIN-2025-05`
- The script mounts the repository at `/app`, reuses cached Qwen weights from `/models`, and saves adapters back to `outputs/` within the volume.
- `Trainer` 输出每步损失，同时自定义回调会在训练结束时打印平均 step 时长、step/s、平均 loss，便于快速评估吞吐。

## Data Utilities
- List available FineWeb snapshots: `modal run scripts/list_fineweb.py::list_files`
- Download snapshots into storage: `modal run scripts/download_fineweb.py::main CC-MAIN-2025-05 CC-MAIN-2025-08 CC-MAIN-2025-13`

## Known Issues & Avoidance
- Loading an entire FineWeb snapshot without limiting shards causes dataset materialization of ~14M rows and can exceed local CLI timeouts. Set `max_dataset_files`/`max_train_samples` in your config for smoke tests, or run `modal run --detach scripts/modal_train_lora.py::train_a100 --config-path ...` for long jobs.
- Always update this section when new pitfalls are discovered so the team does not repeat them.

## Repository Guide
- Contributor workflow and coding standards are documented in `AGENTS.md`.
- Track new volumes, datasets, or modal scripts in this README so the Modal setup stays reproducible.
