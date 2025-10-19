#!/usr/bin/env bash

cat <<'EOF'
Reminder:
- Document the root cause of any training timeout or stall in README.md under "Known Issues & Avoidance" before retrying.
- Limit FineWeb shard counts via the config fields max_dataset_files and max_train_samples for smoke tests.
- Use `modal run --detach scripts/modal_train_lora.py::train_a100 --config-path configs/qwen_lora_smoke.yaml` for long-running jobs to avoid local CLI timeouts.
EOF
