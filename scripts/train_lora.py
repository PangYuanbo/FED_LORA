import argparse

from src.config import load_config
from src.training import train


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tuning harness for Qwen models.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/qwen_lora.yaml",
        help="Path to a YAML configuration file.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    train(config)


if __name__ == "__main__":
    main()
