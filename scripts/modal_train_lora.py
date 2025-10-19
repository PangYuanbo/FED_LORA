import modal

app = modal.App("fed-lora-lora-train")

image = (
    modal.Image.debian_slim()
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir(".", remote_path="/app")
)

model_volume = modal.Volume.from_name("fed-lora-models", create_if_missing=False)
dataset_volume = modal.Volume.from_name("fed-lora-traindatasets", create_if_missing=True)


def _run_training(config_path: str, dataset_path: str | None):
    import os
    import sys
    from pathlib import Path

    repo_root = Path("/app")
    if repo_root.exists():
        if str(repo_root) not in sys.path:
            sys.path.insert(0, str(repo_root))
        os.chdir(repo_root)
    else:
        fallback_root = Path(__file__).resolve().parent
        if str(fallback_root) not in sys.path:
            sys.path.insert(0, str(fallback_root))
        os.chdir(fallback_root)

    from src.config import load_config
    from src.training import train as run_training

    config = load_config(config_path)
    model_local_path = Path("/models/qwen2.5-7b-instruct")
    if model_local_path.exists():
        config.model_name = str(model_local_path)
    if dataset_path is not None:
        config.dataset_path = dataset_path
    run_training(config)


@app.function(
    image=image,
    gpu="H100",
    volumes={
        "/models": model_volume,
        "/datasets": dataset_volume,
    },
    timeout=24 * 60 * 60,
)
def train_h100(config_path: str = "configs/qwen_lora.yaml", dataset_path: str | None = None):
    _run_training(config_path, dataset_path)


@app.function(
    image=image,
    gpu="A100-40GB",
    volumes={
        "/models": model_volume,
        "/datasets": dataset_volume,
    },
    timeout=24 * 60 * 60,
)
def train_a100(config_path: str = "configs/qwen_lora.yaml", dataset_path: str | None = None):
    _run_training(config_path, dataset_path)


@app.local_entrypoint()
def main(
    config_path: str = "configs/qwen_lora.yaml",
    dataset_path: str | None = None,
    gpu_type: str = "H100",
):
    if gpu_type.upper() == "A100":
        train_a100.remote(config_path=config_path, dataset_path=dataset_path)
    else:
        train_h100.remote(config_path=config_path, dataset_path=dataset_path)
