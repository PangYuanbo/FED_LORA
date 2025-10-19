import modal


DEFAULT_SNAPSHOTS = [
    "CC-MAIN-2025-05",
    "CC-MAIN-2025-08",
    "CC-MAIN-2025-13",
]

REPO_ID = "HuggingFaceFW/fineweb"
TARGET_ROOT = "/vol/fineweb"

app = modal.App("fed-lora-fineweb-download")

volume = modal.Volume.from_name("fed-lora-traindatasets", create_if_missing=False)

image = modal.Image.debian_slim().pip_install(
    "huggingface_hub==0.25.2",
    "tqdm==4.67.1",
)


@app.function(image=image, volumes={"/vol": volume}, timeout=4 * 60 * 60)
def download_snapshot(snapshot: str):
    from pathlib import Path

    from huggingface_hub import snapshot_download

    local_dir = Path(TARGET_ROOT) / snapshot
    local_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {snapshot} into {local_dir}")

    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        allow_patterns=[f"data/{snapshot}/*"],
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        resume_download=True,
    )

    print(f"Finished {snapshot}")


@app.local_entrypoint()
def main(*snapshots: str):
    targets = list(snapshots) if snapshots else DEFAULT_SNAPSHOTS
    for snapshot in targets:
        download_snapshot.remote(snapshot)
