import modal


app = modal.App("fed-lora-qwen-download")

volume = modal.Volume.from_name("fed-lora-models", create_if_missing=False)

image = modal.Image.debian_slim().pip_install("huggingface_hub==0.25.2")

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
TARGET_DIR = "/vol/qwen2.5-7b-instruct"


@app.function(image=image, volumes={"/vol": volume}, timeout=60 * 60)
def download():
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=TARGET_DIR,
        local_dir_use_symlinks=False,
        resume_download=True,
    )
