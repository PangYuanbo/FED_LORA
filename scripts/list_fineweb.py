import modal


app = modal.App("fed-lora-fineweb-utils")

image = modal.Image.debian_slim().pip_install("huggingface_hub==0.25.2")


@app.function(image=image, timeout=60)
def list_files():
    from huggingface_hub import HfApi

    api = HfApi()
    files = api.list_repo_files(
        repo_id="HuggingFaceFW/fineweb",
        repo_type="dataset",
    )
    prefixes = set()
    for path in files:
        if path.startswith("data/CC-MAIN-2025") and path.count("/") >= 2:
            prefix = path.split("/", 2)[1]
            prefixes.add(prefix)
    for name in sorted(prefixes):
        print(name)
