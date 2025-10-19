import modal

app = modal.App("fed-lora-fetch-livebench-changelog")

image = modal.Image.debian_slim().pip_install("httpx==0.28.1")

@app.function(image=image)
def fetch():
    import httpx

    url = "https://raw.githubusercontent.com/livebench/livebench/main/changelog.md"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    lines = resp.text.strip().splitlines()
    print("Top entries:")
    for line in lines[:40]:
        print(line)


@app.local_entrypoint()
def main():
    fetch.remote()
