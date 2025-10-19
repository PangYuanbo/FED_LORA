import modal

app = modal.App("fed-lora-fetch-livebench")

image = modal.Image.debian_slim().pip_install("httpx==0.28.1")

@app.function(image=image)
def fetch():
    import httpx

    url = "https://raw.githubusercontent.com/livebench/livebench/main/README.md"
    resp = httpx.get(url, timeout=10)
    resp.raise_for_status()
    text = resp.text.splitlines()
    print("First 20 lines:")
    for line in text[:20]:
        print(line)
    print("\nLast 5 lines:")
    for line in text[-5:]:
        print(line)


@app.local_entrypoint()
def main():
    fetch.remote()
