"""Standalone resumable downloader for Qwen2-VL-7B-Instruct into a private cache."""
import os, sys, time

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"  # avoid the stalling we just saw
CACHE = r"D:\ocr\hf_cache"
os.environ["HF_HOME"] = CACHE
os.makedirs(CACHE, exist_ok=True)

from huggingface_hub import snapshot_download

REPO = "Qwen/Qwen2-VL-7B-Instruct"
print(f"[dl] starting snapshot_download({REPO}) -> {CACHE}", flush=True)
t0 = time.time()
path = snapshot_download(
    repo_id=REPO,
    cache_dir=os.path.join(CACHE, "hub"),
    allow_patterns=["*.json", "*.txt", "*.safetensors", "merges.txt", "vocab.json", "*.py"],
    max_workers=4,
    resume_download=True,
)
print(f"[dl] DONE in {time.time()-t0:.0f}s -> {path}", flush=True)
# touch a sentinel so the poller knows we finished cleanly
open(os.path.join(CACHE, "_DOWNLOAD_COMPLETE"), "w").close()
print("[dl] sentinel written", flush=True)
