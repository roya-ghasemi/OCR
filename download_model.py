"""
Reliable, resumable downloader for Qwen2-VL-7B-Instruct.
- Parallel shard download (max_workers) to beat per-connection throttling.
- Resumes from existing .incomplete blobs; safe to re-run.
"""
import os
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")   # plain resumable downloader
os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "60")

from huggingface_hub import snapshot_download

MODEL = os.getenv("MODEL_NAME", "Qwen/Qwen2-VL-7B-Instruct")

path = snapshot_download(
    repo_id=MODEL,
    max_workers=8,                       # parallel files -> beats per-connection caps
    allow_patterns=[
        "*.safetensors", "*.json", "*.txt",
        "tokenizer*", "vocab*", "merges*", "*.model",
    ],
    resume_download=True,
)
print("DOWNLOAD_COMPLETE:", path)
