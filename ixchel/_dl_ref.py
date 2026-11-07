import os, sys
from huggingface_hub import snapshot_download
root = __import__("os").path.join(__import__("os").path.normpath(__import__("os").path.join(__import__("os").path.dirname(__import__("os").path.abspath(__file__)), "..")), 'data', 'ref_models')
for repo in ("Qwen/Qwen2.5-1.5B", "TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"):
    try:
        p = snapshot_download(repo, local_dir=os.path.join(root, repo.split("/")[1]),
                              allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "tokenizer*", "merges*", "vocab*"])
        print("OK", repo, p, flush=True)
    except Exception as e:
        print("FAIL", repo, repr(e)[:300], flush=True)
