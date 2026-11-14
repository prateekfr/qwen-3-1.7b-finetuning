"""
Download Qwen3-1.7B weights into weights/qwen3-1.7b
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from huggingface_hub import snapshot_download


def main():
    target_dir = ROOT / "weights" / "qwen3-1.7b"
    target_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 60)
    print(f"Downloading Qwen/Qwen3-1.7B to {target_dir}...")
    print("=" * 60)

    snapshot_download(
        repo_id="Qwen/Qwen3-1.7B",
        local_dir=str(target_dir),
        local_dir_use_symlinks=False,  # Important for Windows
    )

    print("\n✅ Download completed successfully!")
    safetensors = list(target_dir.glob("*.safetensors"))
    print(f"Downloaded {len(safetensors)} safetensors shard(s):")
    for s in safetensors:
        size_mb = s.stat().st_size / (1024 * 1024)
        print(f"  - {s.name}: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
