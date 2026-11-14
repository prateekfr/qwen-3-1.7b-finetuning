"""
Environment Verification Script — Qwen, Minus the E

Run this after setting up the conda environment to verify everything works:
    conda activate qwen-no-e
    python scripts/verify_env.py
"""

import sys
import importlib


def check(name: str, test_fn, required: bool = True):
    """Run a check and print pass/fail."""
    tag = "REQUIRED" if required else "OPTIONAL"
    try:
        result = test_fn()
        print(f"  ✅ {name}: {result}")
        return True
    except Exception as e:
        symbol = "❌" if required else "⚠️"
        print(f"  {symbol} {name}: {e}")
        return not required


def main():
    all_ok = True
    print("\n" + "=" * 60)
    print("  Qwen, Minus the E — Environment Verification")
    print("=" * 60)

    # --- Python version ---
    print("\n📦 Python")
    v = sys.version_info
    all_ok &= check("Python version", lambda: f"{v.major}.{v.minor}.{v.micro}")

    # --- Core packages ---
    print("\n📦 Core Packages")
    for pkg in ["torch", "transformers", "accelerate", "peft", "bitsandbytes", "datasets"]:
        all_ok &= check(pkg, lambda p=pkg: importlib.import_module(p).__version__)

    # --- Optional packages ---
    print("\n📦 Optional Packages")
    for pkg in ["yaml", "tqdm"]:
        check(pkg, lambda p=pkg: importlib.import_module(p).__version__ if hasattr(importlib.import_module(p), '__version__') else "installed", required=False)

    # --- CUDA ---
    print("\n🖥️  GPU & CUDA")
    try:
        import torch
        all_ok &= check("CUDA available", lambda: torch.cuda.is_available())
        if torch.cuda.is_available():
            all_ok &= check("GPU name", lambda: torch.cuda.get_device_name(0))
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            all_ok &= check("VRAM", lambda: f"{vram_gb:.2f} GB")
            all_ok &= check("CUDA version", lambda: torch.version.cuda)
            all_ok &= check("bf16 support", lambda: torch.cuda.is_bf16_supported())
        else:
            print("  ❌ No CUDA GPU detected! Training will not work.")
            all_ok = False
    except ImportError:
        print("  ❌ PyTorch not installed!")
        all_ok = False

    # --- Quick 4-bit load test ---
    print("\n🧪 4-bit Quantization Test")
    try:
        import torch
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        check("BitsAndBytesConfig", lambda: "configured successfully")
        print("  ℹ️  Skipping model load test (run separately to avoid download).")
    except Exception as e:
        print(f"  ❌ 4-bit config failed: {e}")
        all_ok = False

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_ok:
        print("  ✅ ALL CHECKS PASSED — Environment is ready!")
    else:
        print("  ❌ SOME CHECKS FAILED — Fix issues above before proceeding.")
    print("=" * 60 + "\n")

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
