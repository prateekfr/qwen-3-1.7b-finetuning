"""
Merge LoRA Adapter with Base Qwen3-1.7B Model.
Produces a standalone, unquantized (float16) HuggingFace model ready for:
1. GGUF / llama.cpp / Ollama conversion
2. Full standalone deployment without PEFT dependencies
"""

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_lora_to_standalone(
    base_model_path: str,
    adapter_path: str,
    output_path: str,
    device: str = "cpu",
):
    print("=" * 70)
    print("        MERGING LORA ADAPTER INTO BASE QWEN3-1.7B WEIGHTS")
    print("=" * 70)
    print(f"Base Model    : {base_model_path}")
    print(f"Adapter Path  : {adapter_path}")
    print(f"Output Path   : {output_path}")
    print(f"Compute Device: {device.upper()}")
    print("=" * 70)

    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    adapter_dir = Path(adapter_path)
    if (adapter_dir / "trainable").exists():
        adapter_dir = adapter_dir / "trainable"

    print("\n[1/4] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(base_model_path), trust_remote_code=True)

    print("[2/4] Loading unquantized float16 base model (this takes ~3.4 GB RAM)...")
    t0 = time.time()
    base_model = AutoModelForCausalLM.from_pretrained(
        str(base_model_path),
        torch_dtype=torch.float16,
        device_map=device,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    print(f"Base model loaded in {time.time() - t0:.2f}s.")

    print(f"[3/4] Attaching LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(
        base_model,
        str(adapter_dir),
        torch_dtype=torch.float16,
        device_map=device,
    )

    print("[4/4] Merging weights via merge_and_unload()...")
    t1 = time.time()
    merged_model = model.merge_and_unload()
    print(f"Merged successfully in {time.time() - t1:.2f}s!")

    print(f"\nSaving merged standalone model to {output_dir}...")
    merged_model.save_pretrained(str(output_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(output_dir))

    print("=" * 70)
    print(f"SUCCESS: Merged standalone model saved to: {output_dir}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--base", type=str, default="weights/qwen3-1.7b", help="Path to base model weights")
    parser.add_argument("--adapter", type=str, default="checkpoints/rl_best", help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, default="weights/qwen3-1.7b-no-e-merged", help="Output path for merged model")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to perform merge on")
    args = parser.parse_args()

    base_path = ROOT / args.base
    adapter_path = ROOT / args.adapter
    output_path = ROOT / args.output

    merge_lora_to_standalone(
        base_model_path=str(base_path),
        adapter_path=str(adapter_path),
        output_path=str(output_path),
        device=args.device,
    )


if __name__ == "__main__":
    main()
