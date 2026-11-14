"""
Test Tokenizer & Forbidden Token Mask
"""
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoTokenizer
from core.sampling import build_forbidden_token_mask
from core.checks import e_pass, text_checks


def main():
    print("=" * 60)
    print("  Testing Tokenizer & Forbidden Token Mask")
    print("=" * 60)

    print("\n1. Loading Qwen3 tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B", trust_remote_code=True)
    vocab_size = len(tokenizer)
    print(f"   Vocab size: {vocab_size}")

    print("\n2. Building forbidden token mask (forbidden_chars='eE', ascii_only=True)...")
    mask = build_forbidden_token_mask(tokenizer, vocab_size, forbidden_chars="eE", ascii_only=True)
    blocked = mask.sum().item()
    allowed = vocab_size - blocked
    print(f"   Blocked tokens: {blocked:,} ({blocked / vocab_size * 100:.2f}%)")
    print(f"   Allowed tokens: {allowed:,} ({allowed / vocab_size * 100:.2f}%)")

    print("\n3. Testing character checking on sample sentences:")
    test_cases = [
        "Hello world",
        "A quick brown fox jumps on a lazy dog",
        "Good night, my pal.",
        "Stay strong and carry on.",
        "A dog sat on a mat.",
        "Café",
    ]
    for s in test_cases:
        checks = text_checks(s)
        pass_val = e_pass(s)
        print(f"   '{s}': e_pass={pass_val} (e_count={checks['e_count']}, norm_no_e={checks['normalized_no_e']})")

    print("\n4. Checking sample allowed tokens:")
    allowed_indices = torch.where(~mask)[0][:15].tolist()
    allowed_tokens = [tokenizer.decode([idx]) for idx in allowed_indices]
    print(f"   First 15 allowed tokens: {allowed_tokens}")

    print("\n" + "=" * 60)
    print("  ✅ Token mask verification completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
