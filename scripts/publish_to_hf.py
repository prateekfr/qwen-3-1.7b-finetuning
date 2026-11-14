"""
Publish Fine-Tuned Qwen3-1.7B Lipogram Models to the Hugging Face Hub.
Supports uploading:
1. LoRA adapter (checkpoints/rl_best)
2. Standalone merged model (weights/qwen3-1.7b-no-e-merged)
3. GGUF model binary (weights/qwen3-1.7b-no-e-f16.gguf)
"""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.rewards import load_dotenv_file

ADAPTER_CARD_TEMPLATE = """---
language:
- en
license: apache-2.0
tags:
- lipogram
- constrained-generation
- grpo
- reinforcement-learning
- post-training
- peft
- lora
- qwen
- qwen3
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-generation
---

# Qwen3-1.7B Lipogram Model — LoRA Adapter (Minus the 'E')

This repository contains the fine-tuned PEFT LoRA adapter for **Qwen3-1.7B**, trained to adhere to a strict character-level constraint: generating fluent, grammatical English **without ever using the letter "e" or "E"** (a lipogram).

Trained and aligned using a 3-layer architecture:
1. **Logit Masking:** Pre-computed boolean mask blocking 103,639 tokens (68.33% of the Qwen3 vocabulary).
2. **Supervised Fine-Tuning (SFT):** 4-bit QLoRA trained on 180 curated, human-verified e-free instruction pairs.
3. **Reinforcement Learning (GRPO):** Group Relative Policy Optimization optimizing response fluency, length consistency, and lexical diversity.

## Benchmark Results (Gadsby Benchmark Suite)

| Model Architecture Stage | Strict Lipogram Pass | Usable Response Rate | N-Gram Diversity |
| :--- | :---: | :---: | :---: |
| Base Qwen-1.7B + Mask | 88.0% | 28.0%* | 91.5% |
| SFT (180 Rewrites) + Mask | 96.0% | 80.0% | 91.5% |
| **RL (GRPO Best) + Mask** | **100.0%** | **88.0%** | **92.5%** |

*\*Note: Base model usable rate reflects pure English ASCII responses; unaligned base models attempt foreign-language token substitutions to evade the constraint.*

## Usage with PEFT

### 1. Requirements
```bash
pip install torch transformers peft
```

### 2. Loading the Adapter with Logit Mask
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 1. Load base model and tokenizer
model_id = "Qwen/Qwen3-1.7B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

base_model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

# 2. Attach fine-tuned lipogram adapter
model = PeftModel.from_pretrained(base_model, "{repo_id}")
model.eval()

# 3. Build forbidden token mask (blocking 'e' and 'E')
vocab_size = model.config.vocab_size
forbidden_mask = torch.zeros(vocab_size, dtype=torch.bool)
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if "e" in token_str.lower():
        forbidden_mask[token_id] = True

# 4. Generate with mask applied to logits
prompt = "Write a short poem about a frog."
inputs = tokenizer(f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n", return_tensors="pt").to("cuda")

with torch.no_grad():
    curr_ids = inputs["input_ids"]
    for _ in range(60):
        outputs = model(input_ids=curr_ids)
        scores = outputs.logits[:, -1, :].clone().float()
        scores[:, forbidden_mask.to("cuda")] = -float("inf")
        next_token = torch.multinomial((scores / 0.7).softmax(-1), num_samples=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        curr_ids = torch.cat([curr_ids, next_token], dim=-1)

print(tokenizer.decode(curr_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

## Citation & Attribution
- **Author & Post-Training Engineer:** Prateek
- **Base Architecture:** Qwen3-1.7B by the Qwen Team / Alibaba Cloud
- **License:** Apache 2.0
"""

STANDALONE_CARD_TEMPLATE = """---
language:
- en
license: apache-2.0
tags:
- lipogram
- constrained-generation
- grpo
- reinforcement-learning
- post-training
- qwen
- qwen3
base_model: Qwen/Qwen3-1.7B
pipeline_tag: text-generation
---

# Qwen3-1.7B Lipogram Model — Standalone Merged FP16 (Minus the 'E')

This repository contains the standalone, fully merged **Qwen3-1.7B** model weights (FP16), post-trained to adhere to a strict character-level constraint: generating fluent, grammatical English **without ever using the letter "e" or "E"** (a lipogram).

Trained and aligned using a 3-layer architecture:
1. **Logit Masking:** Pre-computed boolean mask blocking 103,639 tokens (68.33% of the Qwen3 vocabulary).
2. **Supervised Fine-Tuning (SFT):** 4-bit QLoRA trained on 180 curated, human-verified e-free instruction pairs.
3. **Reinforcement Learning (GRPO):** Group Relative Policy Optimization optimizing response fluency, length consistency, and lexical diversity.

## Benchmark Results (Gadsby Benchmark Suite)

| Model Architecture Stage | Strict Lipogram Pass | Usable Response Rate | N-Gram Diversity |
| :--- | :---: | :---: | :---: |
| Base Qwen-1.7B + Mask | 88.0% | 28.0%* | 91.5% |
| SFT (180 Rewrites) + Mask | 96.0% | 80.0% | 91.5% |
| **RL (GRPO Best) + Mask** | **100.0%** | **88.0%** | **92.5%** |

*\*Note: Base model usable rate reflects pure English ASCII responses; unaligned base models attempt foreign-language token substitutions to evade the constraint.*

## Usage

### 1. Requirements
```bash
pip install torch transformers
```

### 2. Standalone Inference with Logit Mask
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load standalone merged model and tokenizer
model_id = "{repo_id}"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# 2. Build forbidden token mask (blocking 'e' and 'E')
vocab_size = model.config.vocab_size
forbidden_mask = torch.zeros(vocab_size, dtype=torch.bool)
for token_id in range(vocab_size):
    token_str = tokenizer.decode([token_id])
    if "e" in token_str.lower():
        forbidden_mask[token_id] = True

# 3. Generate with mask applied to logits
prompt = "Write a short poem about a frog."
inputs = tokenizer(f"<|im_start|>user\\n{prompt}<|im_end|>\\n<|im_start|>assistant\\n", return_tensors="pt").to("cuda")

with torch.no_grad():
    curr_ids = inputs["input_ids"]
    for _ in range(60):
        outputs = model(input_ids=curr_ids)
        scores = outputs.logits[:, -1, :].clone().float()
        scores[:, forbidden_mask.to("cuda")] = -float("inf")
        next_token = torch.multinomial((scores / 0.7).softmax(-1), num_samples=1)
        if next_token.item() == tokenizer.eos_token_id:
            break
        curr_ids = torch.cat([curr_ids, next_token], dim=-1)

print(tokenizer.decode(curr_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip())
```

## Citation & Attribution
- **Author & Post-Training Engineer:** Prateek
- **Base Architecture:** Qwen3-1.7B by the Qwen Team / Alibaba Cloud
- **License:** Apache 2.0
"""


def main():
    parser = argparse.ArgumentParser(description="Upload fine-tuned lipogram model to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, required=True, help="HF repository ID (e.g. prateekfr/qwen3-1.7b-no-e)")
    parser.add_argument("--model-path", type=str, default="checkpoints/rl_best", help="Path to adapter or merged model directory")
    parser.add_argument("--gguf-file", type=str, default=None, help="Optional path to GGUF file to upload into repository")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face User Access Token (Write)")
    parser.add_argument("--private", action="store_true", help="Create private repository (default: public)")
    args = parser.parse_args()

    load_dotenv_file(ROOT / ".env")
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")

    if not token:
        print("\n" + "=" * 70)
        print("ERROR: Hugging Face Token Required!")
        print("=" * 70)
        print("Please provide a Hugging Face Write Token using:")
        print("  1. --token <YOUR_HF_TOKEN>")
        print("  2. In your .env file: HF_TOKEN=<YOUR_HF_TOKEN>")
        print("  3. Run: huggingface-cli login")
        print("\nYou can generate a token at: https://huggingface.co/settings/tokens")
        print("=" * 70 + "\n")
        sys.exit(1)

    model_dir = Path(args.model_path)
    if not model_dir.is_absolute():
        model_dir = ROOT / model_dir

    if (model_dir / "trainable").exists():
        model_dir = model_dir / "trainable"

    if not model_dir.exists():
        print(f"ERROR: Model directory not found at: {model_dir}")
        sys.exit(1)

    is_adapter = (model_dir / "adapter_config.json").exists()
    model_type_str = "PEFT LoRA Adapter" if is_adapter else "Standalone Merged FP16 Model"

    print("=" * 70)
    print("        UPLOADING LIPOGRAM MODEL TO HUGGING FACE HUB")
    print("=" * 70)
    print(f"Target Repository : {args.repo_id}")
    print(f"Source Directory  : {model_dir}")
    print(f"Model Flavor      : {model_type_str}")
    print(f"Visibility        : {'Private' if args.private else 'Public'}")
    print("=" * 70)

    from huggingface_hub import HfApi, create_repo

    api = HfApi(token=token)

    print("\n[1/3] Creating repository on Hugging Face Hub...")
    try:
        create_repo(repo_id=args.repo_id, token=token, private=args.private, exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{args.repo_id}")
    except Exception as e:
        print(f"Notice during create_repo: {e}")

    print("[2/3] Writing Model Card README.md...")
    if is_adapter:
        card_content = ADAPTER_CARD_TEMPLATE.replace("{repo_id}", args.repo_id)
    else:
        card_content = STANDALONE_CARD_TEMPLATE.replace("{repo_id}", args.repo_id)

    card_path = model_dir / "README.md"
    card_path.write_text(card_content, encoding="utf-8")

    print(f"[3/3] Uploading model folder ({model_dir})...")
    api.upload_folder(
        folder_path=str(model_dir),
        repo_id=args.repo_id,
        repo_type="model",
        commit_message=f"Initial release: Qwen3-1.7B lipogram {model_type_str}",
    )

    if args.gguf_file:
        gguf_p = Path(args.gguf_file)
        if not gguf_p.is_absolute():
            gguf_p = ROOT / gguf_p
        if gguf_p.exists():
            print(f"\n[Bonus] Uploading GGUF binary: {gguf_p.name} ({gguf_p.stat().st_size / (1024**3):.2f} GB)...")
            api.upload_file(
                path_or_fileobj=str(gguf_p),
                path_in_repo=gguf_p.name,
                repo_id=args.repo_id,
                repo_type="model",
                commit_message=f"Add GGUF binary: {gguf_p.name}",
            )

    print("\n" + "=" * 70)
    print("SUCCESS: Model successfully published to Hugging Face Hub!")
    print(f"URL: https://huggingface.co/{args.repo_id}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
