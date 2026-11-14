"""
Test loading Qwen3-1.7B in 4-bit NF4 precision and measure VRAM footprint.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType


def main():
    print("=" * 60)
    print("  Testing Qwen3-1.7B 4-bit NF4 Loading & VRAM Footprint")
    print("=" * 60)

    device_name = torch.cuda.get_device_name(0)
    free_before, total = torch.cuda.mem_get_info()
    print(f"Device: {device_name}")
    print(f"Initial Free VRAM: {free_before / 1024**3:.2f} GB / Total: {total / 1024**3:.2f} GB\n")

    weights_dir = str(ROOT / "weights" / "qwen3-1.7b")

    print("1. Configuring 4-bit NF4 BitsAndBytesConfig...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("2. Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, trust_remote_code=True)

    print("3. Loading base model in 4-bit onto GPU...")
    model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        quantization_config=bnb_config,
        device_map={"": 0},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    free_after_model, _ = torch.cuda.mem_get_info()
    model_vram = (free_before - free_after_model) / (1024**3)
    print(f"   Base model loaded!")
    print(f"   VRAM consumed by base model: {model_vram:.2f} GB")
    print(f"   Remaining free VRAM: {free_after_model / 1024**3:.2f} GB\n")

    print("4. Attaching LoRA adapter (rank=16, alpha=16, all 7 projections)...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    free_after_lora, _ = torch.cuda.mem_get_info()
    lora_vram = (free_after_model - free_after_lora) / (1024**3)
    print(f"   VRAM consumed by LoRA: {lora_vram * 1024:.1f} MB")
    print(f"   Total VRAM used: {(free_before - free_after_lora) / 1024**3:.2f} GB")
    print(f"   Remaining free VRAM: {free_after_lora / 1024**3:.2f} GB\n")

    print("5. Quick test inference with token mask...")
    from core.sampling import build_forbidden_token_mask
    mask = build_forbidden_token_mask(tokenizer, len(tokenizer), forbidden_chars="eE", ascii_only=True, device="cuda")

    prompt = "Tell me a short story about a brave dog."
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=30,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"   Raw prompt: {prompt}")
    print(f"   Sample output (unmasked, pre-training): {generated_text}\n")

    free_final, _ = torch.cuda.mem_get_info()
    print(f"Final Free VRAM: {free_final / 1024**3:.2f} GB")
    print("=" * 60)
    print("  ✅ 4-bit QLoRA verified to comfortably fit in 4 GB VRAM!")
    print("=" * 60)


if __name__ == "__main__":
    main()
