"""
Test trained SFT adapter with forbidden token mask on sample prompts.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from core.sampling import build_forbidden_token_mask
from core.checks import e_pass, text_checks
from scripts.evaluate import NO_E_INSTRUCTION


def generate_with_mask(model, tokenizer, prompt, mask, max_new_tokens=60, temperature=0.7):
    messages = [
        {"role": "system", "content": NO_E_INSTRUCTION},
        {"role": "user", "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(prompt_text, return_tensors="pt")["input_ids"].to("cuda")

    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    curr_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=curr_ids)
            scores = outputs.logits[:, -1, :].clone().float()

            # Apply forbidden token mask (score -> -inf for tokens with 'e' or non-ascii)
            scores[:, mask] = -float("inf")

            # Sample next token
            log_dist = (scores / temperature).log_softmax(-1)
            next_token = torch.multinomial(log_dist.exp(), num_samples=1)

            token_id = next_token.item()
            if token_id == eos_id:
                break

            generated_ids.append(token_id)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    return tokenizer.decode(generated_ids, skip_special_tokens=True).strip()


def main():
    print("=" * 70)
    print("  Testing SFT Adapter + Token Mask Generations")
    print("=" * 70)

    weights_dir = str(ROOT / "weights" / "qwen3-1.7b")
    adapter_dir = str(ROOT / "checkpoints" / "sft_best")

    tokenizer = AutoTokenizer.from_pretrained(weights_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading base model in 4-bit...")
    base_model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading SFT LoRA adapter from {adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    print("Building forbidden token mask...")
    vocab_size = model.config.vocab_size
    mask = build_forbidden_token_mask(tokenizer, vocab_size, forbidden_chars="eE", ascii_only=True, device="cuda")

    test_prompts = [
        "What is a dog?",
        "Why is water important?",
        "Say hi and tell me about the sun.",
        "How do birds fly?",
    ]

    print("\n" + "=" * 70)
    print("  Generated Responses (With Hard Token Mask):")
    print("=" * 70)

    for prompt in test_prompts:
        answer = generate_with_mask(model, tokenizer, prompt, mask, max_new_tokens=50, temperature=0.7)
        checks = text_checks(answer)
        pass_rule = e_pass(answer)
        print(f"\nUser Prompt: \"{prompt}\"")
        print(f"Model Answer: \"{answer}\"")
        print(f"Constraint Check: e_pass={pass_rule} (e_count={checks['e_count']}, norm_no_e={checks['normalized_no_e']})")


if __name__ == "__main__":
    main()
