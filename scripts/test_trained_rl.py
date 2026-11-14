"""
Compare SFT Model vs RL (GRPO) Model generations on challenging test prompts.
Verifies constraint adherence, response length, fluency, and reward scores.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from core.sampling import build_forbidden_token_mask
from core.checks import e_pass, text_checks
from core.rewards import get_reward_manager
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
    eos_id = tokenizer.eos_token_id

    generated_ids = []
    curr_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=curr_ids)
            scores = outputs.logits[:, -1, :].clone().float()

            # Apply forbidden token mask (score -> -inf for all tokens containing 'e')
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
    print("  Comparing SFT vs RL (GRPO) Model Generations")
    print("=" * 70)

    weights_dir = ROOT / "weights" / "qwen3-1.7b"
    sft_adapter_dir = ROOT / "checkpoints" / "sft_best"
    rl_adapter_dir = ROOT / "checkpoints" / "rl_best"
    if (rl_adapter_dir / "trainable").exists():
        rl_adapter_dir = rl_adapter_dir / "trainable"
    elif (ROOT / "checkpoints" / "rl_step_30" / "trainable").exists():
        rl_adapter_dir = ROOT / "checkpoints" / "rl_step_30" / "trainable"

    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    print("Loading 4-bit base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(weights_dir),
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    print(f"Loading SFT adapter from {sft_adapter_dir}...")
    model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir), adapter_name="sft")
    print(f"Loading RL adapter from {rl_adapter_dir}...")
    model.load_adapter(str(rl_adapter_dir), adapter_name="rl")
    model.eval()

    reward_mgr = get_reward_manager()
    vocab_size = model.config.vocab_size
    mask = build_forbidden_token_mask(tokenizer, vocab_size, forbidden_chars="eE", ascii_only=True, device="cuda")

    test_prompts = [
        "What is a dog?",
        "Why is water important?",
        "Say hi and tell me about the sun.",
        "How do birds fly?",
        "What did people do before money existed?",
        "Write a short poem about a frog.",
    ]

    print("\n" + "=" * 70)
    print("  Generation Results: SFT vs RL (GRPO)")
    print("=" * 70)

    for prompt in test_prompts:
        print(f"\n👉 PROMPT: \"{prompt}\"")

        # 1. SFT generation
        model.set_adapter("sft")
        sft_ans = generate_with_mask(model, tokenizer, prompt, mask, max_new_tokens=50, temperature=0.7)
        sft_reward, sft_diag = reward_mgr.evaluate(prompt, sft_ans)

        # 2. RL generation
        model.set_adapter("rl")
        rl_ans = generate_with_mask(model, tokenizer, prompt, mask, max_new_tokens=50, temperature=0.7)
        rl_reward, rl_diag = reward_mgr.evaluate(prompt, rl_ans)

        print(f"  [SFT] \"{sft_ans}\"")
        print(f"        Score: {sft_reward:.4f} | e_pass={e_pass(sft_ans)} | Tokens={len(sft_ans.split())}")
        print(f"  [RL]  \"{rl_ans}\"")
        print(f"        Score: {rl_reward:.4f} | e_pass={e_pass(rl_ans)} | Tokens={len(rl_ans.split())}")


if __name__ == "__main__":
    main()
