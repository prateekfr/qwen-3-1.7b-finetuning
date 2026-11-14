"""
Interactive CLI for Chatting with Qwen3-1.7B Lipogram Models (Minus the 'E').
Allows real-time interactive conversations without the letter 'e'.
Supports switching between Base, SFT, and RL (GRPO) models on the fly.
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from core.sampling import build_forbidden_token_mask
from core.checks import e_pass
from core.rewards import get_reward_manager
from scripts.evaluate import NO_E_INSTRUCTION

# ANSI Color codes for clean terminal presentation
CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
MAGENTA = "\033[95m"
BOLD = "\033[1m"
RESET = "\033[0m"


def print_banner():
    print(f"{CYAN}{BOLD}" + "=" * 75)
    print("      💬 QWEN3-1.7B LIPOGRAM INTERACTIVE TERMINAL (MINUS THE 'E')")
    print("=" * 75 + f"{RESET}")
    print(f"{YELLOW}Constraints  :{RESET} Zero occurrences of 'e' or 'E' in output.")
    print(f"{YELLOW}Commands     :{RESET}")
    print(f"  {BOLD}:mode rl{RESET}    - Switch to RL (GRPO) fine-tuned model (Default)")
    print(f"  {BOLD}:mode sft{RESET}   - Switch to SFT fine-tuned model")
    print(f"  {BOLD}:mode base{RESET}  - Switch to Base Qwen model + token mask")
    print(f"  {BOLD}:temp <val>{RESET} - Set temperature (e.g. :temp 0.7)")
    print(f"  {BOLD}:tokens <n>{RESET} - Set max new tokens (e.g. :tokens 80)")
    print(f"  {BOLD}exit / quit{RESET}- Exit terminal")
    print(f"{CYAN}" + "=" * 75 + f"{RESET}\n")


def generate_response(model, tokenizer, prompt, mask, max_tokens=60, temp=0.7):
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

    start_t = time.time()
    with torch.no_grad():
        for _ in range(max_tokens):
            outputs = model(input_ids=curr_ids)
            scores = outputs.logits[:, -1, :].clone().float()

            # Apply forbidden token mask
            scores[:, mask] = -float("inf")

            # Sample next token
            log_dist = (scores / temp).log_softmax(-1)
            next_token = torch.multinomial(log_dist.exp(), num_samples=1)

            token_id = next_token.item()
            if token_id == eos_id:
                break

            generated_ids.append(token_id)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    gen_time = time.time() - start_t
    response = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return response, len(generated_ids), gen_time


def main():
    weights_dir = ROOT / "weights" / "qwen3-1.7b"
    sft_adapter_dir = ROOT / "checkpoints" / "sft_best"
    rl_adapter_dir = ROOT / "checkpoints" / "rl_best"
    if (rl_adapter_dir / "trainable").exists():
        rl_adapter_dir = rl_adapter_dir / "trainable"

    print(f"{YELLOW}Initializing Qwen3-1.7B in 4-bit NF4...{RESET}")
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        str(weights_dir),
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(base_model, str(sft_adapter_dir), adapter_name="sft")
    has_rl = False
    if rl_adapter_dir.exists():
        model.load_adapter(str(rl_adapter_dir), adapter_name="rl")
        has_rl = True
        current_mode = "rl"
        model.set_adapter("rl")
    else:
        current_mode = "sft"
        model.set_adapter("sft")

    model.eval()

    mask = build_forbidden_token_mask(
        tokenizer,
        model.config.vocab_size,
        forbidden_chars="eE",
        ascii_only=False,
        device="cuda",
        preserve_added_tokens=False,
    )

    reward_mgr = get_reward_manager()
    print_banner()

    max_tokens = 60
    temp = 0.7

    print(f"{GREEN}✓ Ready! Active Model: [{current_mode.upper()}]{RESET}\n")

    while True:
        try:
            prompt = input(f"{BOLD}User > {RESET}").strip()
            if not prompt:
                continue

            if prompt.lower() in ["exit", "quit", ":q"]:
                print(f"\n{YELLOW}Exiting chat session. Goodbye!{RESET}")
                break

            # Command handling
            if prompt.startswith(":mode"):
                parts = prompt.split()
                if len(parts) > 1:
                    target_mode = parts[1].lower()
                    if target_mode == "rl":
                        if has_rl:
                            model.set_adapter("rl")
                            current_mode = "rl"
                            print(f"{GREEN}Switched to RL (GRPO) model.{RESET}\n")
                        else:
                            print(f"{RED}RL adapter not found.{RESET}\n")
                    elif target_mode == "sft":
                        model.set_adapter("sft")
                        current_mode = "sft"
                        print(f"{GREEN}Switched to SFT model.{RESET}\n")
                    elif target_mode == "base":
                        current_mode = "base"
                        print(f"{GREEN}Switched to Base model (adapters disabled).{RESET}\n")
                    else:
                        print(f"{RED}Unknown mode '{target_mode}'. Choose: rl, sft, base.{RESET}\n")
                continue

            if prompt.startswith(":temp"):
                parts = prompt.split()
                if len(parts) > 1:
                    try:
                        temp = float(parts[1])
                        print(f"{GREEN}Temperature set to {temp}.{RESET}\n")
                    except ValueError:
                        print(f"{RED}Invalid float value.{RESET}\n")
                continue

            if prompt.startswith(":tokens"):
                parts = prompt.split()
                if len(parts) > 1:
                    try:
                        max_tokens = int(parts[1])
                        print(f"{GREEN}Max tokens set to {max_tokens}.{RESET}\n")
                    except ValueError:
                        print(f"{RED}Invalid integer value.{RESET}\n")
                continue

            # Generate
            if current_mode == "base":
                with model.disable_adapter():
                    response, tok_count, gen_time = generate_response(
                        model, tokenizer, prompt, mask, max_tokens, temp
                    )
            else:
                response, tok_count, gen_time = generate_response(
                    model, tokenizer, prompt, mask, max_tokens, temp
                )

            # Evaluate constraint & reward
            e_count = response.lower().count("e")
            lipogram_ok = e_pass(response) and (e_count == 0)
            reward, diag = reward_mgr.evaluate(prompt, response)

            e_status = f"{GREEN}PASS (0 'e'){RESET}" if lipogram_ok else f"{RED}FAIL ({e_count} 'e'){RESET}"
            reward_color = GREEN if reward >= 0.70 else (YELLOW if reward >= 0.40 else RED)

            print(f"\n{CYAN}{BOLD}Assistant ({current_mode.upper()}) >{RESET} {response}")
            print(f"{MAGENTA}── [Stats: {tok_count} tokens in {gen_time:.2f}s ({tok_count/gen_time:.1f} tok/s) | Lipogram: {e_status} | Reward: {reward_color}{reward:.4f}{RESET}{MAGENTA}] ──{RESET}\n")

        except KeyboardInterrupt:
            print(f"\n{YELLOW}Session ended by user.{RESET}")
            break
        except Exception as err:
            print(f"\n{RED}Error during generation: {err}{RESET}\n")


if __name__ == "__main__":
    main()
