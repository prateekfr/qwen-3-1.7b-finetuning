"""
4-bit QLoRA GRPO (Group Relative Policy Optimization) Training for Qwen3-1.7B
Optimized specifically for 4 GB VRAM GPUs (NVIDIA RTX 3050 Laptop).

Key Architectural Optimizations:
1. Base weights in 4-bit NF4 quantization (~1.1 GB).
2. Dual-adapter PEFT architecture:
   - 'trainable': Active policy being updated via policy gradients.
   - 'reference': Frozen baseline anchored to SFT starting weights (checkpoints/sft_best).
   Both share the identical 4-bit base weights, consuming only ~66 MB extra VRAM!
3. Micro-batch size 1 during rollout generation and backward pass.
4. Token mask decoding at inference time (-inf on all tokens with 'e').
5. Unified Reward Manager (defaults to Local Composite Heuristics, toggleable to OpenRouter/OpenAI API).
6. KL divergence penalty anchoring policy to SFT distribution.
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, prepare_model_for_kbit_training

from core.sampling import build_forbidden_token_mask
from core.checks import e_pass, text_checks
from core.rewards import get_reward_manager
from scripts.evaluate import NO_E_INSTRUCTION


def generate_single_candidate(
    model,
    tokenizer,
    prompt_ids: torch.Tensor,
    mask: torch.Tensor,
    max_new_tokens: int = 50,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Generate a single candidate completion autoregressively with token masking.
    Returns generated token IDs, log-probabilities, and decoded text.
    """
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    eos_id = tokenizer.eos_token_id
    device = prompt_ids.device

    curr_ids = prompt_ids.clone()
    gen_tokens = []
    gen_log_probs = []

    with torch.no_grad():
        for _ in range(max_new_tokens):
            outputs = model(input_ids=curr_ids)
            logits = outputs.logits[:, -1, :].clone().float()

            # Apply forbidden token mask (score -> -inf for all tokens containing 'e')
            logits[:, mask] = -float("inf")

            # Temperature-scaled log probabilities
            log_dist = (logits / temperature).log_softmax(-1)
            next_token = torch.multinomial(log_dist.exp(), num_samples=1)

            token_id = next_token.item()
            token_log_prob = log_dist[0, token_id].item()

            if token_id == eos_id:
                break

            gen_tokens.append(token_id)
            gen_log_probs.append(token_log_prob)
            curr_ids = torch.cat([curr_ids, next_token], dim=-1)

    decoded_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
    return {
        "token_ids": gen_tokens,
        "log_probs": gen_log_probs,
        "text": decoded_text,
    }


def compute_sequence_log_probs(
    model,
    prompt_ids: torch.Tensor,
    response_ids: List[int],
    mask: torch.Tensor,
    temperature: float = 0.7,
) -> torch.Tensor:
    """
    Forward pass on (prompt + response) to compute log probabilities of the response tokens.
    """
    device = prompt_ids.device
    resp_tensor = torch.tensor([response_ids], dtype=torch.long, device=device)
    full_ids = torch.cat([prompt_ids, resp_tensor], dim=-1)
    attn_mask = torch.ones_like(full_ids)

    outputs = model(input_ids=full_ids, attention_mask=attn_mask)
    logits = outputs.logits.float()

    prompt_len = prompt_ids.size(1)
    # Logits predicting response tokens start at prompt_len - 1
    resp_logits = logits[:, prompt_len - 1 : -1, :]

    # Apply mask
    resp_logits = resp_logits.masked_fill(mask.view(1, 1, -1), -float("inf"))
    log_probs = (resp_logits / temperature).log_softmax(dim=-1)

    # Gather log-probs for actual tokens
    target_tokens = resp_tensor.unsqueeze(-1)  # [1, resp_len, 1]
    gathered = log_probs.gather(dim=-1, index=target_tokens).squeeze(-1)  # [1, resp_len]
    return gathered[0]


def main():
    parser = argparse.ArgumentParser(description="4-bit QLoRA GRPO Training on Qwen3-1.7B")
    parser.add_argument("--steps", type=int, default=100, help="Total GRPO optimization steps")
    parser.add_argument("--group-size", type=int, default=4, help="Number of rollouts per prompt (G)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate for policy updates")
    parser.add_argument("--kl-beta", type=float, default=0.02, help="KL divergence penalty coefficient")
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clipping epsilon")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--max-new-tokens", type=int, default=50, help="Max generated tokens per rollout")
    parser.add_argument("--save-every", type=int, default=20, help="Save checkpoint every N steps")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoints/rl_best if exists")
    args = parser.parse_args()

    print("=" * 70)
    print("  Qwen, Minus the E — Phase 4: 4-bit QLoRA GRPO Training")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("❌ CUDA GPU required for GRPO training!")
        return

    # Paths
    weights_dir = ROOT / "weights" / "qwen3-1.7b"
    sft_adapter_dir = ROOT / "checkpoints" / "sft_best"
    rl_output_dir = ROOT / "checkpoints" / "rl_best"
    rl_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = logs_dir / "grpo_metrics.jsonl"
    prompts_file = ROOT / "data" / "prompts" / "rl_pool.jsonl"

    # Reward Manager (Reads configs/default.yaml -> Local or API)
    reward_mgr = get_reward_manager()
    print(f"Device: {torch.cuda.get_device_name(0)}")
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Initial Free VRAM: {free_mem / 1024**3:.2f} GB / Total: {total_mem / 1024**3:.2f} GB")
    print(f"Configuration:")
    print(f"  - Active Reward Mode: {reward_mgr.mode.upper()}")
    print(f"  - GRPO Steps: {args.steps}")
    print(f"  - Group Size (G): {args.group_size} rollouts per prompt")
    print(f"  - Learning Rate: {args.lr}")
    print(f"  - KL Penalty Beta: {args.kl_beta}")
    print(f"  - PPO Clip Epsilon: {args.clip_eps}")
    print(f"  - Max New Tokens: {args.max_new_tokens}")
    print(f"  - Prompts Source: {prompts_file}\n")

    # 1. Load Tokenizer
    print("1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(str(weights_dir), trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load 4-bit Base Model
    print("2. Loading Base Model in 4-bit NF4...")
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
    base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=False)

    # 3. Attach Dual Adapters (Trainable Policy + Frozen Reference)
    start_from_dir = str(rl_output_dir if (args.resume and (rl_output_dir / "adapter_model.safetensors").exists()) else sft_adapter_dir)
    print(f"3. Attaching Trainable Policy Adapter from {start_from_dir}...")
    model = PeftModel.from_pretrained(base_model, start_from_dir, adapter_name="trainable", is_trainable=True)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    print(f"   Attaching Frozen Reference Adapter from {sft_adapter_dir}...")
    model.load_adapter(str(sft_adapter_dir), adapter_name="reference")
    # Ensure reference adapter requires no gradients
    for name, param in model.named_parameters():
        if "reference" in name:
            param.requires_grad = False

    model.set_adapter("trainable")
    model.print_trainable_parameters()

    # 4. Build Forbidden Token Mask
    print("\n4. Building Forbidden Token Mask...")
    vocab_size = model.config.vocab_size
    mask = build_forbidden_token_mask(tokenizer, vocab_size, forbidden_chars="eE", ascii_only=True, device="cuda")

    # 5. Load Prompt Pool
    print("\n5. Loading Prompt Pool...")
    raw_prompts = [json.loads(line)["prompt"] for line in prompts_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"   Loaded {len(raw_prompts)} prompts from RL pool.")
    random.seed(42)
    random.shuffle(raw_prompts)

    # 6. Optimizer
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    print("-" * 70)
    print("  Starting GRPO Training Loop")
    print("-" * 70)

    start_time = time.time()
    best_mean_reward = 0.0
    history = []

    for step in range(1, args.steps + 1):
        step_start = time.time()
        prompt_text = raw_prompts[(step - 1) % len(raw_prompts)]

        # Prepare chat format input
        messages = [
            {"role": "system", "content": NO_E_INSTRUCTION},
            {"role": "user", "content": prompt_text},
        ]
        chat_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        prompt_ids = tokenizer(chat_text, return_tensors="pt")["input_ids"].to("cuda")

        # --- A. Rollout Generation (G Candidates) ---
        model.eval()
        model.set_adapter("trainable")
        candidates = []
        for _ in range(args.group_size):
            cand = generate_single_candidate(
                model, tokenizer, prompt_ids, mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
            )
            candidates.append(cand)

        # --- B. Reward Scoring ---
        rewards = []
        diagnostics = []
        for cand in candidates:
            r_val, diag = reward_mgr.evaluate(prompt_text, cand["text"])
            rewards.append(r_val)
            diagnostics.append(diag)

        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device="cuda")
        mean_reward = rewards_tensor.mean().item()
        std_reward = rewards_tensor.std().item()

        # --- C. Advantage Estimation ---
        if std_reward > 1e-4:
            advantages = (rewards_tensor - mean_reward) / (std_reward + 1e-6)
        else:
            # If all candidates scored identically, zero advantages (no noisy updates)
            advantages = torch.zeros_like(rewards_tensor)

        # --- D. Policy Gradient Updates (Microbatch Size 1) ---
        model.train()
        model.set_adapter("trainable")
        optimizer.zero_grad()

        total_step_loss = 0.0
        total_kl = 0.0
        active_candidates = 0

        for i, cand in enumerate(candidates):
            adv = advantages[i].item()
            token_ids = cand["token_ids"]
            if len(token_ids) == 0:
                continue

            active_candidates += 1
            old_log_probs = torch.tensor(cand["log_probs"], dtype=torch.float32, device="cuda")

            # 1. Forward pass with trainable adapter
            model.set_adapter("trainable")
            curr_log_probs = compute_sequence_log_probs(
                model, prompt_ids, token_ids, mask, temperature=args.temperature
            )

            # 2. Forward pass with reference adapter (in no_grad)
            with torch.no_grad():
                model.set_adapter("reference")
                ref_log_probs = compute_sequence_log_probs(
                    model, prompt_ids, token_ids, mask, temperature=args.temperature
                )
            model.set_adapter("trainable")

            # 3. Policy Ratio r_t & PPO-Clipped Surrogate
            ratio = torch.exp(curr_log_probs - old_log_probs)
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - args.clip_eps, 1.0 + args.clip_eps) * adv
            surrogate = torch.min(surr1, surr2)

            # 4. KL Divergence Penalty
            log_ref_ratio = ref_log_probs - curr_log_probs
            kl_div = torch.exp(log_ref_ratio) - log_ref_ratio - 1.0
            kl_mean = kl_div.mean()

            # 5. GRPO Loss for this sequence
            sequence_obj = surrogate - args.kl_beta * kl_div
            cand_loss = -sequence_obj.mean()

            # Scale for accumulation across group size G
            scaled_loss = cand_loss / max(1, args.group_size)
            scaled_loss.backward()

            total_step_loss += cand_loss.item()
            total_kl += kl_mean.item()

        if active_candidates > 0:
            torch.nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        step_duration = time.time() - step_start
        avg_loss = total_step_loss / max(1, active_candidates)
        avg_kl = total_kl / max(1, active_candidates)

        # Track Best Model
        saved_tag = ""
        if mean_reward > best_mean_reward and mean_reward >= 0.5:
            best_mean_reward = mean_reward
            model.set_adapter("trainable")
            model.save_pretrained(str(rl_output_dir))
            trainable_sub = rl_output_dir / "trainable"
            if trainable_sub.exists():
                for f in trainable_sub.glob("*"):
                    (rl_output_dir / f.name).write_bytes(f.read_bytes())
            saved_tag = " ⭐ (Best RL model saved!)"

        log_entry = {
            "step": step,
            "prompt": prompt_text[:60] + ("..." if len(prompt_text) > 60 else ""),
            "mean_reward": round(mean_reward, 4),
            "best_reward": round(max(rewards), 4),
            "std_reward": round(std_reward, 4),
            "rewards": [round(r, 4) for r in rewards],
            "loss": round(avg_loss, 4),
            "kl": round(avg_kl, 4),
            "duration_s": round(step_duration, 1),
            "best_mean_reward": round(best_mean_reward, 4),
        }
        history.append(log_entry)

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")

        # Print step progress
        print(f"Step {step:03d}/{args.steps:03d} | Mean Reward: {mean_reward:.4f} (Max: {max(rewards):.4f}) | KL: {avg_kl:.4f} | Time: {step_duration:.1f}s{saved_tag}", flush=True)

        # Periodic checkpoint
        if step % args.save_every == 0:
            model.set_adapter("trainable")
            periodic_dir = ROOT / "checkpoints" / f"rl_step_{step}"
            model.save_pretrained(str(periodic_dir))

    total_time = time.time() - start_time
    print("-" * 70)
    print(f"✅ GRPO Training Complete in {total_time / 60:.1f} minutes!")
    print(f"   Best Mean Group Reward: {best_mean_reward:.4f}")
    print(f"   Saved Adapter Checkpoint: {rl_output_dir}")
    print("-" * 70)


if __name__ == "__main__":
    main()
