"""
4-bit QLoRA Supervised Fine-Tuning (SFT) for Qwen3-1.7B
Optimized for 4 GB VRAM GPUs (NVIDIA RTX 3050 Laptop).

Trains on 180 curated no-e examples with:
- 4-bit NF4 quantization
- LoRA (rank=16, alpha=16) on all 7 projections across all 28 layers
- Batch size 1 + Gradient Accumulation (4 steps) = Effective batch size 4
- Gradient Checkpointing enabled
- Assistant-only loss masking
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from training.sft.sft_utils import load_sft_dataset, get_loader
from scripts.evaluate import NO_E_INSTRUCTION
from core.checks import e_pass, text_checks


def evaluate_val_loss(model, val_loader, device):
    """Compute mean validation loss across validation batches."""
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            num_valid = (shift_labels != -100).sum().item()
            if num_valid > 0:
                total_loss += loss.item()
                total_tokens += num_valid

    model.train()
    return total_loss / max(total_tokens, 1)


def main():
    parser = argparse.ArgumentParser(description="4-bit QLoRA SFT training on Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--grad-accum", type=int, default=4, help="Gradient accumulation steps")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    args = parser.parse_args()

    print("=" * 70)
    print("  Qwen, Minus the E — Phase 3: 4-bit QLoRA SFT Training")
    print("=" * 70)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("❌ CUDA GPU required for training!")
        return

    # Hyperparameters
    epochs = args.epochs
    batch_size = args.batch_size
    grad_accum_steps = args.grad_accum
    learning_rate = args.lr
    max_length = 512
    weights_dir = str(ROOT / "weights" / "qwen3-1.7b")
    train_data_path = str(ROOT / "data" / "rewrites" / "splits" / "train_180.jsonl")
    val_data_path = str(ROOT / "data" / "rewrites" / "splits" / "val.jsonl")

    output_dir = ROOT / "checkpoints" / "sft_best"
    output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = ROOT / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    metrics_log_path = logs_dir / "sft_metrics.jsonl"

    print(f"Device: {torch.cuda.get_device_name(0)}")
    free_mem, total_mem = torch.cuda.mem_get_info()
    print(f"Initial Free VRAM: {free_mem / 1024**3:.2f} GB / Total: {total_mem / 1024**3:.2f} GB")
    print(f"Configuration:")
    print(f"  - Base Model: {weights_dir}")
    print(f"  - Precision: 4-bit NF4 (QLoRA)")
    print(f"  - LoRA: rank=16, alpha=16 (all 7 projection modules)")
    print(f"  - Epochs: {epochs}")
    print(f"  - Batch Size: {batch_size} (Grad Accum: {grad_accum_steps} -> Effective: {batch_size * grad_accum_steps})")
    print(f"  - Learning Rate: {learning_rate}")
    print(f"  - Train Data: {train_data_path}")
    print(f"  - Val Data: {val_data_path}\n")

    # 1. Load Tokenizer
    print("1. Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(weights_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 2. Load 4-bit Quantized Base Model
    print("2. Loading Base Model in 4-bit NF4...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        weights_dir,
        quantization_config=bnb_config,
        device_map={"": 0},
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    # Prepare for k-bit training (enables gradient checkpointing & input require grads)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    # 3. Attach LoRA Adapter
    print("3. Attaching LoRA Adapter...")
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

    # 4. Prepare Datasets & Loaders
    print("\n4. Loading & Tokenizing Datasets...")
    train_dataset = load_sft_dataset(train_data_path, tokenizer, max_length=max_length, system_prompt=NO_E_INSTRUCTION)
    val_dataset = load_sft_dataset(val_data_path, tokenizer, max_length=max_length, system_prompt=NO_E_INSTRUCTION)
    print(f"   Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = get_loader(train_dataset, tokenizer, batch_size=batch_size, shuffle=True)
    val_loader = get_loader(val_dataset, tokenizer, batch_size=batch_size, shuffle=False)

    # 5. Optimizer & Setup
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Initial Validation Loss before training
    print("\n5. Computing Initial Validation Loss...")
    initial_val_loss = evaluate_val_loss(model, val_loader, device)
    print(f"   Initial Held-Out Validation Loss: {initial_val_loss:.4f}\n")

    best_val_loss = initial_val_loss
    best_epoch = 0
    history = []

    print("-" * 70)
    print("  Starting Training Loop")
    print("-" * 70)

    start_time = time.time()
    global_step = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()
        model.train()
        running_train_loss = 0.0
        train_tokens = 0
        optimizer.zero_grad()

        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift logits and labels
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fn(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

            # Scale loss for gradient accumulation
            scaled_loss = loss / grad_accum_steps
            scaled_loss.backward()

            running_train_loss += loss.item()
            num_valid = (shift_labels != -100).sum().item()
            train_tokens += num_valid

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

        avg_train_loss = running_train_loss / len(train_loader)
        val_loss = evaluate_val_loss(model, val_loader, device)
        epoch_duration = time.time() - epoch_start

        # Check for best model
        saved_tag = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            # Save PEFT adapter
            model.save_pretrained(str(output_dir))
            saved_tag = " ⭐ (Best model saved!)"

        log_entry = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(val_loss, 4),
            "best_val_loss": round(best_val_loss, 4),
            "best_epoch": best_epoch,
            "duration_s": round(epoch_duration, 1),
        }
        history.append(log_entry)

        with open(metrics_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Time: {epoch_duration:.1f}s{saved_tag}")

    total_training_time = time.time() - start_time
    print("-" * 70)
    print(f"✅ Training Complete in {total_training_time / 60:.1f} minutes!")
    print(f"   Initial Val Loss: {initial_val_loss:.4f}")
    print(f"   Best Val Loss:    {best_val_loss:.4f} (at Epoch {best_epoch})")
    print(f"   Saved Adapter:    {output_dir}")
    print("-" * 70)


if __name__ == "__main__":
    main()
