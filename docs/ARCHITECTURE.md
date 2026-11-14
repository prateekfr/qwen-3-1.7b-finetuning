# 🏗️ Architecture — Qwen, Minus the E

> **Purpose:** Detailed technical architecture of the project. How every component
> works, how data flows, and how modules connect.

---

## 1. System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INFERENCE PIPELINE                          │
│                                                                    │
│  User Prompt ──► Tokenizer ──► Model (Qwen3-1.7B + LoRA) ──►      │
│                                  │                                 │
│                                  ▼                                 │
│                          Raw Logits (151,936 tokens)               │
│                                  │                                 │
│                                  ▼                                 │
│                     Token Mask (block 103,639 tokens               │
│                      containing e/E/é/etc.)                        │
│                                  │                                 │
│                                  ▼                                 │
│                     Masked Logits ──► Sampling ──► Token           │
│                                                     │              │
│                                              Repeat until EOS      │
│                                                     │              │
│                                                     ▼              │
│                                           E-free response          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 2. Training Pipeline

### Phase 1: Supervised Fine-Tuning (SFT)

```
┌──────────────────────────────────────────────────────────────────┐
│                          SFT PIPELINE                            │
│                                                                  │
│  No Robots + Dolly-15k                                           │
│       │                                                          │
│       ▼                                                          │
│  Filter & rewrite to remove all "e"s ──► 180 curated examples    │
│       │                                                          │
│       ▼                                                          │
│  Qwen3-1.7B (4-bit quantized, frozen)                            │
│       + LoRA adapters (rank=16, α=16, trainable)                 │
│       │                                                          │
│       ▼                                                          │
│  Standard cross-entropy loss on e-free target tokens             │
│       │                                                          │
│       ▼                                                          │
│  Train 10 epochs, save best checkpoint (lowest val loss)         │
└──────────────────────────────────────────────────────────────────┘
```

### Phase 2: Reinforcement Learning (GRPO)

```
┌──────────────────────────────────────────────────────────────────┐
│                         RL / GRPO PIPELINE                       │
│                                                                  │
│  For each of 300 steps:                                          │
│    1. Sample 4 prompts from prompt pool                          │
│    2. Generate N answers per prompt (with mask)                  │
│    3. Score each answer with reward function:                    │
│       - LLM judge for coherence & prompt-following               │
│       - Penalty for constraint violations (shouldn't happen      │
│         with mask, but catches edge cases)                       │
│    4. Compute advantages (GRPO: group-relative)                  │
│    5. Policy gradient update with KL penalty against ref model   │
│       │                                                          │
│       ▼                                                          │
│  Updated LoRA adapter weights                                    │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. Module Breakdown

### `core/mask.py` — Forbidden Token Mask
- Scans the tokenizer vocabulary (151,936 tokens)
- Identifies every token whose decoded text contains `e`, `E`, or accented variants
- Returns a boolean tensor: `True` = allowed, `False` = forbidden
- Applied at each decoding step by setting forbidden logits to `-inf`

### `core/generation.py` — Custom Rollout Generation
- Left-pads prompts for batch generation
- Handles Qwen3's thinking mode (budget tracking)
- Applies the token mask during sampling
- Returns: response IDs, context masks, action masks, log-probs

### `data/prompts/make_prompt_data.py` — Data Pipeline
- Downloads No Robots and Dolly-15k from HuggingFace
- Filters for suitable instruction-response pairs
- Rewrites responses to remove all "e"s (may use LLM assistance)
- Outputs 180 curated examples in chat format
- Splits into train/validation sets

### `training/engine.py` — Custom Training Engine
- Wraps model + optimizer + scheduler
- Handles LoRA weight separation (`.A` and `.B` matrices)
- Checkpoint saving/loading
- Metrics logging to JSON files
- Gradient checkpointing management

### `training/sft.py` — SFT Training Script
- Loads quantized base model + LoRA
- Runs standard causal LM training on e-free examples
- Tracks training/validation loss per epoch
- Saves best checkpoint

### `training/rl/grpo.py` — GRPO Training
- Generates rollouts using `core/generation.py`
- Scores with reward function
- Computes group-relative advantages
- Policy gradient update with KL regularization

### `evaluation/evaluate.py` — Evaluation
- Runs model on test prompts with mask
- Checks: (1) zero "e"s, (2) coherence, (3) prompt relevance
- Reports % of usable responses

---

## 4. LoRA Configuration

| Parameter       | Value                                              |
| --------------- | -------------------------------------------------- |
| Rank (r)        | 16                                                 |
| Alpha (α)       | 16                                                 |
| Dropout         | 0.0 (default)                                      |
| Target Modules  | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |
| Layers          | All 28 transformer layers                          |
| Trainable Params| ~17.4M                                             |
| Precision       | float32 (adapters only)                            |

---

## 5. VRAM Budget (4 GB Target)

| Component                  | Estimated VRAM   |
| -------------------------- | ---------------- |
| Qwen3-1.7B (4-bit NF4)    | ~1.1 GB          |
| LoRA adapters (float32)    | ~0.07 GB         |
| Optimizer states (AdamW)   | ~0.14 GB         |
| Activations (batch=1, grad ckpt) | ~1.0-1.5 GB |
| KV cache during generation | ~0.5-1.0 GB      |
| **Total (training)**       | **~2.8-3.8 GB**  |
| **Headroom**               | **0.2-1.2 GB**   |

> ⚠️ The RL phase is tightest because it needs both training and generation
> in the same step. We may need to alternate: generate → free cache → train.

---

## 6. Key Technical Decisions

1. **Why custom training loops instead of HuggingFace Trainer / TRL?**
   - Matches the original project's approach
   - More educational — we understand every forward/backward pass
   - More control over memory management (critical at 4 GB)
   - Can insert manual `torch.cuda.empty_cache()` between steps

2. **Why QLoRA (4-bit) instead of full bf16?**
   - Qwen3-1.7B in bf16 = ~3.4 GB, leaving <0.6 GB for everything else
   - 4-bit NF4 = ~1.1 GB, leaving ~2.9 GB for training overhead
   - Quality loss from 4-bit is minimal for LoRA fine-tuning

3. **Why not use TRL's GRPOTrainer?**
   - TRL's abstractions hide memory management details
   - On 4 GB VRAM we need granular control over when tensors are allocated/freed
   - Original project also uses custom loops
