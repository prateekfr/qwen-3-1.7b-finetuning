# 🧠 Project Memory — Qwen, Minus the E

> **Purpose:** This file is the master context document for this project. Any fresh
> AI session should read this file FIRST to understand the full state of the project,
> all decisions made, and what to do next.
>
> **Last Updated:** 2026-11-14

---

## 1. What Is This Project?

We are replicating the project [kevinyvv/qwen-3-1.7b-finetune-proj](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)
("Qwen, Minus the E") by Kevin Wu.

**Goal:** Fine-tune the **Qwen3-1.7B** language model to write coherent English text
that **never uses the letter "e"** — a literary constraint called a **lipogram**.

**Original blog post:** https://kevin-wu.ca (October 2026)

---

## 2. How It Works (3-Layer Approach)

### Layer 1: Token Mask (Hard Constraint)
- At decoding time, every token containing `e`, `E`, `é`, etc. gets score = `-inf`
- Out of Qwen's 151,936 tokens, the mask blocks ~103,639 (68%)
- Guarantees zero "e"s structurally, but alone produces gibberish

### Layer 2: Supervised Fine-Tuning (SFT)
- 180 curated chat examples written without the letter "e"
- Sources: HuggingFace *No Robots* + *Databricks Dolly-15k* (filtered & rewritten)
- Teaches the model basic e-free sentence structure

### Layer 3: Reinforcement Learning (GRPO)
- 300 steps × 4 prompts × N answers each
- Reward via LLM judge scoring coherence + prompt-following
- Kevin improved usable answers from 35.0% → 60.5%

---

## 3. Our Hardware Constraints

| Component         | Spec                                              |
| ----------------- | ------------------------------------------------- |
| CPU               | AMD Ryzen 7 4800H (8 cores / 16 threads)          |
| RAM               | 8 GB                                              |
| GPU               | NVIDIA GeForce RTX 3050 Laptop — **4 GB VRAM**    |
| CUDA              | 12.9                                              |
| Driver            | 576.88                                            |
| Disk              | D: 931.5 GB total, ~252 GB free                   |
| Python            | 3.14.4 system, conda 25.5.1 available             |
| OS                | Windows                                           |

### Kevin's Hardware vs Ours
- Kevin used a **16 GB VRAM** GPU — we have **4 GB** (4× less)
- This forces us to use **4-bit quantization (QLoRA)** instead of bf16
- Batch size reduced from 4 to 1 with gradient accumulation
- Gradient checkpointing is mandatory

---

## 4. Key Adaptations From Original

| Kevin's Approach                | Our Adaptation                                      |
| ------------------------------- | --------------------------------------------------- |
| bf16 base weights (~3.4 GB)     | 4-bit NF4 quantization (~1.1 GB)                    |
| Batch size 4                    | Batch size 1 + gradient accumulation (4 steps)       |
| No gradient checkpointing       | Gradient checkpointing ON                            |
| 8 answers per prompt in RL      | 4 answers per prompt (may adjust)                    |
| 300-token gen in RL             | 200-300 tokens (test what fits)                      |
| Custom training loops           | Keep custom, add memory optimizations                |
| Python env unspecified          | Conda env with Python 3.11 for compatibility         |

---

## 5. Project Structure

```
qwen-3-1.7b-finetuning/
├── README.md                    # Project overview & setup instructions
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
│
├── docs/
│   ├── MEMORY.md                # ← YOU ARE HERE — master context file
│   ├── LEARNING_GUIDE.md        # 🎓 Complete AI Engineering & Fine-tuning textbook
│   ├── ARCHITECTURE.md          # Technical architecture & approach
│   ├── PLAN.md                  # Phase-by-phase plan with status tracking
│   ├── EXPERIMENT_LOG.md        # Running log of experiments & results
│   └── HARDWARE.md              # Hardware analysis & VRAM budget
│
├── configs/
│   └── default.yaml             # All training hyperparameters
│
├── core/
│   ├── __init__.py
│   ├── generation.py            # Rollout generation, token masking
│   └── mask.py                  # Forbidden token mask builder
│
├── data/
│   ├── prompts/
│   │   └── make_prompt_data.py  # Data pipeline: fetch & curate datasets
│   ├── raw/                     # Downloaded raw datasets (gitignored)
│   └── processed/               # Processed training data (gitignored)
│
├── training/
│   ├── __init__.py
│   ├── engine.py                # Custom training engine, checkpointing
│   ├── sft.py                   # SFT training script
│   └── rl/
│       ├── __init__.py
│       └── grpo.py              # GRPO reinforcement learning
│
├── evaluation/
│   ├── __init__.py
│   └── evaluate.py              # Evaluation & benchmarking
│
├── scripts/
│   └── verify_env.py            # Environment verification script
│
├── checkpoints/                 # Saved model weights (gitignored)
└── logs/                        # Training logs & metrics (gitignored)
```

---

## 6. Current Status

### ✅ Completed
- [x] Analyzed original project in full (all 13 commits, architecture, lab notebook)
- [x] Cloned full original reference repo into `reference_repo/` for 100% complete reference access
- [x] Assessed hardware feasibility (16 GB original vs 4 GB RTX 3050 Laptop)
- [x] Created clean folder structure & comprehensive markdown memory
- [x] Phase 0: Environment setup (`qwen-no-e` Conda env, PyTorch cu124, bitsandbytes, transformers, peft)
- [x] Phase 1: Data pipeline (imported 200 reviewed rewrites, train_180, val, test, and RL pools)
- [x] Phase 2: Token mask verified (103,639 blocked tokens / 68.33%, matching Kevin's count exactly)
- [x] Phase 3: SFT 4-bit QLoRA completed on RTX 3050 GPU
  - Downloaded `Qwen3-1.7B` base weights (3.87 GB)
  - Trained 4 full epochs: Train loss dropped from 3.1071 -> 1.8191; Val loss reached optimum minimum of **2.5692** at Epoch 3
  - Verified early stopping logic saved the optimal Epoch 3 adapter weights to `checkpoints/sft_best/` (66.55 MB)
  - Confirmed generated text produces fluent, grammatical sentences with **0% 'e's** (`e_pass=True`, `e_count=0`) across diverse test prompts

### 🔄 In Progress
- [ ] Phase 4: Reinforcement Learning (GRPO)

### ⏳ Not Started
- [ ] Phase 5: Evaluation & Publishing
- [ ] Phase 6: Stretch Goals

---

## 7. Key Decisions Log

| Date       | Decision                                         | Reasoning                                  |
| ---------- | ------------------------------------------------ | ------------------------------------------ |
| 2026-11-14 | Use QLoRA (4-bit NF4) instead of bf16            | Only 4 GB VRAM vs Kevin's 16 GB            |
| 2026-11-14 | Batch size 1 + grad accum 4                      | VRAM constraint                            |
| 2026-11-14 | Run everything locally, no cloud GPU             | User preference                            |
| 2026-11-14 | Use conda env with Python 3.11                   | PyTorch/transformers compatibility          |
| 2026-11-14 | Keep custom training loops (not TRL/Trainer)      | Match original project's educational value |

---

## 8. Reference Links

- **Original Repo:** https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj
- **Blog Post:** https://kevin-wu.ca
- **HF Model (final RL):** https://huggingface.co/k33wu/qwen3-1.7b-no-e
- **HF Model (SFT only):** https://huggingface.co/k33wu/qwen3-1.7b-no-e-sft180
- **Base Model:** https://huggingface.co/Qwen/Qwen3-1.7B
- **No Robots Dataset:** https://huggingface.co/datasets/HuggingFaceH4/no_robots
- **Dolly-15k Dataset:** https://huggingface.co/datasets/databricks/databricks-dolly-15k

---

## 9. How to Resume Work

If you are a new AI session picking this up:

1. **Read this file first** (`docs/MEMORY.md`)
2. **Check `docs/PLAN.md`** for current phase and status
3. **Check `docs/EXPERIMENT_LOG.md`** for latest experiment results
4. **Check `docs/ARCHITECTURE.md`** for technical details
5. **Check `docs/HARDWARE.md`** for VRAM budget and constraints
6. Continue from wherever the plan left off
