# Qwen, Minus the E: Constrained Language Modeling and Lipogram Post-Training

[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0%2Bcu124-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org/)
[![PEFT / QLoRA](https://img.shields.io/badge/PEFT-QLoRA%204--bit%20NF4-yellow.svg)](https://github.com/huggingface/peft)
[![VRAM](https://img.shields.io/badge/VRAM%20Cap-4.0%20GB%20(RTX%203050)-green.svg)](https://nvidia.com)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Constraint](https://img.shields.io/badge/Constraint-Strict%20Lipogram%20(Zero%20E)-red.svg)](https://en.wikipedia.org/wiki/Lipogram)

This repository provides an end-to-end post-training pipeline for fine-tuning **Qwen3-1.7B** to adhere to a strict character-level constraint: generating coherent, grammatically sound English without ever using the letter "e" or "E" (a lipogram). The pipeline is engineered to execute entirely on consumer hardware within a **4.0 GB VRAM limit** (NVIDIA GeForce RTX 3050 Laptop GPU).

---

## Table of Contents
1. [Methodology](#methodology)
2. [Systems and Memory Engineering](#systems-and-memory-engineering)
3. [Dual-Engine Reward Framework](#dual-engine-reward-framework)
4. [Benchmark Evaluation](#benchmark-evaluation)
5. [Inference and Interactive CLI](#inference-and-interactive-cli)
6. [Repository Structure](#repository-structure)
7. [Citation and Attribution](#citation-and-attribution)

---

## Methodology

Language models process text as discrete sub-word tokens rather than individual characters. Because token embeddings do not encode constituent characters directly, standard autoregressive models cannot natively respect character-level constraints. To enforce strict lipogram generation without model collapse, this project utilizes a three-layer post-training architecture:

```mermaid
flowchart LR
    A["User Prompt"] --> B["Qwen3-1.7B (4-bit NF4)"]
    B --> C["Logits: z in R^151936"]
    C --> D["Layer 1: Logit Masking\n(Blocks 103,639 tokens)"]
    D --> E["Layer 2: SFT Adapter\n(Syntax & Synonym Grounding)"]
    E --> F["Layer 3: GRPO RL Adapter\n(Repetition & Fluency Optimization)"]
    F --> G["Valid Lipogram Output"]
```

### Layer 1: Logit Masking (Inference Constraint)
- Pre-computes a static boolean mask across the 151,936-token vocabulary.
- Flags all tokens containing "e", "E", or Unicode accented equivalents via NFKD normalization.
- Mask blocks **103,639 tokens (68.33% of the total vocabulary)** by setting logits to $-\infty$ during sampling.

### Layer 2: Supervised Fine-Tuning (SFT)
- Trains on 180 curated, human-verified e-free instruction-response pairs extracted from *No Robots* and *Databricks Dolly-15k*.
- Re-grounds syntax and shifts probability mass toward valid e-free alternatives, replacing common functional vocabulary (*the*, *be*, *are*, *were*, *have*).

### Layer 3: Group Relative Policy Optimization (GRPO)
- Samples a group of $G = 4$ candidate completions per prompt.
- Computes relative advantages by normalizing reward signals within each group:
  $$\hat{A}_i = \frac{R_i - \text{mean}(R)}{\text{std}(R)}$$
- Eliminates the need for a separate Critic/Value network, enabling policy optimization directly under tight hardware constraints.

---

## Systems and Memory Engineering

Standard 16-bit RL pipelines require 16 to 24 GB of VRAM to host the active actor, frozen reference baseline, critic, and reward models. To train on a single 4.0 GB VRAM GPU, the following systems-level optimizations were implemented:

| Memory Component | Standard BF16 Training | Our 4-bit Dual-Adapter Architecture |
| :--- | :---: | :---: |
| Base Model Parameters | 3.40 GB | **1.14 GB** (NormalFloat4) |
| Active Policy Adapter | 3.40 GB | **0.066 GB** (LoRA $r=16, \alpha=16$) |
| Reference Baseline Adapter | 3.40 GB | **0.066 GB** (Shared 4-bit Base Model) |
| Optimizer States (AdamW) | 6.80 GB | **0.13 GB** (Adapter parameters only) |
| Activation Buffers | 4.00+ GB | **~0.60 GB** (Activation Recomputation) |
| **Peak VRAM Consumption** | **> 17.6 GB (OOM)** | **~3.44 GB (Stable Headroom)** |

- **Dual-Adapter PEFT Architecture:** Rather than loading a separate frozen base model to calculate KL divergence drift, both the trainable policy and frozen reference baseline are instantiated as lightweight LoRA adapters attached to the *same* 4-bit base weights. Switching between adapters during rollouts adds only 66 MB of overhead.
- **Activation Recomputation:** Non-reentrant gradient checkpointing is configured on the quantized model layers (`use_reentrant=False`), eliminating activation memory bottlenecks without numerical instability.

---

## Dual-Engine Reward Framework

Configured via a single parameter in `configs/default.yaml` and `.env`:

```yaml
reward:
  mode: "local"   # "local" for offline heuristics, or "api" for remote LLM judge
```

1. **Local Composite Heuristic (Offline, Deterministic):**
   - **Hard Lipogram Gate:** Any occurrence of "e" or "E" immediately assigns $R = 0.0$.
   - **Trapezoidal Length Shaping:** Rewards concise answers (25 to 80 tokens) and penalizes truncation or verbosity.
   - **N-Gram Entropy Penalty:** Penalizes repetitive loops using unique 2-gram and 3-gram ratios.
   - **Lexicon Verification:** Validates tokens against an 80,549-word English dictionary (`data/dictionary/english_words.txt`).
   - **Topical Alignment:** Scores keyword overlap between prompt and response.

2. **Remote API-as-a-Judge (OpenAI / OpenRouter Compatible):**
   - Compatible with OpenAI, OpenRouter, Groq, or local vLLM/Ollama endpoints.
   - Evaluates Coherence (1-5) and Response Relevance (1-5).
   - Automatically falls back to the Local Composite Heuristic if network timeouts or rate limits occur.

---

## Benchmark Evaluation

Evaluated across standardized benchmark prompts from `data/prompts/short_eval_50.jsonl`:

| Model Architecture | Strict Lipogram Pass | Usable Rate | Mean Reward | Repetition Diversity | Lexicon Validity | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Base Qwen-1.7B + Mask** | 88.0% | **28.0%**\* | 0.6731 | 91.5% | 100.0% | Baseline with foreign evasion |
| **SFT (180 Rewrites) + Mask** | 96.0% | **80.0%** | 0.8436 | 91.5% | 100.0% | Syntax restored (0% evasion) |
| **RL (GRPO Best) + Mask** | **100.0%** | **88.0%** | **0.8895** | **92.5%** | 100.0% | Optimal coherence and diversity |

### Note on Base Model Usable Rate (28.0% vs. 56.0% Raw)
An offline heuristic initially scored the Base Model at 56.0% usable because it only verified the absence of the Latin character "e". Qualitative inspection revealed that an unaligned base model circumvents the English constraint by substituting non-ASCII tokens (Chinese, Russian Cyrillic, Japanese) to satisfy prompt concepts. 

Filtering for pure ASCII English (`response.isascii()`) disqualifies 7 out of 14 raw passes, resulting in a true usable rate of **28.0% (7/25)**, aligning with literature baselines (~26%). Both the SFT and RL models exhibited **0% foreign token evasions**, maintaining 80.0% and 88.0% usability under strict ASCII evaluation.

### Post-Training Progression Analysis
| Stage | Baseline Literature Target | Measured Usable Rate | Progression Status |
| :--- | :---: | :---: | :---: |
| **Base Model + Mask** | 26.0% | **28.0%**\* (56.0% raw) | Verified Baseline |
| **SFT (180 Rewrites) + Mask** | 35.0% | **80.0%** | Substantial Uplift |
| **RL (GRPO) + Mask** | 60.5% | **88.0%** | High Alignment |

---

## Inference and Interactive CLI

### Environment Setup
```bash
conda create -n qwen-no-e python=3.11 -y
conda activate qwen-no-e
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft bitsandbytes accelerate datasets pyyaml
python scripts/verify_env.py
```

### Interactive CLI Chat
Run real-time inference with decoding-time masking and dynamic model switching:
```bash
python scripts/chat_no_e.py
```

Available commands within the terminal session:
- `:mode rl` - Activate the GRPO RL model (default)
- `:mode sft` - Activate the SFT model
- `:mode base` - Activate the Base model with token mask
- `:temp <float>` - Adjust sampling temperature (default: 0.7)
- `:tokens <int>` - Set max generation tokens (default: 60)
- `exit` or `quit` - Terminate session

### Automated Benchmark Suite
```bash
python evaluation/benchmark_suite.py --limit 25
```

---

## Repository Structure

```
qwen-3-1.7b-finetuning/
├── checkpoints/
│   ├── sft_best/                  # Optimal SFT adapter weights (Epoch 3, Val Loss: 2.5692)
│   ├── rl_best/                   # Optimal RL adapter weights (Step 26, Mean Reward: 0.9952)
│   └── rl_step_{10,20,30}/        # Periodic training checkpoints
├── configs/
│   └── default.yaml               # Training, LoRA, and reward engine configuration
├── core/
│   ├── checks.py                  # NFKD lipogram validation routines
│   ├── llm_api.py                 # API judge client with exponential backoff
│   ├── rewards.py                 # Dual-engine reward coordinator
│   └── sampling.py                # Logit mask constructor (103,639 tokens blocked)
├── data/
│   ├── dictionary/                # Curated 80,549-word English dictionary
│   ├── prompts/                   # Benchmark prompt sets (short_eval_50.jsonl)
│   └── rewrites/splits/           # 180 validated e-free instruction pairs
├── docs/
│   ├── ARCHITECTURE.md            # Systems and module design documentation
│   ├── EXPERIMENT_LOG.md          # Experiment tracking and metric logs
│   ├── LEARNING_GUIDE.md          # Technical analysis of post-training dynamics
│   ├── MEMORY.md                  # Project context document
│   └── PLAN.md                    # Milestone roadmap
├── evaluation/
│   ├── benchmark_suite.py         # Multi-model evaluation harness
│   └── reports/                   # Output metrics and comparison reports
├── scripts/
│   ├── chat_no_e.py               # Interactive CLI chat interface
│   ├── test_trained_rl.py         # Comparative generation inspection script
│   ├── test_trained_sft.py        # SFT verification script
│   └── verify_env.py              # Environment diagnostic tool
├── training/
│   ├── train_sft_4bit.py          # 4-bit QLoRA SFT training implementation
│   └── train_grpo_4bit.py         # 4-bit Dual-Adapter GRPO implementation
├── weights/
│   └── qwen3-1.7b/                # Base model weights and tokenizer
├── .env.example                   # Environment configuration template
└── README.md                      # Project documentation
```

---

## Citation and Attribution

- **Lead Engineer & Author:** Prateek
- **Base Model:** [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) by Alibaba Cloud / Qwen Team
- **License:** Apache 2.0
