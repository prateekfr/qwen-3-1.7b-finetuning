# Qwen, Minus the E 🚫📧

Fine-tuning [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) to write coherent English text that **never uses the letter "e"** — a literary constraint called a [lipogram](https://en.wikipedia.org/wiki/Lipogram).

Inspired by [Kevin Wu's original project](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj).

## How It Works

LLMs process text as sub-word **tokens**, not individual characters. They literally cannot "see" what letters are inside each token. Teaching a model to avoid a specific letter requires a three-layer approach:

1. **Token Mask** — At decoding time, every token containing `e`/`E`/`é` gets its logit set to `-∞`. This guarantees zero "e"s but alone produces gibberish.
2. **Supervised Fine-Tuning (SFT)** — Train on 180 curated e-free chat examples so the model learns to form coherent sentences without common words like *the*, *he*, *she*, *are*.
3. **Reinforcement Learning (GRPO)** — Further optimize with an LLM judge reward for coherence and prompt-following.

## Adaptations

This replication runs on a **4 GB VRAM GPU** (RTX 3050 Laptop) vs. the original's 16 GB, using:
- **QLoRA (4-bit NF4 quantization)** instead of bf16
- **Batch size 1** with gradient accumulation
- **Gradient checkpointing** for memory savings

## Project Structure

```
├── docs/               # Project documentation & memory
│   ├── MEMORY.md       # Master context (read this first in new sessions)
│   ├── LEARNING_GUIDE.md # 🎓 In-depth AI Engineering textbook & key terms
│   ├── ARCHITECTURE.md # Technical architecture
│   ├── PLAN.md         # Execution plan with status
│   ├── EXPERIMENT_LOG.md # Running experiment results
│   └── HARDWARE.md     # Hardware constraints & VRAM budget
├── core/               # Core utilities
│   ├── mask.py         # Forbidden token mask builder
│   └── generation.py   # Custom rollout generation
├── data/               # Data pipeline
│   └── prompts/        # Dataset curation scripts
├── training/           # Training loops
│   ├── engine.py       # Custom training engine
│   ├── sft.py          # Supervised fine-tuning
│   └── rl/grpo.py      # GRPO reinforcement learning
├── evaluation/         # Evaluation scripts
├── configs/            # Hyperparameter configs
├── checkpoints/        # Saved model weights
└── logs/               # Training logs
```

## Setup

```bash
conda create -n qwen-no-e python=3.11 -y
conda activate qwen-no-e
pip install -r requirements.txt
python scripts/verify_env.py
```

## References

- [Original Project (Kevin Wu)](https://github.com/kevinyvv/qwen-3-1.7b-finetune-proj)
- [Blog Post](https://kevin-wu.ca)
- [Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B)
- [Kevin's Final Model](https://huggingface.co/k33wu/qwen3-1.7b-no-e)
- [Kevin's SFT Model](https://huggingface.co/k33wu/qwen3-1.7b-no-e-sft180)

## License

MIT
