# 📋 Execution Plan — Qwen, Minus the E

> **Purpose:** Phase-by-phase execution plan with status tracking.
> Update checkboxes and dates as work progresses.
>
> **Last Updated:** 2026-11-14

---

## Phase 0: Environment Setup
**Status:** ✅ Completed
**Target:** Day 1

- [x] Create conda environment (`qwen-no-e`, Python 3.11)
- [x] Install PyTorch with CUDA 12.x
- [x] Install core dependencies (transformers, peft, bitsandbytes, accelerate, datasets)
- [x] Run `scripts/verify_env.py` to confirm GPU, CUDA, and all packages
- [ ] Test loading Qwen3-1.7B in 4-bit mode (fits in 4 GB?)
- [ ] Measure actual VRAM usage of quantized model

**Exit Criteria:** `verify_env.py` passes all checks, model loads in <2 GB VRAM.

---

## Phase 1: Data Pipeline
**Status:** ✅ Completed
**Target:** Day 1-2

- [x] Implement `data/prompts/make_prompt_data.py`
- [x] Download No Robots dataset from HuggingFace
- [x] Download Dolly-15k dataset from HuggingFace
- [x] Filter for suitable instruction-response pairs
- [x] Rewrite/curate responses to remove all "e"s (180 examples)
- [x] Create train/validation split (e.g., 160/20 or train_180 + val)
- [x] Save processed data to `data/rewrites/splits/`
- [x] Validate: every response has zero "e"s

**Exit Criteria:** 180 validated e-free examples saved, train/val split ready.

---

## Phase 2: Token Mask Implementation
**Status:** ✅ Completed
**Target:** Day 2

- [x] Implement `core/sampling.py` — scan vocabulary for e-containing tokens
- [x] Test token mask builder against Qwen3 tokenizer
- [x] Measure mask statistics (103,639 / 151,669 tokens blocked, matching Kevin's count exactly)
- [x] Verify mask handles edge cases (accented chars like café via NFKD normalization)

**Exit Criteria:** Token mask confirmed to block exactly 103,639 tokens (68.33%).

---

## Phase 3: Supervised Fine-Tuning (SFT) ⬅️ CURRENT
**Status:** 🔄 In Progress
**Target:** Day 2-3

- [ ] Implement `training/engine.py` — custom training engine
- [ ] Implement `training/sft.py` — SFT training script
- [ ] Load Qwen3-1.7B in 4-bit NF4 with LoRA (rank=16, α=16)
- [ ] Configure: batch=1, grad_accum=4, grad_checkpointing=True
- [ ] Train 10 epochs, LR=1e-4
- [ ] Track train/val loss per epoch
- [ ] Save best checkpoint (lowest validation loss)
- [ ] Test SFT model + mask for coherence
- [ ] Log results in `docs/EXPERIMENT_LOG.md`

**Exit Criteria:** SFT model produces somewhat coherent e-free responses.
Target: ~35% usable response rate (matching Kevin's SFT baseline).

---

## Phase 4: Reinforcement Learning (GRPO)
**Status:** ⏳ Not Started
**Target:** Day 3-5

- [ ] Implement reward function (rule-based or LLM judge)
- [ ] Implement `training/rl/grpo.py` — GRPO training loop
- [ ] Configure: 300 steps, 4 prompts × 4 answers, LR=2e-5, KL=0.02
- [ ] Test VRAM usage during rollout generation
- [ ] If OOM: reduce answers per prompt, reduce max gen length
- [ ] Run full RL training
- [ ] Monitor for reward hacking / length shrinking
- [ ] Log results in `docs/EXPERIMENT_LOG.md`

**Exit Criteria:** Measurable improvement over SFT baseline.
Target: >50% usable response rate.

---

## Phase 5: Evaluation & Publishing
**Status:** ⏳ Not Started
**Target:** Day 5-6

- [ ] Implement `evaluation/evaluate.py`
- [ ] Run full evaluation on test set
- [ ] Compare results with Kevin's benchmarks
- [ ] Upload LoRA adapter to HuggingFace
- [ ] Write comprehensive README.md
- [ ] Initialize git repo, make clean commits
- [ ] Push to GitHub

**Exit Criteria:** Project published on GitHub with README, model on HuggingFace.

---

## Phase 6: Stretch Goals
**Status:** ⏳ Not Started

- [ ] Try different forbidden letters (a, t, s — progressively harder)
- [ ] Add interactive demo (Gradio)
- [ ] Experiment with different LoRA ranks
- [ ] Write blog post about the experience

---

## Timeline Summary

| Phase | Description             | Days  | Status         |
| ----- | ----------------------- | ----- | -------------- |
| 0     | Environment Setup       | Day 1 | ✅ Completed   |
| 1     | Data Pipeline           | Day 1-2 | ✅ Completed   |
| 2     | Token Mask              | Day 2   | ✅ Completed   |
| 3     | SFT Training            | Day 2-3 | 🔄 In Progress |
| 4     | RL / GRPO Training      | Day 3-5 | ⏳ Not Started |
| 5     | Eval & Publishing       | Day 5-6 | ⏳ Not Started |
| 6     | Stretch Goals           | Bonus   | ⏳ Not Started |
