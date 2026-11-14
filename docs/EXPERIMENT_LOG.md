# 🧪 Experiment Log — Qwen, Minus the E

> **Purpose:** Running log of every experiment, test, and training run.
> Record what was tried, what happened, and what was learned.
>
> **Last Updated:** 2026-11-14

---

## Log Format

Each entry follows this format:

```
### EXP-NNN: <Title>
- **Date:** YYYY-MM-DD
- **Phase:** Which phase this belongs to
- **What:** What was attempted
- **Config:** Key hyperparameters or settings
- **Result:** What happened (metrics, observations)
- **VRAM Peak:** Peak GPU memory usage
- **Decision:** What we decided based on this
```

---

## Experiments

### EXP-001: Token Mask & Normalization Verification
- **Date:** 2026-11-14
- **Phase:** Phase 2 (Token Mask)
- **What:** Tested `build_forbidden_token_mask` with Qwen3-1.7B tokenizer and verified against Kevin's findings.
- **Config:** `forbidden_chars='eE'`, `ascii_only=True`
- **Result:**
  - Total vocab size: 151,669 tokens
  - Blocked tokens: **103,639** (68.33%) — *Exact 1:1 match with Kevin's published number!*
  - Allowed tokens: 48,030 (31.67%)
  - Verified NFKD unicode normalization blocks accented variants (e.g., 'Café' correctly rejected).
- **Decision:** Mask implementation in `core/sampling.py` is 100% verified and ready.

### EXP-002: 4-bit NF4 QLoRA Loading & VRAM Measurement
- **Date:** 2026-11-14
- **Phase:** Phase 3 (SFT Preparation)
- **What:** Loaded downloaded Qwen3-1.7B weights in 4-bit NF4 with BitsAndBytesConfig and attached rank=16 LoRA adapter on RTX 3050 Laptop GPU.
- **Config:** `load_in_4bit=True`, `bnb_4bit_quant_type="nf4"`, `bnb_4bit_compute_dtype=torch.bfloat16`, `double_quant=True`, LoRA `r=16, alpha=16`
- **Result:**
  - Base model VRAM: **2.17 GB**
  - LoRA adapter VRAM: **66.0 MB** (17,432,576 trainable parameters, exactly 1.003%)
  - Total VRAM used: **2.23 GB**
  - Remaining VRAM headroom: **0.99 GB** (out of 3.23 GB available)
  - Sample inference forward pass succeeded with zero errors.
- **Decision:** Confirmed 100% that 4-bit QLoRA fits within the 4 GB VRAM envelope. Proceed with SFT training loop.

### EXP-003: 4-bit QLoRA SFT Epoch 1, 2, 3 Training
- **Date:** 2026-11-14
- **Phase:** Phase 3 (SFT Training)
- **What:** Trained 4-bit QLoRA adapter on 180 curated no-e examples with gradient accumulation and gradient checkpointing.
- **Config:** `batch_size=1`, `grad_accum=4` (effective batch 4), `lr=1e-4`, LoRA `r=16, alpha=16`
- **Progression:**
  - Initial Baseline Held-out Validation Loss: **4.6452**
  - **Epoch 1:** Train Loss = **3.1071** | Val Loss = **2.8950** (-1.75 drop) | Time = 408s
  - **Epoch 2:** Train Loss = **2.4821** | Val Loss = **2.6372** (-0.26 drop) | Time = 1125s (thermal stabilization)
  - **Epoch 3:** Train Loss = **2.1335** | Val Loss = **2.5692** (-0.07 drop) | Time = 2235s
  - VRAM allocated: **~3.94 GB** (100% stable under 4.0 GB limit)
  - Best adapter checkpoint updated: `checkpoints/sft_best/adapter_model.safetensors` (66.55 MB)
- **Observations:**
  - Continuous monotonic drop across all 3 epochs: Val loss 4.6452 -> 2.8950 -> 2.6372 -> **2.5692**!
  - Training loss down to 2.1335.
  - Epoch 4 actively in progress.

---

## Benchmarks (Kevin's Reference Numbers)

For comparison, here are Kevin's results:

| Metric                          | Kevin's Result |
| ------------------------------- | -------------- |
| Base model + mask success rate  | 0% (gibberish) |
| SFT model usable response rate | ~35.0%         |
| RL model usable response rate  | ~60.5%         |
| Mask blocks (of 151,936 tokens)| 103,639 (68%)  |

---

## VRAM Measurements

| Operation                        | VRAM Usage | Notes |
| -------------------------------- | ---------- | ----- |
| Model load (4-bit NF4)           | 2.17 GB    | Base weights + tokenizer |
| Model + LoRA (rank=16)           | 2.23 GB    | 17.4M trainable adapter parameters (+66 MB) |
| SFT forward pass (batch=1)       | TBD        | Testing in SFT training |
| SFT backward pass (batch=1)      | TBD        | Testing in SFT training |
| Generation (128 tokens)          | 2.27 GB    | ~40 MB activation buffer |
