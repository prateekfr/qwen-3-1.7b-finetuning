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
| Model load (4-bit NF4)           | TBD        |       |
| Model + LoRA                     | TBD        |       |
| SFT forward pass (batch=1)       | TBD        |       |
| SFT backward pass (batch=1)      | TBD        |       |
| Generation (128 tokens)          | TBD        |       |
| Generation (300 tokens)          | TBD        |       |
| RL full step                     | TBD        |       |
