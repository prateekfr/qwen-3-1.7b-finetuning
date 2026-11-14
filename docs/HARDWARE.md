# 🖥️ Hardware Analysis — Qwen, Minus the E

> **Purpose:** Detailed hardware constraints, VRAM budget, and memory management
> strategies for running this project on 4 GB VRAM.
>
> **Last Updated:** 2026-11-14

---

## 1. Full System Specs

| Component   | Specification                              |
| ----------- | ------------------------------------------ |
| CPU         | AMD Ryzen 7 4800H (8C/16T, 2.9-4.2 GHz)   |
| RAM         | 8 GB DDR4                                  |
| GPU         | NVIDIA GeForce RTX 3050 Laptop (4 GB GDDR6)|
| GPU Arch    | Ampere (GA107, Compute 8.6)                |
| CUDA        | 12.9                                       |
| Driver      | 576.88                                     |
| TDP         | 35W (laptop variant)                       |
| Disk        | D: Crucial NVMe — 931.5 GB (252 GB free)   |
| OS          | Windows                                    |

### GPU Capabilities
- ✅ bf16 support (Ampere architecture)
- ✅ tf32 support
- ✅ INT8 / INT4 compute
- ✅ CUDA 12.x compatible
- ⚠️ Only 4 GB VRAM (vs. typical ML GPUs: 16-80 GB)
- ⚠️ 128-bit memory bus (lower bandwidth than desktop)

---

## 2. VRAM Budget Analysis

### Model Memory (Static)

| Component                        | bf16     | 4-bit NF4 |
| -------------------------------- | -------- | ---------- |
| Qwen3-1.7B base weights         | ~3.4 GB  | ~1.1 GB    |
| LoRA adapters (17.4M, float32)   | ~0.07 GB | ~0.07 GB   |
| Optimizer states (AdamW, fp32)   | ~0.14 GB | ~0.14 GB   |
| **Subtotal (static)**            | ~3.6 GB  | ~1.3 GB    |

### Training Memory (Dynamic — per step)

| Component                        | Batch=4   | Batch=1 + GradCkpt |
| -------------------------------- | --------- | ------------------- |
| Activations                      | ~4-6 GB   | ~1.0-1.5 GB        |
| Gradients                        | ~0.07 GB  | ~0.07 GB            |
| **Subtotal (dynamic)**           | ~4-6 GB   | ~1.0-1.5 GB        |

### Generation Memory (during RL rollouts)

| Component                        | Estimated  |
| -------------------------------- | ---------- |
| KV cache (1 sequence, 300 tokens)| ~0.3-0.5 GB|
| Logits buffer                    | ~0.02 GB   |
| **Subtotal (generation)**        | ~0.3-0.5 GB|

### Total Budget

| Scenario              | Estimated Total | Fits in 4 GB? |
| --------------------- | --------------- | ------------- |
| bf16, batch=4         | ~7.6-9.6 GB     | ❌ No         |
| bf16, batch=1         | ~4.6-5.5 GB     | ❌ No         |
| 4-bit, batch=1, no GC | ~3.3-3.8 GB    | ⚠️ Tight      |
| **4-bit, batch=1, GC**| **~2.3-2.8 GB** | **✅ Yes**    |
| 4-bit + RL generation | ~2.6-3.3 GB     | ⚠️ Tight      |

> GC = Gradient Checkpointing

---

## 3. Memory Management Strategies

### Must-Have
1. **4-bit NF4 quantization** (bitsandbytes) — saves ~2.3 GB
2. **Gradient checkpointing** — trades compute for memory, saves ~0.5-1 GB
3. **Batch size 1** with gradient accumulation (4 steps)
4. **Close all other GPU-using apps** before training

### Should-Have
5. **`torch.cuda.empty_cache()`** between generation and training in RL
6. **Separate generation and training** phases in RL (don't keep both in memory)
7. **CPU offload** for optimizer states if needed (accelerate)
8. **Memory-mapped datasets** to keep system RAM usage low

### Nice-to-Have
9. **Mixed precision** for LoRA adapters (fp16 instead of fp32)
10. **Reduce LoRA rank** to 8 if 16 doesn't fit
11. **Shorter generation** (200 tokens instead of 300) in RL

---

## 4. System RAM Considerations

With only 8 GB system RAM:
- Windows OS uses ~2-3 GB
- Python process + dataset loading: ~1-2 GB
- HuggingFace model loading buffer: ~1-2 GB
- **Available for training: ~3-4 GB**

**Mitigations:**
- Close browser and other apps during training
- Use streaming/memory-mapped datasets
- Load model directly to GPU (don't keep CPU copy)
- Set `low_cpu_mem_usage=True` when loading model

---

## 5. Thermal Considerations

The RTX 3050 Laptop has a 35W TDP and will thermal throttle under sustained load:
- Monitor GPU temp during training (`nvidia-smi`)
- If >85°C, consider pausing between epochs
- Use a laptop cooling pad if available
- Training will be slower than a desktop GPU

---

## 6. Expected Training Times (Rough Estimates)

| Phase                  | Kevin's (16GB GPU) | Ours (4GB, 4-bit) |
| ---------------------- | ------------------- | ------------------- |
| SFT (10 epochs)        | ~30-60 min          | ~1-3 hours          |
| RL (300 steps)         | ~2-4 hours          | ~4-10 hours         |
| Full pipeline          | ~3-5 hours          | ~5-13 hours         |

> These are rough estimates. Actual times depend on many factors.
> 4-bit quantized models have overhead from dequantization during forward pass.
