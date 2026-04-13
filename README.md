# GPU Arithmetic Emulator: Code Reference

This document explains what each emulator component does and why, without investigation history. For the full story of how each behavior was discovered, see `gpu_rounding_prediction_writeup.md`.

## Overview

The emulator predicts every BF16 bit of an A100 GPU's inference output for a transformer forward pass, running entirely on CPU. It achieves 0 BF16 diffs on both the FFN block and the attention block of Qwen3-4B (layer 20, validated at 500 and 4,000 tokens — 0/16,384,000 elements at the longer length).

The GPU is fully deterministic. The apparent "hardware noise" is undocumented determinism from three sources: tensor core block FMA arithmetic, MUFU special-function approximations, and compiler-induced FMA fusion. All three are characterizable.

## Tensor Core Matmul (`tc_emulator.py`, `tc_profiles.py`)

**What it does.** Emulates the A100's `mma.sync.aligned.m16n8k16` instruction for BF16 inputs with FP32 accumulation.

**Why it's not just multiply-and-add.** The tensor core doesn't compute individual FP32 products. It processes blocks of 8 input pairs at once in a fixed-point accumulator:

1. Multiply BF16 significands as integers, sum exponents. Keep the product **denormalized** (not rounded to FP32).
2. Find the max exponent among the 8 products + the FP32 accumulator.
3. Align all 9 values into a 26-bit fixed-point window (24 FP32 bits + 1 extra alignment bit + 1 integer bit for products ≥ 2.0).
4. Sum as integers. **Truncate** (not round-to-nearest) to FP32.

**Parameters** (A100, BF16→FP32): NFMA=8, neab=1, truncation, denormalized products. Two block FMA invocations per k=16 MMA step.

**`matmul(A, B)`** — Full matmul with sequential K-walk. Returns raw FP32 accumulator.

**`block_fma_batch(acc, A_block, B_block)`** — Single block FMA applied to an existing accumulator. Used for FA2's PV matmul, which accumulates across KV tiles into the same register (not starting from zero each time).

**Compiler flags.** The C emulator is compiled with `gcc -ffp-contract=off -fno-fast-math`. Without `-ffp-contract=off`, gcc fuses multiply+add into FMA, producing different rounding than the separate alignment-and-truncate the tensor core performs.

## MUFU Hardware Instructions (`mufu_emulator.py`)

Three special-function instructions used in transformer inference, each a hardware lookup+interpolation unit that differs from IEEE arithmetic:

**MUFU.RSQ** (reciprocal square root) — Used in RMSNorm. Differs from IEEE `1/sqrt(x)` by 0–2 ULP depending on mantissa and exponent parity. Probed exhaustively: 2×2²³ values (mantissa × exponent parity), cached as two 8MB correction tables per GPU. One-time ~30s probe.

**MUFU.EX2** (base-2 exponential) — Used in FlashAttention's softmax (`exp2f` under `--use_fast_math`). Output depends on the full 32-bit input pattern, not just the fractional part. Full 4-billion-entry correction table (4.3GB), indexed by raw uint32 bits. Probed via inline PTX `ex2.approx.ftz.f32`.

**MUFU.RCP** (reciprocal) — Used in FlashAttention for `1.0f/sum` (under `--use_fast_math`, `1.0f/x` compiles to `rcp.approx.ftz.f32`). Differs from IEEE division on 13.2% of inputs. Correction table indexed by 2²³ mantissa values. Must be probed with actual PTX, not `1.0f/x` in C (which gives IEEE division).

## FFN Chain (`ffn_chain_test.py`)

Emulates: residual → RMSNorm → gate_proj → SiLU → up_proj → SiLU(gate)×up → down_proj → residual add.

### RMSNorm

Four behaviors to match:

1. **Reduction tree.** PyTorch's `reduce_kernel` assigns 32 threads per row, each loading 80 elements via 4-wide vectorized reads into 4 accumulators, combined left-to-right, then warp-shuffled with decreasing offsets (16,8,4,2,1). Emulated in `emulate_pytorch_reduce.py`.

2. **Variance division.** nvcc replaces `acc / 2560.0f` with `acc * (1.0f/2560.0f)`. The precomputed reciprocal has rounding, so multiply-by-reciprocal ≠ true division on ~20% of inputs.

3. **MUFU.RSQ.** `torch.rsqrt` compiles to bare `MUFU.RSQ` on SM 8.0 (no Newton-Raphson refinement). Uses probed correction table.

4. **Cast ordering.** Normalize in FP32, cast to BF16, then weight multiply in BF16. Not: all-FP32-then-cast (wrong), or all-BF16 (wrong).

### Matmul Projections

`tc_emulator.matmul()` for gate, up, and down projections. Sequential K-walk matching CUTLASS {128,128,64}/{64,64,64}/{16,8,16}.

### SiLU

Standard FP32 sigmoid × x. CPU numpy matches GPU at BF16 with no special handling.

## Attention Chain (`attn_chain_test.py`)

Emulates: residual → RMSNorm → Q/K/V projection → QK-norm → RoPE → FlashAttention-2 core → O projection → residual add.

### Pre-attention (RMSNorm, projections, QK-norm, RoPE)

Same components as FFN chain. QK-norm is per-head RMSNorm (head_dim=128).

**RoPE** requires matching the GPU's precision path at three points:

1. `inv_freq` must be computed with torch (not numpy) — `base ** (int64_arange.float() / dim)` rounds differently from numpy's exponentiation on some entries.
2. `cosf`/`sinf` must use GPU CUDA libm (not CPU glibc). These are software library functions with different polynomial approximations, differing by 1 ULP on ~0.001% of inputs. Cached from GPU computation (~2MB), or could be probed into exhaustive correction tables (~16GB).
3. The rotation arithmetic itself (`q * cos + rotate_half(q) * sin`) is element-wise BF16 multiply-and-add, matching between CPU and GPU.

**BF16 memory boundaries.** Between pipeline stages, the GPU stores tensors as BF16 in global memory and reloads them. The emulator must snap to BF16 (`to_bf16_f32()`) at every such boundary: after QK-norm, after RoPE, before FA2. Without this, sub-BF16 FP32 differences in the emulator's intermediates get extracted as different BF16 significands by the tensor core, amplifying across many KV tiles. This was invisible at seq_len=500 but caused 9,708 diffs at seq_len=4,000 before the fix.

### FlashAttention-2 Core

The FA2 kernel fuses QK^T matmul, online softmax, and PV matmul into a single kernel with kBlockM=128, kBlockN=64, causal masking, reverse tile iteration.

**Tile structure.** For each m_block of 128 query rows, iterate KV tiles in reverse (n_block = max-1 down to 0). First tile uses `Is_first` path (initialize max/sum). Subsequent tiles rescale running state.

**QK^T matmul.** `tc_emulator.matmul(Q_tile, K_tile.T)` — standard sequential K-walk, same as CUTLASS.

**Online softmax.** Per-thread computation (4 threads per row, 16 columns each, pairs pattern: thread 0 owns columns {0,1,8,9,16,17,...,56,57}):

- `row_max`: thread-local max, then `Allreduce<4>` (shfl_xor(2) then shfl_xor(1))
- `scale_apply_exp2`: `exp2f(S * scale_log2 - max_scaled)` via FFMA + MUFU.EX2
- `reduce_sum`: thread-local sequential accumulation, NO quad allreduce (deferred)

**Rescale between tiles.** When a new tile has a larger max:
```
rescale = MUFU.EX2((prev_max - cur_max) * scale_log2)
O_acc *= rescale          // separate FMUL
row_sum = FMA(row_sum, rescale, exp2[0])  // FMUL+FADD fused into FFMA
row_sum += exp2[1]        // FADD
row_sum += exp2[2]        // FADD
...
```

The **FMA fusion** on the row_sum rescale is critical. Under `--use_fast_math` with `--fmad=true`, the compiler sees `row_sum *= rescale` (in `softmax_rescale_o`) followed by `row_sum += exp2[0]` (in `thread_reduce_`, called via `reduce_sum`, all `__forceinline__`) and fuses them into a single FFMA. FMA rounds once instead of twice, giving a different FP32 result 23.2% of the time. The emulator matches this by computing via float64 (exact product + sum → single rounding to float32).

Note: O_acc rescale is NOT fused — it's followed by the entire `scale_apply_exp2` loop before PV matmul, so the compiler doesn't fuse it with anything.

**PV matmul.** `tc_emulator.block_fma_batch(O_acc, P_bf16, V_tile)` — accumulates into the live O_acc register, not from zero. The block FMA alignment window depends on the accumulator magnitude, so `matmul(P,V) + O_acc` ≠ `accumulate_into(O_acc, P, V)` at FP32.

**Final normalization.**
```
total_sum = (row_sum[0] + row_sum[2]) + (row_sum[1] + row_sum[3])  // Allreduce<4>
inv_sum = MUFU.RCP(total_sum)
output = O_acc * inv_sum  // FMUL, then cast to BF16
```

### O Projection

`tc_emulator.matmul()`, same as other projections.

## File Summary

| File | What |
|------|------|
| `tc_profiles.py` | Tensor core parameters per GPU (NFMA, neab, rounding, MMA shape) |
| `tc_emulator.py` | Block FMA matmul emulator (C + OpenMP, ctypes) |
| `mufu_emulator.py` | MUFU.RSQ/EX2/RCP emulator (probed correction tables) |
| `emulate_pytorch_reduce.py` | PyTorch reduction tree emulator |
| `ffn_chain_test.py` | FFN block: capture, emulate, compare |
| `attn_chain_test.py` | Attention block: capture, emulate, compare |
| `cutlass_gemm_flex` | CUTLASS binary for ground-truth matmul |

## Key Principle

Every emulator choice matches a specific GPU behavior:

| Emulator does this | Because the GPU does this |
|---|---|
| Block FMA with 26-bit window, truncation | Tensor core fixed-point accumulator |
| Denormalized products in alignment | Raw significand multiply, no normalize-to-FP32 |
| `block_fma_batch` with live accumulator | FA2's `gemm_rs` passes `acc_o` as MMA C operand |
| MUFU correction tables | Hardware lookup ≠ IEEE arithmetic |
| Full 4B-entry EX2 table | Output depends on all 32 input bits |
| FMA for rescale + first add | `--fmad=true` fuses across `__forceinline__` boundaries |
| Multiply-by-reciprocal for variance | nvcc constant-division optimization |
| Parallel reduction tree for sum-of-squares | PyTorch `reduce_kernel` warp-shuffle pattern |
| `gcc -ffp-contract=off` | Prevent C compiler from doing the same FMA fusion |
| `to_bf16_f32()` at memory boundaries | GPU stores/loads as BF16 between pipeline stages |
| GPU-cached cos/sin for RoPE | CUDA libm's `cosf`/`sinf` ≠ glibc's `cosf`/`sinf` |
| `inv_freq` via torch, not numpy | Different exponentiation rounding in math libraries |
