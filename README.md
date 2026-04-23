# GPU Arithmetic Emulator

A proof-of-concept demonstrating that GPU floating-point inference is fully deterministic and predictable in software. Given knowledge of the hardware's tensor core arithmetic and the software stack's kernel behavior, a CPU-only emulator can reproduce every output bit of a transformer forward pass — not approximately, but exactly.

This is research code, not a polished product. It is optimized for correctness and auditability, not for efficiency, modularity, or ease of use. The primary audience is researchers interested in GPU floating-point determinism, AI governance, or reproducible inference.

For a detailed narrative of how the emulator was developed and what was learned at each step, see `gpu_rounding_prediction_writeup.md`.

## Results

The emulator achieves 0 BF16 diffs on a full FFN block (RMSNorm + 3 matmul projections + SiLU + residual add) of Qwen3-4B layer 20, validated across three NVIDIA GPU generations and a sequence-length sweep up to 8000 tokens (0 FP32 raw-accumulator diffs confirmed on A100):

| GPU | Architecture | Tensor Core | NFMA | neab | BF16 diffs |
|-----|-------------|-------------|------|------|------------|
| A100 | Ampere (sm_80) | CoFDA(F=24) | 8 | 1 | 0 |
| L40S | Ada Lovelace (sm_89) | CoFDA(F=24) | 8 | 1 | 0 |
| H100 | Hopper (sm_90) | FDA(F=25) | 16 | 2 | 0 |

On A100, the **attention chain** (FlashAttention 2.8.3) is validated at 0 diffs at seq_len=500 and seq_len=4000, and the **full 36-layer forward pass** of Qwen3-4B (attention + FFN + LM head) is validated at 0 diffs on 32 and 250 real tokens — every logit bit exact (37.98M logit values at seq=250).

### Two emulator targets

`ffn_chain_test.py` supports two matmul-kernel targets:

- **`EMULATOR_TARGET=cutlass`** (default): emulator matches CUTLASS. Validated bit-exact on A100, L40S, H100.
- **`EMULATOR_TARGET=cublas`**: emulator matches the actual library dispatch (`torch.matmul` → cuBLAS) via a pre-built dispatch catalog and recipe library (see `cublas_recipes.py`, `build_catalog.py`). Validated at 0 BF16 diffs on A100, L40S, H100 — including a confidential-computing H100 pod — at seq_len ∈ {100, 250, 1000, 4000}. One exception: on H100 at seq=100 the `down_proj` shape dispatches to NVIDIA's proprietary `nvjet_*` kernel family, which the recipe library does not yet emulate (see Limitations).

## How to Use

**Requirements:** An NVIDIA GPU, Python 3.10+, PyTorch with CUDA, numpy. CUTLASS headers for the ground-truth matmul binary (optional but recommended for three-way comparison).

**Run the FFN chain test (end-to-end):**

```bash
# 1. Compile the CUTLASS ground-truth binary (adjust -arch for your GPU)
git clone --depth 1 https://github.com/NVIDIA/cutlass.git /workspace/cutlass
nvcc -o cutlass_gemm_flex cutlass_gemm_flex.cu \
    -I /workspace/cutlass/include \
    -I /workspace/cutlass/tools/util/include \
    -arch=sm_80 -std=c++17 -O2

# 2. Run all three phases: model capture → CUTLASS → emulator comparison
python3 ffn_chain_test.py all 256
```

This loads Qwen3-4B, captures layer-20 intermediate tensors on the GPU, runs CUTLASS matmuls for ground truth, then runs the CPU emulator and compares every intermediate stage. MUFU correction tables are probed automatically on first run (~60s for the EX2 table). The GPU is auto-detected via `tc_profiles.detect_gpu()`.

**Run the attention chain test (A100 only, requires FlashAttention 2):**

```bash
python3 attn_chain_test.py all 256
```

**Run on a new GPU:** No code changes needed. The emulator auto-detects the GPU, selects the matching profile from `tc_profiles.py`, and re-probes the MUFU correction tables. Profiles are registered for V100, A100, A2, A30, L40S, L40, Ada RTX 1000, H100, H200, and B200.

**Target cuBLAS instead of CUTLASS (optional):**

```bash
# 1. Compile the cuBLASLt dispatch inspector
nvcc -o cublaslt_inspect cublaslt_inspect.cu -lcublasLt -std=c++17 -O2

# 2. Build a dispatch catalog for the seq lengths you want to validate
python3 build_catalog.py --seq-lens 100 250 1000 4000 --no-verify

# 3. Run Phase 3 in cuBLAS-target mode
EMULATOR_TARGET=cublas python3 ffn_chain_test.py all 4000
```

The catalog is a static JSON artifact; once built it drives CPU-only emulation via `catalog_lookup.py` with no further GPU access. See "Dispatch catalog and recipe library" below for what gets characterized and what doesn't.

The GPU is fully deterministic. The apparent "hardware noise" is undocumented determinism from three sources: tensor core block FMA arithmetic, MUFU special-function approximations, and compiler-induced FMA fusion. All three are characterizable.

## Tensor Core Matmul (`tc_emulator.py`, `tc_profiles.py`)

**What it does.** Emulates the `mma.sync.aligned.m16n8k16` instruction for BF16 inputs with FP32 accumulation. The emulator is parameterized by a per-GPU profile — no code changes are needed for new architectures, only a profile entry in `tc_profiles.py` and MUFU re-probing.

**Why it's not just multiply-and-add.** The tensor core doesn't compute individual FP32 products. It processes blocks of NFMA input pairs at once in a fixed-point accumulator:

1. Multiply BF16 significands as integers, sum exponents. Keep the product **denormalized** (not rounded to FP32).
2. Find the max exponent among the NFMA products + the FP32 accumulator.
3. Shift all significands left by `neab` bits, then align to max exponent within a fixed-point window of `(2 + 23 + neab)` bits. Truncate bits that fall below the window.
4. Sum as integers. **Truncate** (not round-to-nearest) to FP32.

The `neab` left-shift is critical: the hardware adds `neab` extra precision bits to the bottom of every significand *before* alignment, not just after. This is a single line in K&M's MATLAB model (`sigVals = bitshift(sigVals, neab)`) but affects every value in the accumulator. Getting this wrong produces systematic 1-bit errors that compound across K-steps.

**Ampere/Ada vs Hopper.** On A100 and L40, the K=16 MMA step is computed as a **chain of two FDA operations** with NFMA=8 each (CoFDA): the first 8 products are accumulated and normalized to FP32, then the result feeds as the accumulator for the second 8 products. On H100, the same instruction computes all 16 products in a **single FDA** with NFMA=16 and a wider alignment window (neab=2). This distinction is documented by Xie et al. (MMA-Sim, arXiv:2511.10909) as "CoFDA vs FDA" and confirmed by Khattak & Mikaitis (arXiv:2512.07004).

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

## Dispatch Catalog and Recipe Library (`cublas_recipes.py`, `build_catalog.py`, `catalog_lookup.py`)

The CUTLASS-target story works because CUTLASS is open source: the tile configuration, K-iteration order, and reduction scheme are all readable. cuBLAS is not open source, so for `EMULATOR_TARGET=cublas` we characterize the dispatch empirically and model the accumulation order per kernel family.

**Recipe library** (`cublas_recipes.py`). Four bit-accurate kernel models covering the cuBLAS paths we've observed on Ampere/Ada/Hopper:

- `single_walk` — one sequential K-walk. Matches kernels with split_k=1 and no intra-CTA K partitioning.
- `split_k_cutlass_bf16_out` — stock CUTLASS kGemm Split-K with a BF16 round-trip per partition (semaphore-serial, `INPLACE_ATOMIC` reduction scheme).
- `split_k_sliced_kernel` — sliced-K variant (`kPartitionsK > 1`): two warps per CTA walk interleaved stripes of K in separate FP32 accumulators, FADD-summed before the epilogue. Required for the `sliced1x2` kernel family.
- `split_k_workspace_outtype` — `GemmSplitKParallel + ReduceSplitK`: each CTA writes an independent FP32→BF16 partial to workspace, a separate reduction kernel sums them in FP32. Different BF16-rounding pattern from `INPLACE_ATOMIC`.

**Dispatch inspector** (`cublaslt_inspect.cu`). A small cuBLASLt harness that queries `cublasLtMatmulAlgoGetHeuristic` for any (M, N, K) and reports the top-ranked algo's tile, stages, split-K, reduction scheme, swizzle, inner, and cluster parameters. Uses `tn` layout (`TRANSA=T, TRANSB=N`) to match PyTorch's `F.linear` dispatch.

**Catalog builder** (`build_catalog.py`). Sweeps a grid of shapes, queries the inspector for each, identifies the recipe family from the dispatch metadata + kernel symbol captured via `torch.profiler`, and emits a JSON catalog of `(shape → recipe + params)`. An optional verification run per shape confirms the recipe matches cuBLAS bit-exactly.

**CPU-only runtime** (`catalog_lookup.py`). At lookup time no GPU is required: `catalog_matmul(A, B, catalog)` returns the bit-exact cuBLAS output for any shape covered by the catalog.

This is the same structure proposed in the writeup's "kernel descriptors" section — a compact dispatch descriptor per shape plus a published recipe for each kernel family. For closed-source libraries where the descriptor can't be published, the same information is recoverable by black-box sweeping with the inspector.

## File Summary

| File | What |
|------|------|
| `tc_profiles.py` | Tensor core parameters per GPU (NFMA, neab, rounding, MMA shape) |
| `tc_emulator.py` | Block FMA matmul emulator (C + OpenMP, ctypes) |
| `mufu_emulator.py` | MUFU.RSQ/EX2/RCP emulator (probed correction tables) |
| `emulate_pytorch_reduce.py` | PyTorch reduction tree emulator |
| `ffn_chain_test.py` | FFN block: capture, emulate, compare (supports `EMULATOR_TARGET`) |
| `attn_chain_test.py` | Attention block: capture, emulate, compare |
| `capture_forward.py` | CUTLASS-consistent full-model forward pass capture |
| `emulate_forward.py` | Full-model forward pass emulator (all 36 layers + LM head) |
| `block_emulators.py` | Emulator-side FFN and attention block builders |
| `cublas_recipes.py` | Four bit-accurate recipes covering the cuBLAS kernel families |
| `cublaslt_inspect.cu` | cuBLASLt dispatch inspector (algo, tile, split-K, reduction scheme) |
| `build_catalog.py` | One-time shape sweep → static `cublas_catalog.json` |
| `catalog_lookup.py` | CPU-only runtime shape-to-recipe lookup |
| `cublas_gemm_fp32.cu` | cuBLASLt harness with FP32 output (determinism probe) |
| `cask_probe.py` | Scans libcublasLt's `.cask_resource` for nvjet cubins (see Limitations) |
| `cutlass_gemm_flex.cu` | CUTLASS binary for ground-truth matmul (source) |

## Limitations

- **`nvjet_*` kernel family on H100** — at some shapes (e.g. Qwen3-4B `down_proj` with seq=100 on H100) cuBLAS dispatches to NVIDIA's proprietary `nvjet_sm90_*` family, which is not a CUTLASS-template kernel and whose bodies are not present in `libcublasLt`'s `.nv_fatbin`. A separate `.cask_resource` section contains 110.7 MB of data with no valid CUDA-ELF headers, no plaintext kernel-name strings, and no recognizable container magic (entropy 6.91 bits/byte) — consistent with proprietary packed storage. Extracting nvjet cubins from this section would require reverse-engineering the container format and is left to future work. The other three cuBLAS kernel families are fully covered by the recipe library.
- **Attention + multi-GPU + multi-layer** — the attention chain is validated on A100 only; extending FA2 emulation to L40S/H100 is engineering, not research (the hardware generalization is already demonstrated by the FFN chain). Full 36-layer forward-pass validation is on A100, seq ≤ 250.
- **Single CUTLASS tile config** — tested `{128,128,64}/{64,64,64}/{16,8,16}`. The block-FMA model is tile-independent, but different tile configs change the K-iteration order and need verification.

## Key Principle

Every emulator choice matches a specific GPU behavior:

| Emulator does this | Because the GPU does this |
|---|---|
| Block FMA with (2+23+neab)-bit window, truncation | Tensor core fixed-point accumulator |
| Significands shifted left by neab before alignment | Hardware adds extra precision bits at the bottom (K&M, arXiv:2512.07004) |
| Denormalized products in alignment | Raw significand multiply, no normalize-to-FP32 |
| NFMA=8 CoFDA on Ampere/Ada, NFMA=16 FDA on Hopper | Different block sizes across GPU generations (MMA-Sim, arXiv:2511.10909) |
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
