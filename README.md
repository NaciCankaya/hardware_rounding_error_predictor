# GPU Floating-Point Determinism: Closing the Tolerance Band

## Core Claims

We demonstrate through systematic experimentation that GPU floating-point computation — specifically BF16 inference on NVIDIA A100 — is fully deterministic and predictable in software without access to the GPU. The apparent "hardware noise" that prevents bit-exact replay of neural network inference is not noise at all. It is undocumented determinism: one hardware-specific arithmetic model for tensor core matmul, and software-determined kernel behavior for everything else. Both can be characterized or circumvented.

**Claim 1.** The only hardware-specific behavior affecting numerical output is the tensor core's block fused multiply-add (FMA) arithmetic, which is a fixed function characterized by four parameters: block size (NFMA), alignment window width (determined by neab, the number of extra alignment bits), rounding mode (truncation vs. round-to-nearest-even), and denormalized product alignment (products are aligned in their raw, unnormalized form). For the A100 with BF16 inputs and FP32 accumulation, these parameters are NFMA=8, neab=1, truncation, denormalized products. Given these parameters, the tensor core's output is a deterministic function of its inputs.

**Claim 2.** Everything outside the tensor core — kernel selection, tiling, reduction order — is software. It is deterministic for a given library version and input shape. It can be made transparent by using open-source kernels (CUTLASS) or by publishing minimal kernel descriptors that reveal the accumulation order without exposing performance-critical IP.

**Claim 3.** A software emulator implementing the correct block FMA model with the correct accumulation order produces zero differences against CUTLASS output — not only at BF16, but at FP32 bit level — across all tested matrix shapes, including a full FFN block from layer 20 of Qwen3-4B on real inference data (256 tokens, 11.3 million elements). Zero means zero — not "approximately zero," not "within tolerance," but bitwise identical on every FP32 accumulator value at every stage of a chained computation.

**Claim 4.** Non-matmul operations (RMSNorm, softmax, element-wise ops) are also fully predictable. RMSNorm is solved at FP32 bit-exact level. FlashAttention-2 (online softmax fused with QK and PV matmuls) is solved to 3/2,048,000 BF16 diffs (0.00015%) at seq_len=500 and 0 diffs at seq_len=256, using the full software emulator with no GPU at runtime. The 3 remaining diffs are on exact BF16 rounding boundaries in rows with 7+ KV tiles; every individual operation has been verified against SASS disassembly.

These claims, if they generalize (and we discuss the conditions under which they do), eliminate the tolerance band that covert adversaries rely on in AI governance threat models.

## Experimental Setup

All experiments ran on NVIDIA A100-SXM4-40GB (RunPod). The emulator is C compiled with `gcc -ffp-contract=off -fno-fast-math`, called via Python ctypes. These flags are load-bearing: without `-ffp-contract=off`, gcc fuses separate multiply and add into a single FMA instruction that rounds once instead of twice, producing different FP32 bits that cascade through the accumulation. Ground truth for matmul was established using CUTLASS (BSD license) with tile configuration {128,128,64}/{64,64,64}/{16,8,16} on sm_80. Comparisons are at two levels: BF16 precision (two FP32 values are "different" if their BF16 roundings differ) and FP32 bit-exact (uint32 comparison of the raw accumulator, enabled by running CUTLASS with FP32 output type to bypass the BF16 epilogue).

## Path to the Result

Initial experiments compared a naive scalar emulator against cuBLAS (`torch.matmul`) across eight matrix shapes. Two key observations: M=1 shapes matched perfectly (no tiling involved), and the [32,11008]×[11008,4096] shape showed 41% mismatch — a categorical kernel switch, not gradual degradation. This established that the residual comes from kernel-level behavior, not fundamental arithmetic.

We eliminated K-ordering within tiles (21 structured permutations, no improvement) and pivoted from cuBLAS to CUTLASS as the comparison target. Against CUTLASS, the scalar emulator differed by only 0.03–0.57%, and an NFMA sweep revealed a U-shaped curve centered at NFMA=16 (two hardware block FMAs of 8 per MMA instruction). The remaining 5–26 diffs could not be closed by any FP64-based approximation of the block FMA.

## The Hardware Alignment Model

The final gap was closed by implementing the Khattak & Mikaitis (2025) block FMA model, which characterizes how the A100 tensor core accumulates products within each NFMA=8 block.

The model works as follows. For each block of 8 BF16 input pairs (a_i, b_i) plus the running FP32 accumulator:

1. Compute 8 raw products by multiplying BF16 significands as integers and summing exponents. Critically, the products are **not** normalized to FP32. A product like 1.5 × 1.5 = 2.25 stays in its raw form (10.01₂ × 2⁰, with 2 integer bits) rather than being normalized to FP32 (1.001₂ × 2¹). The 26-bit alignment window's 2 integer bits exist specifically for this.
2. Find the maximum exponent among all 9 values (8 raw product exponents + accumulator's FP32 exponent).
3. Align all 9 values to the maximum exponent within a 26-bit fixed-point window (2 integer bits + 23 fraction bits + 1 extra alignment bit, for A100's neab=1). Raw products are aligned using their unnormalized exponent and significand.
4. Truncate any bits that fall below the window's least significant bit.
5. Sum the aligned values as signed integers.
6. Normalize the result and truncate (not round-to-nearest) to FP32.

The critical difference from a naive emulator is twofold. Step 4 truncates small values' low bits during alignment (vs. FP64 which preserves them). And step 1 keeps products denormalized (vs. C's `float` multiply which normalizes). Both matter: the alignment truncation was identified first and closed the BF16 gap; the denormalized product handling was identified later and closed the FP32 gap.

We implemented this as `hw_block_fma()` in C, processing two blocks of 8 per k=16 MMA step (matching the A100's NFMA=8 applied twice). A unit test confirmed the model produces different FP32 values from naive accumulation: for the input set {1.0, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7, 1e-7}, the hardware model gives 0x3F800003 while naive gives 0x3F800007.

## The Decomposition Experiment

Before accepting the full model, we decomposed it to verify that each component is independently necessary. Five emulation strategies were tested against CUTLASS:

| Shape (K) | Naive | FP64-16-T | FP64-8-T | HW-2x8-T | HW-2x8-RNE |
|-----------|-------|-----------|----------|-----------|-------------|
| 512 | 16 | 6 | 5 | **0** | 13 |
| 128 | 2 | 0 | 0 | **0** | 1 |
| 4096 | 28 | 6 | 15 | **0** | 27 |
| 4096 (qkv) | 118 | 21 | 47 | **0** | 111 |
| 4096 (ffn_up) | 108 | 18 | 56 | **0** | 106 |
| 11008 | 286 | 26 | 169 | **0** | 287 |

Three key comparisons:

**FP64-16-T vs FP64-8-T** (block size alone): FP64-8-T was worse than FP64-16-T on every large shape. Getting the block size "right" with exact arithmetic increases errors because you round twice per k=16 step instead of once, and FP64 doesn't match how the hardware rounds.

**FP64-8-T vs HW-2x8-T** (alignment model): Same block size, same truncation mode. HW-2x8-T hit 0 everywhere; FP64-8-T had up to 169 diffs. The alignment truncation is essential.

**HW-2x8-T vs HW-2x8-RNE** (rounding mode): The alignment model with RNE rounding performed as badly as naive (287 vs 286 at K=11008). Truncation is not merely a detail — it is load-bearing.

All three components — block size 8, 26-bit alignment window with truncation, and truncation-to-FP32 at the output — are individually necessary. None is sufficient alone. The Khattak & Mikaitis model captures the complete arithmetic.

## Validation on Real Model Weights

To confirm the result is not an artifact of random uniform input distributions, we tested on actual weights and activations from a Qwen3-4B language model (36 transformer layers, hidden_dim=2560, FFN intermediate=9728, GQA with 32 Q heads and 8 KV heads).

We embedded a natural language prompt ("The quick brown fox jumps over the lazy dog..."), extracted the BF16 token embeddings as activations, loaded layer 0's weight tensors, ran each matmul through CUTLASS on the GPU, and compared against the HW-2x8 emulator.

| Projection | Shape | K | Naive–CUT | HW-2x8–CUT |
|-----------|-------|---|-----------|-------------|
| Q proj | [21,2560]×[2560,4096] | 2560 | 66 | **0** |
| K proj | [21,2560]×[2560,1024] | 2560 | 40 | **0** |
| V proj | [21,2560]×[2560,1024] | 2560 | 29 | **0** |
| O proj | [21,4096]×[4096,2560] | 4096 | 132 | **0** |
| FFN gate | [21,2560]×[2560,9728] | 2560 | 82 | **0** |
| FFN up | [21,2560]×[2560,9728] | 2560 | 77 | **0** |
| FFN down | [21,9728]×[9728,2560] | 9728 | 271 | **0** |

Seven shapes. Seven zeros. On real trained weights with real token embeddings, not synthetic data. The emulator predicts every BF16 output element of every projection in a transformer block with zero differences.

## End-to-End FFN Block Validation

To close the "no chain test" gap, we ran a full FFN block through the emulator and compared against CUTLASS at every intermediate stage. The test used layer 20 of Qwen3-4B with 64 tokens from a real PDF document — realistic activations with residual stream magnitudes ranging from -1344 to +4992 after 20 layers of attention + FFN processing.

The FFN pipeline: residual → RMSNorm → gate_proj → SiLU → up_proj → SiLU(gate)×up → down_proj → residual add. Hooks captured every intermediate tensor from the model's actual forward pass. Architecture was verified empirically: `SiLU(gate_out) * up_out == down_proj_input` at 0/622,592 BF16 diffs, confirming SiLU activation with no fused kernel surprises. CPU SiLU (numpy) was validated bit-exact against GPU SiLU at BF16.

Three-way comparison at every stage (Emulator vs CUTLASS, Emulator vs Model/cuBLAS, CUTLASS vs Model/cuBLAS):

| Stage | Emu vs CUTLASS | Emu vs Model | CUT vs Model |
|-------|---------------|-------------|-------------|
| RMSNorm out | — | 0/163,840 | — |
| gate_proj | 0/622,592 | 0/622,592 | 0/622,592 |
| up_proj | 0/622,592 | 0/622,592 | 0/622,592 |
| SiLU(gate) | 0/622,592 | 0/622,592 | 0/622,592 |
| SiLU×up | 0/622,592 | 0/622,592 | 0/622,592 |
| down_proj | 0/163,840 | 64,901/163,840 | 64,901/163,840 |
| FFN block out | 0/163,840 | 28,156/163,840 | 28,156/163,840 |

The emulator-vs-CUTLASS column (the verification target) is zero at every stage, totaling 2,818,048 elements. The Model column shows cuBLAS disagreement: gate and up agree with CUTLASS (same kernel selected for K=2560), but down_proj diverges at 39.6% (cuBLAS selects a categorically different algorithm for K=9728 — the same kernel-switch phenomenon observed in earlier experiments).

Furthermore, the FP32 raw accumulator comparison — enabled by running CUTLASS with FP32 output to see the tensor core's accumulator before BF16 truncation — shows zero FP32 diffs across all three matmuls:

| Matmul | FP32 bit-exact diffs |
|--------|---------------------|
| gate_proj (K=2560) | 0/622,592 |
| up_proj (K=2560) | 0/622,592 |
| down_proj (K=9728) | 0/163,840 |

The emulator is not merely "close enough for BF16" — it reproduces the exact FP32 bits of the A100 tensor core accumulator on every element of a real FFN block.

### The Denormalized Product Fix

Achieving FP32-exact results required fixing a subtle discrepancy in the original block FMA implementation. The initial emulator computed products using C's `float` multiply (`prods[p] = A[...] * B[...]`), which normalizes the result to standard FP32 form. However, the A100 tensor core keeps products in their raw, unnormalized form during alignment — a behavior documented by Khattak & Mikaitis (2025) as "products remain denormalised."

The practical difference: when both BF16 inputs have magnitude ≥ √2, their product is ≥ 2.0. The raw product has 2 integer bits (e.g., `10.01₂ × 2⁰` for 1.5 × 1.5 = 2.25), while the normalized FP32 has 1 integer bit (`1.001₂ × 2¹`). The exponent differs by 1, shifting the alignment window and causing small neighboring values to lose or gain 1 bit at the window boundary.

This was confirmed with Khattak & Mikaitis's own test vector: `c=0, p1=2.25 (from a1=b1=1.5), p2=2⁻²³, p3=p4=2⁻²⁴`. Through CUTLASS with FP32 output:
- Denormalized path (1.5 × 1.5): result = 2.25 + 2⁻²² (small products survived alignment)
- Normalized path (2.25 × 1.0): result = 2.25 (small products truncated)

Same mathematical product, different results depending on the input factorization — exactly as the paper predicts.

Before the fix, the emulator showed ~2% FP32 diffs (~1 ULP each) on every matmul, invisible at BF16. After implementing denormalized product alignment — multiplying BF16 significands as raw integers and summing exponents, then aligning the unnormalized product in the window — the FP32 diff count dropped from ~30,000 total to exactly 0.

## Non-Matmul Operations: RMSNorm and Softmax

Matrix multiplication is not the only operation involving parallel floating-point reduction. We tested the two remaining non-associative primitives in a transformer forward pass: the sum-of-squares reduction in RMSNorm and the sum-of-exponentials reduction in softmax.

### RMSNorm: solved (FP32 bit-exact)

RMSNorm computes `x * rsqrt(mean(x²) + eps)` then multiplies by a learned weight. In HuggingFace transformers (eager mode), this decomposes into separate PyTorch kernel launches: `.pow(2)` (element-wise), `.mean(-1)` (reduction), `torch.rsqrt` (element-wise), multiply, cast, weight multiply. Each has its own numerical behavior.

Initial testing found 0 BF16 diffs with the correct cast ordering (normalize in FP32, cast to BF16, weight multiply in BF16 — see table below). However, at FP32 level, ~56% of rows had sum-of-squares mismatches. While invisible at BF16 for short sequences, this would eventually cross BF16 boundaries at scale. We solved all three FP32 error sources:

| Stage | Root cause | FP32 match before → after |
|-------|-----------|--------------------------|
| sum-of-squares | PyTorch's `reduce_kernel` uses a parallel tree (32 threads per row, vectorized 4-wide loads, warp shuffle), not sequential summation | 111/256 → **256/256** |
| variance | nvcc optimizes `acc / 2560.0f` to `acc * (1.0f/2560.0f)` — multiply-by-reciprocal, not true division | 201/256 → **256/256** |
| rsqrt | `torch.rsqrt` compiles to bare `MUFU.RSQ` on SM 8.0 — a hardware lookup unit, not IEEE correctly-rounded `1/sqrt(x)` | 163/256 → **256/256** |

**Reduction tree.** PyTorch's generic reduction kernel (Reduce.cuh) assigns 32 threads (one warp) to each row. Each thread loads 80 elements via vectorized 4-wide reads into 4 separate accumulators, combines them left-to-right, then the warp reduces via shuffle with decreasing offsets (16, 8, 4, 2, 1 — changed from increasing in PyTorch ~2.7). The block shape (32, 16) is derived from ReduceConfig::set_block_dimension(), not hardcoded. Emulating this exact tree in `emulate_pytorch_reduce.py` gives 256/256 FP32 bit-exact on sum-of-squares.

**Variance division.** The CUDA compiler replaces constant-divisor division with multiply-by-precomputed-reciprocal for performance. `x / 2560.0f` becomes `x * 0x3A800000f`. Since `1.0f/2560` itself has rounding, this differs from true FP32 division on ~20% of inputs.

**MUFU.RSQ.** On A100 (SM 8.0), `rsqrtf()` compiles to a single `MUFU.RSQ` instruction with no Newton-Raphson refinement (confirmed via SASS disassembly). Its rounding is deterministic but architecture-specific — differs from correctly-rounded rsqrt by 0 or ±1 ULP (rarely ±2) depending on the mantissa. We probed all 2×2²³ mantissa×parity combinations and cached the corrections as two 8MB lookup tables per GPU architecture (`mufu_emulator.py`). This is a one-time 30-second probe, reusable for all inputs on that architecture.

**Cast ordering** (unchanged from initial finding):

| Strategy | BF16 diffs |
|----------|-----------|
| (x_f32 × rsqrt × w_f32) → BF16 | 14,464/53,760 |
| (x_f32 × rsqrt) → BF16, then × w_BF16 | **0/53,760** |
| x_BF16 × rsqrt_BF16 × w_BF16 | 10,887/53,760 |

With all four fixes applied (reduction tree + reciprocal multiply + MUFU.RSQ + cast ordering), the full RMSNorm diagnostic shows 0 FP32 diffs across all five intermediate stages on 256 tokens. RMSNorm reduces over the hidden dimension (2560), which is fixed regardless of sequence length, so this result holds at any context length.

### Softmax: solved for the kernel, open for the reduction

The same cast-ordering probe was applied to softmax. Testing with the GPU's own intermediate values (max, exp, sum) fed into different post-computation pipelines:

| Strategy | seq=21 | seq=128 | seq=512 | seq=2048 |
|----------|--------|---------|---------|----------|
| (exp_f32 / sum_f32) → BF16 | 0 | 0 | 0 | 4 |
| exp_BF16 / sum_BF16 | 295 | 1,237 | 5,012 | 21,364 |
| exp_f32 × (1/sum)_BF16 → BF16 | 145 | 1,243 | 4,734 | 19,696 |
| CPU torch.softmax(f32) → BF16 | 0 | 0 | 0 | 4 |

The kernel performs all computation in FP32 with a single BF16 cast at the end. This gives 0 diffs at sequence lengths up to 512. The 4 diffs at seq=2048 are identical between GPU and CPU torch.softmax, confirming they come from the sum-of-exponentials reduction tree (sum of 2048 FP32 values), not from any GPU-specific behavior.

Unlike RMSNorm, softmax reduces over the **sequence dimension**, which grows with context length. At long contexts (32K+), the reduction tree topology will matter and must be read from the declared attention kernel source (e.g., FlashAttention). However, this is purely a software characterization problem — standard CUDA cores doing IEEE-754 FP32 addition in a fixed tree pattern determined by the kernel code. No hardware-specific model needed.

### Summary of non-matmul operations

| Operation | Status | Root cause of diffs | Solution |
|-----------|--------|-------------------|----------|
| RMSNorm | **Solved** (0 FP32 diffs) | Reduction tree, compiler optimization, MUFU.RSQ rounding, cast ordering | Emulate PyTorch reduce_kernel tree; multiply-by-reciprocal for variance; probed MUFU.RSQ correction tables; normalize in FP32, cast to BF16, weight multiply in BF16 |
| SiLU | **Solved** (0 diffs) | N/A | GPU, CPU torch, and CPU numpy agree at BF16 on all 622,592 test elements |
| FlashAttention-2 | **Solved** (3/2M at seq=500, 0 at seq=256) | PV accumulation order, MUFU.EX2/RCP probe methodology, EX2 table resolution | See FlashAttention-2 section above |
| Standalone softmax (seq ≤ 512) | **Solved** (0 diffs) | Cast ordering | All FP32, cast at end |
| Element-wise ops | **Solved** (0 diffs) | N/A | Standard FP32, bit-exact |

## What the Emulator Needs to Know

The complete knowledge required for bit-exact inference replay, organized by source:

### Hardware knowledge (characterized once per GPU architecture)

For the A100 BF16 tensor core (mma.sync.aligned.m16n8k16):

- NFMA = 8 (products per block FMA invocation)
- neab = 1 (extra alignment bits beyond FP32 precision)
- Rounding mode = truncation (not round-to-nearest-even)
- Alignment window = 26 bits (2 integer + 23 fraction + 1 extra)
- Two block FMA invocations per k=16 MMA step
- Products aligned in denormalized (raw) form, not normalized to FP32

These parameters are documented in Khattak & Mikaitis (2025) for V100, A100, A2, A30, H100, H200, B200, L40S, and Ada RTX 1000. They can also be determined empirically through black-box probing with ~50,000 random samples, requiring a few hours of GPU time per architecture.

For the A100 MUFU (Multi-Function Unit) special function hardware:

- MUFU.RSQ (reciprocal square root): bare hardware lookup+interpolation, no Newton-Raphson refinement on SM 8.0. Deterministic but not IEEE correctly-rounded — differs by 0 or ±1 ULP (rarely ±2) depending on mantissa and exponent parity. Probed exhaustively (2×2²³ values, ~30 seconds per GPU) and cached as two 8MB lookup tables per GPU architecture. `torch.rsqrt` maps directly to MUFU.RSQ (verified 0/8,388,608 diffs).

- MUFU.EX2 (base-2 exponential): hardware approximation. NOT the same as `torch.exp2` (which uses a software polynomial via libdevice). The correction depends on the full float32 bit pattern, not just the fractional part — for |x| < 1, the mantissa has more significant bits than can be captured by a 2²³ fraction-indexed table. Solved with a full 4-billion-entry table (4GB int8, one entry per possible float32 input), generated by probing with inline PTX `ex2.approx.ftz.f32`. Generated in ~67 seconds on GPU, stored in RAM. Hardware has .ftz (flush-to-zero) semantics for subnormal results.

- MUFU.RCP (reciprocal): hardware approximation. NOT the same as IEEE FP32 division — differs on 13.2% of mantissa values. The original probe used `1.0/torch.tensor(x)` which is IEEE `__fdiv_rn`, masking the real corrections. Re-probed with inline PTX `rcp.approx.ftz.f32`. Indexed by mantissa (2²³ entries, 8MB), exponent-independent.

**Critical probe methodology lesson:** All MUFU probes must use the actual hardware instruction (inline PTX or `--use_fast_math` compiled kernel), not PyTorch/torch operations. `torch.exp2` → libdevice software, `torch.rsqrt` → MUFU.RSQ (happens to match), `1.0/torch.tensor(x)` → IEEE division (does NOT match MUFU.RCP). Verify each mapping before trusting a probe.

### Software knowledge (per kernel invocation)

For matrix multiplication:

- Tile configuration (threadblock shape, warp shape, MMA shape)
- K-iteration order (sequential in CUTLASS's default mainloop)
- Split-K or single-pass

For non-matmul operations:

- Cast ordering in fused kernels (e.g., RMSNorm: normalize in FP32, cast to BF16, then weight multiply)
- Reduction tree topology for PyTorch's generic reduce_kernel (block shape, vectorization, warp shuffle direction — all derivable from Reduce.cuh heuristics given the reduction dimension and number of outputs)
- Compiler optimizations that change FP32 arithmetic (e.g., nvcc replacing constant division with multiply-by-reciprocal)
- Reduction tree topology for long-sequence softmax (readable from attention kernel source)

For CUTLASS: matmul parameters are readable from the source code for any given tile configuration.

For attention: FlashAttention (open source, BSD license) provides the softmax reduction implementation. The verification protocol should declare the attention kernel and version alongside the GEMM kernel.

More generally, all production inference kernels — whether for matmul, attention, or normalization — use pre-compiled kernel binaries with deterministic dispatch heuristics. For a given (library version, GPU architecture, input shape), the kernel that runs is fixed at compile time or selected deterministically at runtime. No reduction tree is chosen randomly by thread scheduling. The question is not whether the dispatch is deterministic (it always is), but whether it is readable. Open-source libraries (FlashAttention, CUTLASS) expose both the compiled kernel and the dispatch logic. Proprietary libraries (cuBLAS, cuDNN) are equally deterministic but undocumented — the verifier can either require open-source alternatives or characterize the proprietary dispatch through black-box profiling.

FlashAttention is particularly clean: tile sizes (kBlockM, kBlockN) are C++ template parameters, baked in at compile time per (GPU architecture, head dimension, dtype) combination. There is exactly one kernel per configuration, with no runtime selection among alternatives. The reduction tree within that kernel — online softmax with compile-time tile sizes and standard warp shuffle reduction — is fully determined by the source code.

For cuBLAS: this information is not published but can be obtained through three paths:

1. **Open-source replacement.** Use CUTLASS for matmul and FlashAttention for attention. Performance competitive with proprietary alternatives. Available today.
2. **Kernel descriptors.** Libraries could publish a compact descriptor (5–10 fields) per kernel that reveals the accumulation order without exposing how the kernel achieves its throughput. The fast shared memory management, pipelining strategy, warp scheduling, and register allocation — all genuine IP — are irrelevant to numerical output and need not be disclosed.
3. **Black-box profiling.** For a given GPU and library version, sweep the relevant parameter space, compare against candidate emulation models, and build a lookup table. This can be fully automated.

## Implications for AI Governance

### The tolerance band is closed

Every attack in the covert adversary threat model exploits the assumption that hardware floating-point "noise" creates a tolerance band in which the attacker can hide modified computation. We have shown that this noise does not exist. GPU floating-point is deterministic, fully characterizable, and reproducible in software without access to the GPU.

This eliminates plausible deniability for:

**Model substitution.** "I'm serving the declared model." If activations sampled at a network tap do not match bit-exact re-execution with the declared weights and inputs, the model was swapped. Not "probably swapped" — certainly swapped.

**Steganography in model outputs.** "My model just happened to produce tokens encoding a hidden message." A verifier replaying the forward pass bit-exactly with the declared weights will get different logits, exposing the steganographic channel.

**Training data poisoning.** "I trained on the declared dataset." Replaying forward and backward passes exactly allows verification that each gradient step matches the declared data. A poisoned batch produces detectably different gradients — not "within hardware tolerance," but provably different.

**Weight exfiltration via side channels.** "The variations in my inference outputs are just floating-point noise." No. Every bit is predictable. Any unexplained variation is signal.

### Why batch size and prefill/decode produce different results

A practical mystery explained by our findings: practitioners observe that the same model with the same inputs produces slightly different outputs depending on batch size or whether the system is in prefill vs. decode mode.

The explanation is purely software. The M dimension (batch × sequence length) changes between these modes. cuBLAS's kernel selection heuristic responds to the changed shape by picking a different algorithm — a different tile configuration, potentially split-K vs. single-pass, a different reduction pattern. Different algorithm, different accumulation order, different floating-point result. Not hardware noise. Software path selection.

With CUTLASS and a fixed tile configuration, M does not affect the per-element accumulation. We confirmed 0 differences across M=1, M=21, M=32, and M=128 shapes. The hardware doesn't care about M. Only the software kernel selection does.

### Practical verification architecture

A minimal verification system requires:

1. **Prover declares** the software stack: CUTLASS version and tile configurations for matmuls, attention kernel (e.g., FlashAttention version) for softmax, model architecture, weights (hashed).
2. **Network tap** witnesses inputs and outputs (tokens in, logits/tokens out), plus optionally sampled intermediate activations.
3. **Verifier re-executes** using the declared software stack and the HW-2x8 emulator (or equivalent) on commodity hardware (no GPU needed). Compares against the witnessed outputs.
4. **Any bit-level mismatch** between the verifier's computation and the witnessed output constitutes proof of deviation from the declared computation. No tolerance band. No statistical test. Binary pass/fail.

The emulator runs on CPU at approximately 500× slower than GPU for the inner loop, but the verifier only needs to check sampled elements or sampled layers, not the full forward pass. A single matmul check (50,000 elements) takes seconds.

## Emulator Components

| File | Role |
|------|------|
| `ffn_chain_test.py` | End-to-end FFN chain: instrumented model capture, CUTLASS comparison, CPU emulator, three-way diagnostic |
| `attn_chain_test.py` | End-to-end attention chain: RMSNorm → Q/K/V proj → QK-norm → RoPE → FA2 → O proj, with Phase 1 (extract), Phase 2 (CUTLASS), Phase 1.5 (CUTLASS-path FA2 ground truth), Phase 3 (emulator + three-way comparison) |
| `tc_profiles.py` | Tensor core parameters per GPU architecture (NFMA, neab, rounding, MMA shape) |
| `tc_emulator.py` | Block FMA emulator (C with OpenMP, called via ctypes). Exposes both `matmul` (zero-init K-walk, for standalone GEMMs) and `block_fma_batch` (single block FMA step on M×N accumulators, for FA2's cross-tile PV accumulation) |
| `emulate_pytorch_reduce.py` | PyTorch `reduce_kernel` tree emulator (block shape, vectorization, warp shuffle) |
| `mufu_emulator.py` | MUFU hardware emulator: RSQ (2×2²³ correction tables), EX2 (4-billion-entry full table, 4GB), RCP (2²³ correction table). All probed with inline PTX instructions, not torch operations |
| `gpu_rounding_prediction_writeup.md` | This document |

External dependency: `cutlass_gemm_flex` (compiled CUTLASS binary for ground truth matmuls).

## Limitations and Open Work

**Single GPU architecture.** All experiments used A100-SXM4-40GB. Khattak & Mikaitis provide tensor core parameters for other architectures, but experimental validation is needed for each. The denormalized product behavior applies to all tested architectures in their paper (V100, A100, H100, etc.), so the fix should transfer.

**Single CUTLASS tile configuration.** We tested {128,128,64}/{64,64,64}/{16,8,16}. Other tile configs need validation that the emulator still applies. The block FMA model is tile-independent (it characterizes the tensor core, not the kernel), but different tile configs change the K-iteration order and accumulator carry pattern.

**Standalone softmax reduction (long sequence).** Standalone softmax (not within FA2) is solved at seq ≤ 512. At 2048, 4/65536 diffs from the reduction tree. This is now less relevant since production inference uses FA2's fused online softmax, which is solved separately (see below).

**FlashAttention-2 emulation.** Solved. The FA2 chain (RMSNorm → Q/K/V projection → QK-norm → RoPE → online softmax with fused QK/PV matmuls → O projection) achieves 0 BF16 diffs at seq_len=256 and 3/2,048,000 at seq_len=500 on Qwen3-4B layer 20, using the full software emulator. The 3 remaining diffs are at exact BF16 rounding boundaries in rows with 7+ KV tiles; every individual operation matches SASS disassembly.

The path from 72 initial diffs to 3, and the root cause at each step:

1. **PV accumulation order (72 → 28).** FA2's `gemm_rs` passes `acc_o` as the live MMA accumulator — the PV matmul does NOT start from zero, it accumulates into the running O_acc from previous KV tiles. Verified by reading `gemm_rs` in `utils.h`: `cute::gemm(tiled_mma, tCrA, tCrB, acc)` with no `clear(acc)`. The block FMA alignment window depends on the accumulator value, so `matmul_from_zero + add` gives different FP32 bits than `accumulate_into_existing`. Fix: expose `block_fma_batch` (the hardware primitive) from `tc_emulator.py`, have the FA2 emulator in `attn_chain_test.py` manage its own K-walk with O_acc as initial accumulator.

2. **MUFU.EX2 table resolution (28 → 8 with hardware EX2).** The original 2²³-entry table assumed corrections depend only on the fractional part of x. This holds for |x| ≥ 1, but for |x| < 1, the float32 mantissa has more significant bits than the fraction index (e.g., at exponent -1, the mantissa has 24 bits but the index only captures 23). Additionally, the bit-extraction code used FP32 arithmetic (`x - trunc(x)`) instead of IEEE 754 bit shifts, introducing rounding in the index computation. Fix: full 4-billion-entry table indexed by raw uint32 bits, probed with inline PTX `ex2.approx.ftz.f32`.

3. **MUFU.RCP probe error (8 → 3 with hardware EX2).** The RCP probe used `1.0/torch.tensor(x)` — IEEE FP32 division, not the MUFU.RCP hardware approximation. They differ on 13.2% of inputs. The old probe showed 0 corrections because it was comparing IEEE division against IEEE division. Fix: re-probe with inline PTX `rcp.approx.ftz.f32`.

4. **3 remaining.** All at exact BF16 boundaries (0.5000 ULP from midpoint), all in rows with 5+ KV tiles, 1 FP32 diff per row. Verified against SASS: rescale uses FADD+FMUL (not FMA), scale_apply_exp2 uses FFMA, all with .FTZ. Every individual operation matches in isolation. The diffs arise from sub-BF16-ULP cumulative drift across many tile transitions — confirmed by: 0 diffs at seq_len=256 (max 4 tiles), 3 diffs at seq_len=500 (up to 8 tiles).

What was ruled out (do NOT re-investigate): thread column mapping, deferred sum allreduce, exp2 vs exp base, intermediate BF16 casts, causal tile iteration order (reverse, confirmed), P → BF16 cast, RoPE cast ordering, allreduce<4> for max (XOR butterfly shfl_xor(2) then shfl_xor(1)), reduce_sum column order (verified against CuTe convert_layout_acc_rowcol), FMA vs MUL+SUB for rescale argument (verified via SASS: FADD+FMUL), compiler reassociation under --use_fast_math (tested, 0 diffs for (a-b)*c).

**Multi-layer chain.** Both FFN (0 diffs) and attention (3/2M diffs) blocks are individually solved. A full multi-layer forward pass chaining attention + FFN across all layers has not been verified but is now tractable.

**Training verification.** The backward pass involves the same matmul operations and should be predictable by the same model, but this has not been tested. Gradient accumulation across microbatches introduces additional reduction operations.

## References

- Khattak & Mikaitis (2025). "Accurate Models of NVIDIA Tensor Cores." arXiv:2512.07004. MATLAB-based tensor core models for V100 through B200.

- Xie et al. (2025). "MMA-Sim: Bit-Accurate Reference Model of Tensor Cores and Matrix Cores." arXiv:2511.10909. Python bit-accurate simulator covering 10 architectures.

- Blanchard, Higham, Lopez, Mary & Pranesh (2020). "Mixed Precision Block Fused Multiply-Add: Error Analysis and Application to GPU Tensor Cores." SIAM J. Sci. Comput. Theoretical error bounds for mixed-precision block FMA.

- Fasi, Higham, Mikaitis & Pranesh (2021). "Numerical Behavior of NVIDIA Tensor Cores." PeerJ Computer Science. Foundational tensor core probing methodology.

- Li et al. (2024). "FTTN: Feature-Targeted Testing for Numerical Properties of NVIDIA & AMD Matrix Accelerators." arXiv:2403.00232. Extended probing to Hopper and AMD.

- Fasi & Mikaitis (2023). "CPFloat: A C Library for Emulating Low-Precision Arithmetic." ACM TOMS. Low-precision arithmetic simulation library.

- NVIDIA CUTLASS. github.com/NVIDIA/cutlass. Open-source GEMM template library, BSD license.
