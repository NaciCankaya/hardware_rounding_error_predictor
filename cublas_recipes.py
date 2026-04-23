"""
cuBLAS-matching recipes for the FFN chain emulator.

Each recipe takes BF16-precision FP32 numpy arrays A and B, returns a
BF16-precision FP32 numpy array that should match cuBLAS's BF16 output
bit-exactly for the kernel family the recipe targets.

Three recipe families currently implemented:
  - single_walk:               one sequential K-walk, matches Split-K=1/no-reduction kernels
  - split_k_cutlass_bf16_out:  serial Split-K with BF16 round-trip per partition
                                (stock CUTLASS kGemm mode, kPartitionsK=1)
  - split_k_sliced_kernel:     sliced-K variant (kPartitionsK>1); two warps per CTA
                                walk interleaved stripes of K, FADD-summed before the
                                LinearCombination epilogue, then outer Split-K semaphore-serial

Derivations: gemm_universal.h kGemm branch, linear_combination.h set_k_partition,
epilogue.h PartitionsK>1 reduction loop, and SASS disassembly of
ampere_bf16_s1688gemm_bf16_128x64_sliced1x2_ldg8_f2f_nn for the sliced variant.
"""
import math
import numpy as np
import torch
from tc_profiles import get_profile, detect_gpu
from tc_emulator import TensorCoreEmulator

# Module-level emulator instance (reused across recipe calls).
# Initialized lazily so import doesn't require a GPU.
_EMU = None
def _emu():
    global _EMU
    if _EMU is None:
        _EMU = TensorCoreEmulator(
            get_profile(detect_gpu(), input_fmt="bf16", output_fmt="fp32")
        )
    return _EMU


def _bf16_round_fp32(fp32_np):
    """Round FP32 values to BF16 precision, keep as FP32 container."""
    return torch.from_numpy(fp32_np).bfloat16().float().numpy()


# ----------------------------------------------------------------------
# Recipe: stock CUTLASS Split-K + BF16 round-trip per partition
# ----------------------------------------------------------------------
def split_k_cutlass_bf16_out(A_np, B_np, split_k=1, tb_K=64):
    """
    For stock CUTLASS GEMM templates with kPartitionsK == 1
    (no intra-CTA warp reduction). First partition writes BF16 directly;
    subsequent partitions read BF16 back, FP32-add, write BF16.
    """
    K = A_np.shape[1]
    gemm_k_iters = math.ceil(K / (split_k * tb_K))
    gemm_k_size  = gemm_k_iters * tb_K
    acc = None
    for s in range(split_k):
        start, end = s * gemm_k_size, min((s + 1) * gemm_k_size, K)
        partial = _emu().matmul(A_np[:, start:end], B_np[start:end, :])
        acc = (partial + acc).astype(np.float32) if acc is not None else partial
        acc = _bf16_round_fp32(acc)
    return acc


# ----------------------------------------------------------------------
# Recipe: sliced-K (kPartitionsK > 1) + outer Split-K
# ----------------------------------------------------------------------
def split_k_sliced_kernel(A_np, B_np, split_k=1, tb_K=64, warp_K=32):
    """
    For kernels with sliced-K (kPartitionsK > 1): each CTA has multiple warps
    walking interleaved warp_K stripes of its K-slice in separate FP32
    accumulators, then FADD-summed before the epilogue. Outer Split-K is the
    same semaphore-serial + BF16 round-trip as in split_k_cutlass_bf16_out.
    """
    K = A_np.shape[1]
    gemm_k_iters = math.ceil(K / (split_k * tb_K))
    gemm_k_size  = gemm_k_iters * tb_K
    k_partitions = tb_K // warp_K
    result = None
    for s in range(split_k):
        cta_start = s * gemm_k_size
        cta_end   = min((s + 1) * gemm_k_size, K)
        partition_accum = []
        for p in range(k_partitions):
            A_parts, B_parts = [], []
            for base in range(cta_start, cta_end, tb_K):
                w_start = base + p * warp_K
                w_end   = min(base + (p + 1) * warp_K, cta_end)
                if w_start < cta_end:
                    A_parts.append(A_np[:, w_start:w_end])
                    B_parts.append(B_np[w_start:w_end, :])
            A_w = np.concatenate(A_parts, axis=1)
            B_w = np.concatenate(B_parts, axis=0)
            partition_accum.append(_emu().matmul(A_w, B_w))
        cta_partial = partition_accum[0]
        for p in range(1, k_partitions):
            cta_partial = (cta_partial + partition_accum[p]).astype(np.float32)
        result = (cta_partial + result).astype(np.float32) if result is not None else cta_partial
        result = _bf16_round_fp32(result)
    return result


# ----------------------------------------------------------------------
# Recipe: single-walk (no Split-K, no sliced-K)
# ----------------------------------------------------------------------
def single_walk(A_np, B_np, **_ignored):
    """Matches kernels with split_k=1 and no intra-CTA K partitioning —
    structurally identical to what the base emulator already does."""
    return _bf16_round_fp32(_emu().matmul(A_np, B_np))


# ----------------------------------------------------------------------
# Recipe: WORKSPACE_OUTTYPE reduction (CUTLASS GemmSplitKParallel + ReduceSplitK)
# ----------------------------------------------------------------------
def split_k_workspace_outtype(A_np, B_np, split_k=1, tb_K=64):
    """
    WORKSPACE_OUTTYPE reduction scheme: CUTLASS's GemmSplitKParallel path
    plus a separate ReduceSplitK reduction kernel.

    Mechanics (differs from semaphore-serial INPLACE_ATOMIC):
      1. Each CTA independently computes its K-slice FP32 partial,
         casts to BF16 (output dtype), stores to its own slot of a
         workspace buffer. No ordering between CTAs; no read-modify-write.
      2. A separate reduction kernel then loads all the BF16 partials,
         sums them in FP32 (a single clean accumulation in partition-index
         order), and does one final BF16 cast to the output.

    The BF16 rounding pattern differs from split_k_cutlass_bf16_out:
    there, the output is BF16-round-tripped at every partition boundary.
    Here, each partial is BF16-rounded once before the final FP32 sum,
    and the sum is BF16-rounded once at the end.
    """
    K = A_np.shape[1]
    gemm_k_iters = math.ceil(K / (split_k * tb_K))
    gemm_k_size  = gemm_k_iters * tb_K

    # Phase 1: each CTA produces an FP32 partial, casts to BF16 (output dtype)
    partials_bf16 = []
    for s in range(split_k):
        start, end = s * gemm_k_size, min((s + 1) * gemm_k_size, K)
        partial_fp32 = _emu().matmul(A_np[:, start:end], B_np[start:end, :])
        partials_bf16.append(_bf16_round_fp32(partial_fp32))

    # Phase 2: reduction kernel sums BF16 partials in FP32, final BF16 cast
    result = np.zeros_like(partials_bf16[0])
    for p in partials_bf16:
        result = (result + p).astype(np.float32)
    return _bf16_round_fp32(result)
