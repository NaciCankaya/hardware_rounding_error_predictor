#!/usr/bin/env python3
"""
Emulation of PyTorch's reduce_kernel for FP32 sum reduction.

SOFTWARE STACK: This entire module exists because we are running on:
  1. PyTorch (ATen CUDA kernels) — specifically Reduce.cuh's reduce_kernel
  2. HuggingFace transformers — which implements RMSNorm as .pow(2).mean(-1),
     dispatching two separate PyTorch kernels (element-wise + reduction)

If EITHER of these changes, this module may need to be rewritten:
  - A fused RMSNorm kernel (e.g., Apex, Triton, FlashNorm) would have a
    completely different internal reduction tree
  - torch.compile may fuse .pow(2) into the reduction, changing the
    accumulation order
  - A different framework (JAX, TensorRT) would use different reduction code

Every specific sub-assumption within this module is marked with:
    # SOFTWARE STACK ASSUMPTION: ...

HARDWARE is handled separately by tc_profiles.py / tc_emulator.py.
This file has NO hardware assumptions — only software ones.

Usage:
    /venv/main/bin/python emulate_pytorch_reduce.py          # diagnostic
    /venv/main/bin/python emulate_pytorch_reduce.py test     # synthetic
"""

import numpy as np
import struct
import os
import sys


def fp32_add(a, b):
    """IEEE-754 FP32 addition. Hardware-independent (IEEE-754 spec)."""
    return np.float32(np.float32(a) + np.float32(b))


# ============================================================
# Configuration: derived from PyTorch's ReduceConfig heuristics
# ============================================================

# SOFTWARE STACK ASSUMPTION: vt0=4 is the default template parameter in
# ReduceOp (Reduce.cuh line ~300: "int vt0=4").  This is a compile-time
# constant in PyTorch.  Changed → different accumulator grouping.
VT0 = 4

# SOFTWARE STACK ASSUMPTION: input_vec_size defaults to vt0 in the template
# (Reduce.cuh: "int input_vec_size=vt0").  If PyTorch changes this default
# or specializes it for certain dtypes, the vectorized path changes.
INPUT_VEC_SIZE = VT0

# HARDWARE CONSTANT (not software): WARP_SIZE=32 for all NVIDIA GPUs.
# If AMD ROCm (wavefront=64), this changes — but then the entire
# Reduce.cuh codepath differs (USE_ROCM branches).
WARP_SIZE = 32

# SOFTWARE STACK ASSUMPTION: MAX_NUM_THREADS=512 for all types except
# complex<double> (which uses 256).  See mnt_wrapper<T> in Reduce.cuh.
# For FP32 sum reduction, 512 is correct.
MAX_NUM_THREADS = 512


def compute_block_shape(reduction_dim, num_outputs, max_threads=MAX_NUM_THREADS,
                        warp_size=WARP_SIZE):
    """
    Reproduce PyTorch's set_block_dimension() from Reduce.cuh.

    SOFTWARE STACK ASSUMPTION: This reproduces the EXACT heuristic from
    PyTorch's ReduceConfig::set_block_dimension<T>().  If PyTorch changes
    the heuristic (e.g., different balancing between block_width and
    block_height), the block shape changes and the entire reduction tree
    changes with it.

    Returns: (block_width, block_height)
    """
    def last_pow2(n):
        n |= (n >> 1); n |= (n >> 2); n |= (n >> 4)
        n |= (n >> 8); n |= (n >> 16)
        return max(1, n - (n >> 1))

    dim0 = reduction_dim
    dim1 = num_outputs

    dim0_pow2 = last_pow2(dim0) if dim0 < max_threads else max_threads
    dim1_pow2 = last_pow2(dim1) if dim1 < max_threads else max_threads

    block_width = min(dim0_pow2, warp_size)
    block_height = min(dim1_pow2, max_threads // block_width)
    block_width = min(dim0_pow2, max_threads // block_height)

    return block_width, block_height


def should_vectorize(reduction_dim, contiguous=True):
    """
    Determine whether PyTorch uses the vectorized reduction path.

    SOFTWARE STACK ASSUMPTION: Vectorization is enabled when:
      - reduction is on the fastest-striding (contiguous) dimension
      - reduction_dim >= 128
      - num_reduce_dims == 1
      - fastest_moving_stride == sizeof(scalar_t)
    These conditions are from gpu_reduce_kernel() in Reduce.cuh.
    If PyTorch changes the threshold (e.g., from 128 to 256), this breaks.
    """
    return contiguous and reduction_dim >= 128


# ============================================================
# Vectorized path: input_vectorized_thread_reduce_impl
# ============================================================
def thread_reduce_vectorized(data_row, tx, stride, vec_size=INPUT_VEC_SIZE):
    """
    Emulate input_vectorized_thread_reduce_impl for one thread.

    SOFTWARE STACK ASSUMPTION: Vector load of vec_size consecutive elements,
    with vec_size separate accumulators combined left-to-right at the end.
    This matches Reduce.cuh's input_vectorized_thread_reduce_impl.
    The combination order is: ((acc[0]+acc[1])+acc[2])+acc[3].

    SOFTWARE STACK ASSUMPTION: Assumes data pointer is aligned to
    alignof(aligned_vector<float, vec_size>) = 16 bytes, so shift=0
    and the alignment-head handling is skipped.  For contiguous [M, H]
    tensors where H*sizeof(float) is divisible by 16, this holds for
    every row.  Non-contiguous tensors or unusual H values may violate this.
    """
    N = len(data_row)
    accs = [np.float32(0.0)] * vec_size
    idx = tx

    while idx * vec_size + vec_size - 1 < N:
        base = idx * vec_size
        for i in range(vec_size):
            accs[i] = fp32_add(accs[i], data_row[base + i])
        idx += stride

    # Tail
    tail_start = N - (N % vec_size)
    if N % vec_size > 0 and tx < (N % vec_size):
        if tail_start + tx < N:
            accs[0] = fp32_add(accs[0], data_row[tail_start + tx])

    # Combine: left-to-right
    for i in range(1, vec_size):
        accs[0] = fp32_add(accs[0], accs[i])

    return accs[0]


# ============================================================
# Non-vectorized path: thread_reduce_impl (contiguous)
# ============================================================
def thread_reduce_nonvec(data_row, tx, stride, vt0=VT0):
    """
    Emulate thread_reduce_impl with contiguous offset calc.

    SOFTWARE STACK ASSUMPTION: Strided access with vt0 separate accumulators.
    The contiguous path uses calc = [](idx) { return idx; }, meaning
    element_stride=1.  This matches contiguous tensor reduction only.
    For non-contiguous tensors, the offset calculator changes.
    """
    N = len(data_row)
    value_list = [np.float32(0.0)] * vt0
    idx = tx

    while idx + (vt0 - 1) * stride < N:
        for i in range(vt0):
            value_list[i] = fp32_add(value_list[i], data_row[idx + i * stride])
        idx += stride * vt0

    # Tail
    for i in range(vt0):
        if idx >= N:
            break
        value_list[i] = fp32_add(value_list[i], data_row[idx])
        idx += stride

    # Combine: left-to-right
    for i in range(1, vt0):
        value_list[0] = fp32_add(value_list[0], value_list[i])

    return value_list[0]


# ============================================================
# Warp-level reduction (block_x_reduce)
# ============================================================
def warp_reduce(thread_vals, decreasing=True):
    """
    Emulate block_x_reduce.

    SOFTWARE STACK ASSUMPTION: Warp shuffle direction.
      - PyTorch >= ~2.7 (merged Oct 2025): DECREASING offsets (16,8,4,2,1)
      - PyTorch <= 2.6: INCREASING offsets (1,2,4,8,16)
    Check your version:
      grep -A2 "Intra-warp reduction" .../torch/include/ATen/native/cuda/Reduce.cuh

    Handles both cases:
      - block_width <= WARP_SIZE: pure warp shuffle (no shared memory)
      - block_width > WARP_SIZE:  shared memory tree first, then warp shuffle
    """
    vals = np.array(thread_vals, dtype=np.float32).copy()
    dim_x = len(vals)

    # Phase 1: shared memory reduction (only if block_width > WARP_SIZE)
    if dim_x > WARP_SIZE:
        offset = dim_x // 2
        while offset >= WARP_SIZE:
            new_vals = vals.copy()
            for t in range(offset):
                if t + offset < dim_x:
                    new_vals[t] = fp32_add(vals[t], vals[t + offset])
            vals = new_vals
            offset >>= 1
        dim_x = WARP_SIZE

    # Phase 2: intra-warp shuffle
    if decreasing:
        offset = dim_x >> 1
        while offset > 0:
            new_vals = vals.copy()
            for t in range(dim_x):
                src = t + offset
                other = vals[src] if src < dim_x else np.float32(0.0)
                new_vals[t] = fp32_add(vals[t], other)
            vals = new_vals
            offset >>= 1
    else:
        offset = 1
        while offset < dim_x:
            new_vals = vals.copy()
            for t in range(dim_x):
                src = t + offset
                other = vals[src] if src < dim_x else np.float32(0.0)
                new_vals[t] = fp32_add(vals[t], other)
            vals = new_vals
            offset <<= 1

    return vals[0]


# ============================================================
# Full row reduction
# ============================================================
def reduce_row(data_row, block_width, vectorize=True, warp_shfl_decreasing=True):
    """
    Reduce one row of FP32 values using the PyTorch reduction tree.
    Returns the SUM (not mean — the project/division step is separate).
    """
    thread_vals = np.zeros(block_width, dtype=np.float32)

    for tx in range(block_width):
        if vectorize:
            thread_vals[tx] = thread_reduce_vectorized(data_row, tx, stride=block_width)
        else:
            thread_vals[tx] = thread_reduce_nonvec(data_row, tx, stride=block_width)

    return warp_reduce(thread_vals, decreasing=warp_shfl_decreasing)


# ============================================================
# Matrix-level: process all rows
# ============================================================
def emulate_sum_reduce(x_matrix_bf16_f32, block_width=None, vectorize=None,
                       warp_shfl_decreasing=True):
    """
    Drop-in replacement for:
        emu_sumsq = np.sum(x_f32 ** 2, axis=-1, keepdims=True)

    SOFTWARE STACK ASSUMPTION: x.pow(2) is a SEPARATE element-wise kernel
    that runs before the reduction kernel.  The reduction receives already-
    squared FP32 values.  If PyTorch fuses squaring into the reduction
    (e.g., via torch.compile or a custom fused kernel), the accumulation
    changes because squaring would happen inside ops.reduce() instead of
    beforehand.

    SOFTWARE STACK ASSUMPTION: The FP32 square x*x is computed by a standard
    vectorized_elementwise_kernel, which is IEEE-754 exact (same on CPU/GPU).

    Returns: (M, 1) FP32 array of sum-of-squares per row.
    """
    M, H = x_matrix_bf16_f32.shape
    x = np.asarray(x_matrix_bf16_f32, dtype=np.float32)

    # Element-wise square — same kernel on CPU and GPU (IEEE-754)
    squared = (x * x).astype(np.float32)

    # Derive block shape if not provided
    if block_width is None:
        block_width, _ = compute_block_shape(H, M)
    if vectorize is None:
        vectorize = should_vectorize(H)

    result = np.zeros((M, 1), dtype=np.float32)
    for row in range(M):
        result[row, 0] = reduce_row(squared[row], block_width, vectorize, warp_shfl_decreasing)
    return result


# ============================================================
# Diagnostic
# ============================================================
def load_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def diagnostic(data_dir="ffn_chain_data"):
    """Compare all variants against GPU ground truth."""
    import json
    meta_path = os.path.join(data_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"No {meta_path} found. Run ffn_chain_test.py extract first.")
        return

    with open(meta_path) as f:
        meta = json.load(f)

    M = meta["seq_len"]
    H = meta["hidden_dim"]
    eps = meta["eps"]

    block_width, block_height = compute_block_shape(H, M)
    vec = should_vectorize(H)
    elems_per_thread = (H + block_width - 1) // block_width

    print(f"Seq={M}, Hidden={H}, eps={eps}")
    print(f"Block = ({block_width}, {block_height}), "
          f"{block_width} threads per row, stride={block_width}, "
          f"{elems_per_thread} elements/thread")
    print(f"Vectorize={vec} (reduction_dim={H} >= 128)")
    print()

    residual = load_bin(os.path.join(data_dir, "model_ffn_residual.bin"), (M, H))
    gpu_sumsq = load_bin(os.path.join(data_dir, "gpu_rms_sumsq.bin"), (M, 1))
    gpu_variance = load_bin(os.path.join(data_dir, "gpu_rms_variance.bin"), (M, 1))

    import torch
    residual_bf16 = torch.tensor(residual).bfloat16().float().numpy()

    variants = [
        ("vec + shfl_dec", True, True),
        ("vec + shfl_inc", True, False),
        ("non-vec + shfl_dec", False, True),
        ("non-vec + shfl_inc", False, False),
    ]

    print(f"  {'Variant':<28} {'SumSq FP32':>14} {'Var FP32':>14} {'SumSq BF16':>14}")
    print("  " + "-" * 72)

    best_match = 0
    best_cfg = None

    for name, v, sd in variants:
        ss_match = var_match = ss_bf16 = 0

        for row in range(M):
            squared = (residual_bf16[row] * residual_bf16[row]).astype(np.float32)
            emu_ss = reduce_row(squared, block_width=block_width, vectorize=v, warp_shfl_decreasing=sd)
            emu_var = np.float32(emu_ss / np.float32(H))

            if struct.pack('f', float(emu_ss)) == struct.pack('f', gpu_sumsq[row, 0]):
                ss_match += 1
            if struct.pack('f', float(emu_var)) == struct.pack('f', gpu_variance[row, 0]):
                var_match += 1
            if torch.tensor(emu_ss).bfloat16() == torch.tensor(gpu_sumsq[row, 0]).bfloat16():
                ss_bf16 += 1

        tag = " ***" if ss_match == M else ""
        print(f"  {name:<28} {ss_match:>5}/{M:<5}    {var_match:>5}/{M:<5}    {ss_bf16:>5}/{M:<5}{tag}")

        if ss_match > best_match:
            best_match = ss_match
            best_cfg = (name, v, sd)

    # Baseline
    print()
    ss_match = var_match = 0
    for row in range(M):
        x_row = residual_bf16[row].astype(np.float32)
        emu_ss = np.sum(x_row ** 2).astype(np.float32)
        emu_var = np.float32(emu_ss / np.float32(H))
        if struct.pack('f', float(emu_ss)) == struct.pack('f', gpu_sumsq[row, 0]):
            ss_match += 1
        if struct.pack('f', float(emu_var)) == struct.pack('f', gpu_variance[row, 0]):
            var_match += 1
    print(f"  {'numpy sequential':<28} {ss_match:>5}/{M:<5}    {var_match:>5}/{M:<5}")

    if best_cfg and best_match == M:
        print(f"\n  *** PERFECT MATCH: {best_cfg[0]} — {M}/{M} rows FP32 bit-exact ***")
    elif best_cfg:
        print(f"\n  Best: {best_cfg[0]} ({best_match}/{M})")


def quick_test():
    """Quick test with synthetic data."""
    import torch
    np.random.seed(42)
    H = 2560
    x = torch.randn(H).bfloat16().float().numpy()
    squared = (x * x).astype(np.float32)

    bw, bh = compute_block_shape(H, 1)
    print(f"Reducing {H} FP32 elements: block_width={bw} (from heuristic)")
    print()

    for name, vec, shfl in [
        ("vec + dec", True, True),
        ("vec + inc", True, False),
        ("non-vec + dec", False, True),
        ("non-vec + inc", False, False),
    ]:
        result = reduce_row(squared, block_width=bw, vectorize=vec, warp_shfl_decreasing=shfl)
        print(f"  {name:<20}: {float(result):.8e}")

    print(f"  {'numpy sequential':<20}: {float(np.sum(squared)):.8e}")
    print(f"  {'torch CPU':<20}: {float(torch.tensor(squared).sum()):.8e}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        quick_test()
    else:
        diagnostic()
