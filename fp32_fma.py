"""The fused multiply-add of FP32 values, a * b + c with one rounding, as the GPU computes it
(FlashAttention's softmax exponent and lane sums). probe_fma.py measures the H100's on operand
triples that tell one rounding from two; check_fma.py compares this function with it."""
import numpy as np


def fma_f32_vec(a, b, c):
    """Vectorized FMA: a[:]*b + c[:] with a single rounding to FP32, as the GPU's fused
    multiply-add (measured on the H100: check_fma.py).

    The product of two FP32 values is exact in float64. The float64 sum is rounded to
    odd (its rounding error by TwoSum; an inexact sum with an even last bit moves to its
    neighbour on the error's side), which the rounding to FP32 then takes as it would the
    exact sum. Rounding the float64 sum to nearest instead rounds twice: where the exact
    sum lies just beside the middle of two FP32 values, that gives the other one."""
    p = np.asarray(a, np.float64) * np.float64(b)
    c = np.asarray(c, np.float64)
    with np.errstate(invalid="ignore", over="ignore"):
        s = p + c
        t = s - p
        err = (p - (s - t)) + (c - t)
        even = (np.asarray(s).view(np.int64) & 1) == 0
        odd = np.nextafter(s, np.where(err > 0, np.inf, -np.inf))
        return np.where((err != 0) & even & np.isfinite(s), odd, s).astype(np.float32)
