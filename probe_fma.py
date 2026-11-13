#!/usr/bin/env python3
"""The GPU's fused multiply-add on operand triples that tell one rounding from two.

ties:    (1 + i 2^-12)(1 + j 2^-12) with i and j odd lies exactly between two FP32 values; an
         addend of magnitude 2^-60 .. 2^-100 of the product decides the fused result, and
         vanishes in a float64 sum.
random:  every distance between the product's and the addend's exponents.

torch.addcmul(c, a, b) is one kernel, c + a * b, which the compiler contracts to the fused
multiply-add instruction (the control: a * b and + c as two operations rounds twice in FP32).
Writes <MUFU_CACHE_DIR>/<gpu>/fma_check.npz: a, b, c, fused, two_ops (uint32 bit patterns) and
ties (the number of tie triples at the front). check_fma.py compares fp32_fma.fma_f32_vec with it.

    python probe_fma.py          on the GPU (torch only)
"""
import os

import numpy as np
import torch

import mufu_emulator as M


def triples(n=200_000, m=4_000_000):
    rng = np.random.default_rng(7)
    i, j = 2 * rng.integers(0, 1024, n) + 1, 2 * rng.integers(0, 1024, n) + 1
    scale_a, scale_b = 2.0 ** rng.integers(-20, 20, n), 2.0 ** rng.integers(-20, 20, n)
    a = ((1 + i * 2.0 ** -12) * scale_a).astype(np.float32)
    b = ((1 + j * 2.0 ** -12) * scale_b).astype(np.float32)
    c = (rng.choice([-1.0, 1.0], n) * 2.0 ** rng.integers(-100, -59, n) * scale_a * scale_b).astype(np.float32)

    def pat(e_lo, e_hi, k):
        sign = rng.integers(0, 2, k).astype(np.uint32) << 31
        return (sign | (rng.integers(e_lo, e_hi, k).astype(np.uint32) << 23) | rng.integers(0, 1 << 23, k).astype(np.uint32)).view(np.float32)

    ra, rb = pat(100, 150, m), pat(100, 150, m)
    rc = ((ra.astype(np.float64) * rb.astype(np.float64)) * 2.0 ** rng.integers(-60, 60, m) * rng.choice([-1.0, 1.0], m)).astype(np.float32)
    rc = np.where(np.isfinite(rc), rc, np.float32(1.0)).astype(np.float32)
    return np.concatenate([a, ra]), np.concatenate([b, rb]), np.concatenate([c, rc]), n


if __name__ == "__main__":
    a, b, c, n = triples()
    ta, tb, tc = (torch.from_numpy(v).cuda() for v in (a, b, c))
    fused = torch.addcmul(tc, ta, tb)
    two = ta * tb
    two = two + tc
    bits = lambda v: v.view(np.uint32)
    path = os.path.join(M.CACHE_DIR, M._detect_gpu_name(), "fma_check.npz")
    np.savez(path, a=bits(a), b=bits(b), c=bits(c), ties=n, fused=bits(fused.cpu().numpy()), two_ops=bits(two.cpu().numpy()))
    print(f"{torch.cuda.get_device_name(0)}: {len(a)} triples -> {path}")
