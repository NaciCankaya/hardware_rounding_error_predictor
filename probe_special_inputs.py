#!/usr/bin/env python3
"""What the GPU returns for special inputs of MUFU.RCP, MUFU.RSQ and MUFU.EX2: zeros, infinities,
NaNs, subnormals, negative values, and the values around the flush-to-zero bounds.

Uses the probe kernels of mufu_emulator.py (rcp.approx.ftz.f32, exp2f with --use_fast_math) and
torch.rsqrt, as the emulator's tables are measured. Writes
<MUFU_CACHE_DIR>/<gpu>/special_inputs.npz: in_bits, rcp, rsq, ex2 (uint32 bit patterns), fixed
(the number of hand-picked inputs at the front). check_special_inputs.py compares the emulator
with it.

    python probe_special_inputs.py          on the GPU
"""
import os

import numpy as np
import torch

import mufu_emulator as M

FIXED = [0x00000000, 0x80000000, 0x7F800000, 0xFF800000,
         0x7FC00000, 0xFFC00000, 0x7F800001, 0xFF800001, 0x7FBFFFFF, 0x7FFFFFFF, 0xFFFFFFFF, 0x7FC12345, 0xFFA54321,
         0x00000001, 0x000001FF, 0x00400000, 0x007FFFFF, 0x80000001, 0x80400000, 0x807FFFFF,
         0x00800000, 0x00800001, 0x00FFFFFF, 0x01000000, 0x7F7FFFFF, 0x7F000000, 0x7E800000, 0x7E800001, 0x7E7FFFFF,
         0x80800000, 0xBF800000, 0xC0200000, 0xFE800001, 0xFF7FFFFF, 0x3F800000, 0x40000000]


def inputs():
    rng = np.random.default_rng(11)

    def pats(sign, e_lo, e_hi, n):
        exp = rng.integers(e_lo, e_hi, n).astype(np.uint32) << 23
        return (np.uint32(sign) << 31) | exp | rng.integers(0, 1 << 23, n).astype(np.uint32)

    return np.concatenate([np.array(FIXED, np.uint32),
                           pats(1, 1, 255, 200_000),                                # negative normal
                           pats(0, 0, 1, 100_000), pats(1, 0, 1, 100_000),          # subnormal
                           pats(0, 255, 256, 50_000), pats(1, 255, 256, 50_000),    # NaN (and inf)
                           pats(0, 253, 255, 100_000), pats(0, 1, 3, 100_000)])     # largest, smallest normals


if __name__ == "__main__":
    gpu = M._detect_gpu_name()
    bits = inputs()
    x = bits.view(np.float32)
    out = {"in_bits": bits,
           "rcp": M._gpu_rcp_fast_math(x).view(np.uint32),
           "ex2": M._gpu_exp2_fast_math(x).view(np.uint32),
           "rsq": torch.rsqrt(torch.tensor(x, device="cuda")).cpu().numpy().view(np.uint32)}
    path = os.path.join(M.CACHE_DIR, gpu, "special_inputs.npz")
    np.savez(path, fixed=len(FIXED), **out)
    print(f"{torch.cuda.get_device_name(0)}: {len(bits)} inputs -> {path}")
