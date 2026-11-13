#!/usr/bin/env python3
"""The emulator against the GPU's results on special inputs (probe_special_inputs.py): zeros,
infinities, NaNs, subnormals, negative values and the flush-to-zero bounds, for MUFU.RCP,
MUFU.RSQ and MUFU.EX2. No GPU needed; exits with 1 on a mismatch.

    python check_special_inputs.py [gpu name, default H100]
"""
import os
import sys

import numpy as np

from mufu_emulator import CACHE_DIR, MUFUEmulator

if __name__ == "__main__":
    gpu = sys.argv[1] if len(sys.argv) > 1 else "H100"
    z = np.load(os.path.join(CACHE_DIR, gpu, "special_inputs.npz"))
    bits = z["in_bits"]
    x = bits.view(np.float32)
    m = MUFUEmulator(gpu_name=gpu)
    biased, frac = (bits >> 23) & 0xFF, bits & 0x7FFFFF
    kind = np.select([(biased == 255) & (frac != 0), biased == 255, (biased == 0) & (frac == 0), biased == 0],
                     ["NaN", "inf", "zero", "subnormal"], "normal")
    bad = 0
    for name, f in (("rcp", m.rcp), ("rsq", m.rsq), ("ex2", m.ex2)):
        emu = f(x).view(np.uint32)
        diff = emu != z[name]
        bad += int(diff.sum())
        by_kind = ", ".join(f"{k} {int(diff[kind == k].sum())}/{int((kind == k).sum())}" for k in ("normal", "subnormal", "zero", "inf", "NaN"))
        print(f"MUFU.{name.upper()}: {int(diff.sum())} mismatches / {len(bits)} special inputs ({by_kind})")
        for k in np.flatnonzero(diff)[:5]:
            print(f"    {bits[k]:#010x}: hardware {z[name][k]:#010x}, emulator {emu[k]:#010x}")
    sys.exit(1 if bad else 0)
