#!/usr/bin/env python3
"""fp32_fma.fma_f32_vec against the GPU's fused multiply-add (probe_fma.py). No GPU needed; exits
with 1 on a mismatch.

    python check_fma.py [gpu name, default H100]
"""
import os
import sys

import numpy as np

from fp32_fma import fma_f32_vec
from mufu_emulator import CACHE_DIR

if __name__ == "__main__":
    gpu = sys.argv[1] if len(sys.argv) > 1 else "H100"
    z = np.load(os.path.join(CACHE_DIR, gpu, "fma_check.npz"))
    a, b, c = (z[k].view(np.float32) for k in "abc")
    n = int(z["ties"])
    one = fma_f32_vec(a, b, c).view(np.uint32)
    twice = (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32).view(np.uint32)
    bad = 0
    for name, sl in (("ties", slice(0, n)), ("random", slice(n, None))):
        d1, d2 = int((one[sl] != z["fused"][sl]).sum()), int((twice[sl] != z["fused"][sl]).sum())
        bad += d1
        print(f"{name}: fma_f32_vec differs from the hardware on {d1} of {len(a[sl])} triples; "
              f"the sum rounded in float64 first would differ on {d2}")
    sys.exit(1 if bad else 0)
