#!/usr/bin/env python3
"""
Unit test: which tensor core profile does CUTLASS on H100 actually use?

Runs a small matmul through CUTLASS and compares against the emulator
with both A100 (nfma=8, neab=1) and H100 (nfma=16, neab=2) profiles.

If the CUTLASS binary compiled for sm_90 uses mma.sync (same instruction
as A100), its block FMA may match the A100 profile rather than H100's.
Khattak & Mikaitis may have characterized H100 on wgmma, not mma.sync.

Usage:
    python3 test_h100_profile.py
"""

import numpy as np
import subprocess
import os
import sys

# Add workspace to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tc_profiles import get_profile
from tc_emulator import TensorCoreEmulator

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def to_bf16_f32(x):
    """Snap FP32 values to BF16 grid."""
    if HAS_TORCH:
        return torch.tensor(x).bfloat16().float().numpy()
    raise RuntimeError("torch required for BF16 conversion")


def run_cutlass(M, K, N, A, B, cutlass_bin="./cutlass_gemm_flex"):
    """Run CUTLASS GEMM and return raw FP32 output."""
    a_path = "/tmp/_test_profile_a.bin"
    b_path = "/tmp/_test_profile_b.bin"
    d_path = "/tmp/_test_profile_d.bin"

    A.astype(np.float32).tofile(a_path)
    B.astype(np.float32).tofile(b_path)

    cmd = [cutlass_bin, str(M), str(K), str(N), a_path, b_path, d_path, "fp32"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(f"CUTLASS failed: {r.stderr}")
        return None

    return np.fromfile(d_path, dtype=np.float32).reshape(M, N)


def main():
    cutlass_bin = "./cutlass_gemm_flex"
    if not os.path.exists(cutlass_bin):
        print(f"ERROR: {cutlass_bin} not found")
        return

    # Use a K dimension that exercises multiple blocks:
    # K=16: one MMA step → nfma=8 does 2 blocks, nfma=16 does 1 block
    # K=32: two MMA steps
    # Use random BF16 data with varied magnitudes to trigger alignment differences
    rng = np.random.RandomState(42)

    profiles_to_test = [
        ("A100", "bf16", "fp32"),  # nfma=8, neab=1, window=26
        ("H100", "bf16", "fp32"),  # nfma=16, neab=2, window=27
    ]

    # Build emulators
    emulators = {}
    for gpu, ifmt, ofmt in profiles_to_test:
        p = get_profile(gpu, ifmt, ofmt)
        emulators[gpu] = TensorCoreEmulator(p)
        print(f"{gpu}: nfma={p.nfma}, neab={p.neab}, window={p.window_bits}, "
              f"round={p.round_mode}")

    print()

    # Test multiple shapes to get a clear signal
    test_shapes = [
        (4, 16, 4),    # minimal: 1 MMA step
        (4, 32, 4),    # 2 MMA steps
        (4, 64, 4),    # 4 MMA steps
        (16, 16, 16),  # full MMA tile
        (32, 256, 32), # realistic small
        (64, 2560, 64),# realistic Qwen-like K
    ]

    for M, K, N in test_shapes:
        # Generate BF16-snapped random data with mixed magnitudes
        A_raw = rng.randn(M, K).astype(np.float32) * rng.choice([0.01, 0.1, 1.0, 10.0], size=(M, K))
        B_raw = rng.randn(K, N).astype(np.float32) * rng.choice([0.01, 0.1, 1.0, 10.0], size=(K, N))
        A = to_bf16_f32(A_raw)
        B = to_bf16_f32(B_raw)

        # CUTLASS ground truth
        cut = run_cutlass(M, K, N, A, B, cutlass_bin)
        if cut is None:
            continue

        # Emulator predictions
        print(f"Shape [{M},{K}]x[{K},{N}]:")
        for gpu, emu in emulators.items():
            emu_out = emu.matmul(A, B)
            fp32_match = int(np.sum(emu_out.view(np.uint32) == cut.view(np.uint32)))
            fp32_total = cut.size
            fp32_diff = fp32_total - fp32_match

            # BF16-level comparison
            emu_bf16 = to_bf16_f32(emu_out)
            cut_bf16 = to_bf16_f32(cut)
            bf16_diffs = int(np.sum(emu_bf16.view(np.uint32) != cut_bf16.view(np.uint32)))

            print(f"  {gpu:5s} profile: FP32 {fp32_match}/{fp32_total} exact "
                  f"({fp32_diff} diffs), BF16 {bf16_diffs} diffs")

        print()

    # Also dump SASS to see which instruction was used
    print("=" * 60)
    print("SASS instruction check:")
    print("=" * 60)
    r = subprocess.run(
        ["cuobjdump", "-sass", cutlass_bin],
        capture_output=True, text=True
    )
    if r.returncode == 0:
        mma_lines = [l.strip() for l in r.stdout.split('\n')
                     if 'HMMA' in l or 'MMA' in l.upper()]
        seen = set()
        for l in mma_lines:
            # Extract just the instruction mnemonic
            parts = l.split()
            for p in parts:
                if 'HMMA' in p or 'MMA' in p.upper():
                    if p not in seen:
                        seen.add(p)
                        print(f"  {p}")
        if not seen:
            print("  No MMA/HMMA instructions found in SASS")
    else:
        print(f"  cuobjdump failed: {r.stderr}")


if __name__ == "__main__":
    main()
