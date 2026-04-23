#!/usr/bin/env python3
"""
MUFU Special Function Unit Emulator

Emulates NVIDIA's MUFU hardware instructions using probed correction tables.
MUFU instructions are NOT IEEE correctly-rounded — they use silicon
lookup+interpolation units whose rounding is deterministic but architecture-specific.

Supported instructions:
  - MUFU.RSQ: reciprocal square root (used by rsqrtf / torch.rsqrt)
  - MUFU.EX2: base-2 exponential (used by exp2f in FlashAttention)
  - MUFU.RCP: reciprocal (used by 1.f/x with --use_fast_math)

The emulator works by:
  1. Computing the correctly-rounded result in software (high precision)
  2. Applying a per-input correction (0 or ±1 ULP, rarely ±2)
     probed from the actual GPU hardware

PROBE METHODOLOGY (critical — previous bugs came from getting this wrong):
  Probes MUST use the actual hardware instruction, not torch operations.
  torch.exp2 → libdevice software polynomial (NOT MUFU.EX2)
  torch.rsqrt → MUFU.RSQ (happens to match, but verify per arch)
  1.0/torch.tensor(x) → IEEE __fdiv_rn division (NOT MUFU.RCP)
  The probes in this file use inline PTX assembly (ex2.approx.ftz.f32,
  rcp.approx.ftz.f32) compiled with --use_fast_math to get the real
  hardware output. If a torch operation is used, it must be verified
  against inline PTX first (see RSQ: 0/8M diffs, confirmed safe).

Correction tables:
  - RSQ: Two tables per GPU (even/odd exponent parity, 8MB each).
         Indexed by mantissa — all 2^23 values probed, exponent-independent.
  - EX2: Full 4-billion-entry table (4GB int8, one entry per possible
         float32 input). The original 2^23 fraction-indexed table was
         insufficient because for |x| < 1, float32 has more significant
         bits than the 23-bit fraction index can capture. The full table
         is generated on GPU in ~67s, stored in RAM (not cached to disk
         due to size).
  - RCP: One table per GPU (mantissa-indexed, 8MB). Indexed by mantissa
         — all 2^23 values probed, exponent-independent. 13.2% of entries
         have nonzero corrections on A100 (MUFU.RCP is NOT IEEE-exact).
  - Cache location: ./mufu_cache/<gpu_name>/ (RSQ and RCP only; EX2
    is regenerated each run)

HARDWARE-SPECIFIC: Different GPU architectures (V100, A100, H100, etc.)
have different MUFU silicon and produce different corrections.  Tables
must be re-probed for each architecture.

Usage:
    from mufu_emulator import MUFUEmulator
    mufu = MUFUEmulator()              # auto-detects GPU, loads/probes LUTs
    result = mufu.rsq(x_f32_array)     # emulates MUFU.RSQ, returns f32
    result = mufu.ex2(x_f32_array)     # emulates MUFU.EX2, returns f32
    result = mufu.rcp(x_f32_array)     # emulates MUFU.RCP, returns f32
"""

import numpy as np
import os
import sys


CACHE_DIR = "mufu_cache"


def _detect_gpu_name():
    """Detect GPU name for cache directory."""
    try:
        import torch
        if torch.cuda.is_available():
            # Normalize: "NVIDIA A100-SXM4-40GB" → "A100"
            full_name = torch.cuda.get_device_name(0)
            for tag in ["A100", "A30", "A2", "H100", "H200", "B200",
                        "V100", "L40S", "L40", "L4", "RTX"]:
                if tag in full_name:
                    return tag
            return full_name.replace(" ", "_")
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(["nvidia-smi", "--query-gpu=name",
                                 "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            name = result.stdout.strip().split("\n")[0]
            for tag in ["A100", "A30", "A2", "H100", "H200", "B200",
                        "V100", "L40S", "L40", "L4", "RTX"]:
                if tag in name:
                    return tag
            return name.replace(" ", "_")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    return None


def _probe_mufu_rsq(gpu_name):
    """
    Probe MUFU.RSQ for all 2^23 mantissa values at two exponent parities.

    MUFU.RSQ only depends on the mantissa and the parity of the (mathematical)
    exponent.  The output exponent is exact: -(e+1)/2 or -(e/2), handled by
    integer arithmetic on the exponent field.

    We probe at:
      biased_exp=127 (math_exp=0, "even") → mantissa maps [1.0, 2.0)
      biased_exp=128 (math_exp=1, "odd")  → mantissa maps [2.0, 4.0)

    Returns: (corr_even, corr_odd) — int8 arrays of shape (2^23,)
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No GPU available to probe MUFU.RSQ.  Either:\n"
            f"  1. Run this once on a {gpu_name} GPU to generate the tables, or\n"
            f"  2. Copy mufu_cache/{gpu_name}/ from a machine that has one."
        )

    detected = _detect_gpu_name()
    if detected and detected != gpu_name:
        print(f"  WARNING: probing on {detected} but cache says {gpu_name}")

    N = 2 ** 23
    mantissa_bits = np.arange(N, dtype=np.uint32)
    corrections = {}

    for label, biased_exp in [("even", 127), ("odd", 128)]:
        # Build all float32 values with this exponent and every mantissa
        float_bits = np.uint32(biased_exp << 23) | mantissa_bits
        floats = np.frombuffer(float_bits.tobytes(), dtype=np.float32)

        # GPU rsqrt (= bare MUFU.RSQ on SM 8.0+, no Newton-Raphson)
        # SOFTWARE STACK ASSUMPTION: torch.rsqrt compiles to rsqrtf()
        # which is bare MUFU.RSQ on SM 8.0 (A100).  On older architectures
        # or future ones, rsqrtf() might include NR refinement, which would
        # make these tables wrong.  Verify by disassembling:
        #   nvcc -arch=sm_XX -cubin then cuobjdump --dump-sass
        gpu_out = torch.rsqrt(
            torch.tensor(floats, device='cuda')
        ).cpu().numpy()

        # Correctly-rounded rsqrt (via float64)
        exact = (1.0 / np.sqrt(floats.astype(np.float64))).astype(np.float32)

        # Signed ULP difference
        gpu_bits = gpu_out.view(np.int32)
        exact_bits = exact.view(np.int32)
        ulp_diff = gpu_bits - exact_bits

        # Sanity check: should be within ±2 ULP
        max_err = int(np.max(np.abs(ulp_diff)))
        assert max_err <= 2, f"MUFU.RSQ error > 2 ULP ({max_err}) — unexpected hardware"

        corrections[label] = ulp_diff.astype(np.int8)

        n_zero = int(np.sum(ulp_diff == 0))
        print(f"  {label} (biased_exp={biased_exp}): "
              f"{n_zero}/{N} exact, ±1: {N - n_zero - int(np.sum(np.abs(ulp_diff) >= 2))}, "
              f"±2: {int(np.sum(np.abs(ulp_diff) >= 2))}")

    return corrections["even"], corrections["odd"]


_fast_math_rcp_module = None

def _gpu_rcp_fast_math(inputs_np):
    """Run rcp.approx.ftz.f32 on GPU (actual MUFU.RCP instruction).

    torch's 1.0/x uses IEEE division (__fdiv_rn), which differs from
    MUFU.RCP. FA2 uses MUFU.RCP via 1.f/x compiled with --use_fast_math.
    """
    import torch
    global _fast_math_rcp_module
    if _fast_math_rcp_module is None:
        from torch.utils.cpp_extension import load_inline
        cuda_source = """
__global__ void mufu_rcp_kernel(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(in[i]));
        out[i] = r;
    }
}

torch::Tensor mufu_rcp(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32);
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mufu_rcp_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""
        cpp_source = "torch::Tensor mufu_rcp(torch::Tensor input);"
        _fast_math_rcp_module = load_inline(
            name='mufu_rcp_probe',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['--use_fast_math'],
            functions=['mufu_rcp'],
            verbose=False
        )
        print("  Compiled --use_fast_math RCP kernel")
    t = torch.tensor(inputs_np, device='cuda', dtype=torch.float32)
    return _fast_math_rcp_module.mufu_rcp(t).cpu().numpy()


def _probe_mufu_rcp(gpu_name):
    """
    Probe MUFU.RCP (reciprocal) for all 2^23 mantissa values.

    MUFU.RCP computes 1/x. Unlike RSQ, there is no exponent parity:
      rcp(2^e × 1.m) = 2^(-e) × rcp(1.m)
    The exponent is simply negated (exact), and the mantissa lookup
    depends only on m. One table suffices.

    WHY inline PTX instead of torch: 1.0/torch.tensor(x) compiles to
    IEEE FP32 division (__fdiv_rn), NOT the MUFU.RCP approximation.
    On A100, MUFU.RCP differs from IEEE on 13.2% of mantissa values.
    The original probe used torch division and found 0 corrections —
    it was comparing IEEE against IEEE, not measuring the hardware.
    FA2's normalize step (1.f/sum compiled with --use_fast_math) uses
    the actual MUFU.RCP instruction, so the corrections must match it.

    We probe at biased_exp=127 (x ∈ [1.0, 2.0)) and verify at biased_exp=128.

    Returns: corr — int8 array of shape (2^23,)

    HARDWARE-SPECIFIC: Different GPU architectures have different MUFU.RCP
    silicon. Tables must be re-probed for each architecture.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No GPU available to probe MUFU.RCP.  Either:\n"
            f"  1. Run this once on a {gpu_name} GPU to generate the tables, or\n"
            f"  2. Copy mufu_cache/{gpu_name}/ from a machine that has one."
        )

    detected = _detect_gpu_name()
    if detected and detected != gpu_name:
        print(f"  WARNING: probing on {detected} but cache says {gpu_name}")

    N = 2 ** 23
    mantissa_bits = np.arange(N, dtype=np.uint32)

    # Probe at biased_exp=127: x ∈ [1.0, 2.0)
    float_bits = np.uint32(127 << 23) | mantissa_bits
    floats = np.frombuffer(float_bits.tobytes(), dtype=np.float32)

    # GPU reciprocal via MUFU.RCP (PTX rcp.approx.ftz.f32)
    gpu_out = _gpu_rcp_fast_math(floats)

    # Correctly-rounded reciprocal (via float64)
    exact = (1.0 / floats.astype(np.float64)).astype(np.float32)

    # ULP difference
    gpu_bits = gpu_out.view(np.int32)
    exact_bits = exact.view(np.int32)
    ulp_diff = gpu_bits - exact_bits

    max_err = int(np.max(np.abs(ulp_diff)))
    assert max_err <= 2, f"MUFU.RCP error > 2 ULP ({max_err}) — unexpected hardware"

    n_zero = int(np.sum(ulp_diff == 0))
    n_pm1 = int(np.sum(np.abs(ulp_diff) == 1))
    n_pm2 = int(np.sum(np.abs(ulp_diff) >= 2))
    print(f"  biased_exp=127: {n_zero}/{N} exact, ±1: {n_pm1}, ±2: {n_pm2}")

    corr = ulp_diff.astype(np.int8)

    # Verify at biased_exp=128: x ∈ [2.0, 4.0)
    float_bits_v = np.uint32(128 << 23) | mantissa_bits
    floats_v = np.frombuffer(float_bits_v.tobytes(), dtype=np.float32)
    gpu_v = _gpu_rcp_fast_math(floats_v)
    exact_v = (1.0 / floats_v.astype(np.float64)).astype(np.float32)
    corr_v = (gpu_v.view(np.int32) - exact_v.view(np.int32)).astype(np.int8)
    match = int(np.sum(corr == corr_v))
    print(f"  verify exp=128 vs exp=127: {match}/{N} ({match/N*100:.1f}%)")

    return corr


_fast_math_ex2_module = None

def _gpu_exp2_fast_math(inputs_np):
    """Run exp2f with --use_fast_math on GPU (actual MUFU.EX2 instruction).

    torch.exp2 uses libdevice software exp2, which differs from the
    hardware MUFU.EX2 instruction that FA2 uses via exp2f + --use_fast_math.
    This compiles a small CUDA kernel with --use_fast_math to get the
    real hardware output.
    """
    import torch
    global _fast_math_ex2_module
    if _fast_math_ex2_module is None:
        from torch.utils.cpp_extension import load_inline
        cuda_source = """
__global__ void mufu_ex2_kernel(const float* __restrict__ in,
                                float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = exp2f(in[i]);
}

torch::Tensor mufu_ex2(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda() && input.dtype() == torch::kFloat32);
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    mufu_ex2_kernel<<<blocks, threads>>>(
        input.data_ptr<float>(), output.data_ptr<float>(), n);
    return output;
}
"""
        cpp_source = "torch::Tensor mufu_ex2(torch::Tensor input);"
        _fast_math_ex2_module = load_inline(
            name='mufu_ex2_probe',
            cpp_sources=cpp_source,
            cuda_sources=cuda_source,
            extra_cuda_cflags=['--use_fast_math'],
            functions=['mufu_ex2'],
            verbose=False
        )
        print("  Compiled --use_fast_math exp2 kernel")
    t = torch.tensor(inputs_np, device='cuda', dtype=torch.float32)
    return _fast_math_ex2_module.mufu_ex2(t).cpu().numpy()


def _probe_mufu_ex2(gpu_name):
    """
    Probe MUFU.EX2 for all 2^23 fraction values, positive and negative.

    MUFU.EX2 computes 2^x. The hardware uses trunc-toward-zero range reduction:
      n = trunc(x), f = x - n
      Positive x: f ∈ [0, 1)   → positive fraction table
      Negative x: f ∈ (-1, 0]  → negative fraction table

    Within each sign, the error depends only on the fraction (verified:
    consecutive integer offsets agree 100%, e.g. n=0 vs n=1, n=-1 vs n=-2).

    We probe at n=0 (positive, x ∈ [0, 1)) and n=-1 (negative, x ∈ [-1, 0)).

    Returns: (corr_pos, corr_neg) — int8 arrays of shape (2^23,)

    HARDWARE-SPECIFIC: Different GPU architectures have different MUFU.EX2
    silicon. Tables must be re-probed for each architecture.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"No GPU available to probe MUFU.EX2.  Either:\n"
            f"  1. Run this once on a {gpu_name} GPU to generate the tables, or\n"
            f"  2. Copy mufu_cache/{gpu_name}/ from a machine that has one."
        )

    detected = _detect_gpu_name()
    if detected and detected != gpu_name:
        print(f"  WARNING: probing on {detected} but cache says {gpu_name}")

    N = 2 ** 23
    frac_indices = np.arange(N, dtype=np.float64)
    fracs = frac_indices * (2.0 ** -23)  # f_i = i / 2^23, exactly representable

    corrections = {}

    for label, offset in [("positive", 0), ("negative", -1)]:
        # Construct inputs: x = offset + f_i
        # n=0:  x ∈ [0, 1)   — all exactly representable in FP32
        # n=-1: x ∈ [-1, 0)  — all exactly representable in FP32
        inputs = (np.float64(offset) + fracs).astype(np.float32)

        # GPU exp2 via MUFU.EX2 (--use_fast_math kernel)
        gpu_out = _gpu_exp2_fast_math(inputs)

        # IEEE correctly-rounded exp2 (via float64)
        exact = np.exp2(inputs.astype(np.float64)).astype(np.float32)

        # ULP difference
        gpu_bits = gpu_out.view(np.int32)
        exact_bits = exact.view(np.int32)
        ulp_diff = gpu_bits - exact_bits

        max_err = int(np.max(np.abs(ulp_diff)))
        assert max_err <= 2, f"MUFU.EX2 {label} error > 2 ULP ({max_err})"

        n_zero = int(np.sum(ulp_diff == 0))
        n_pm1 = int(np.sum(np.abs(ulp_diff) == 1))
        n_pm2 = int(np.sum(np.abs(ulp_diff) >= 2))
        print(f"  {label} (n={offset}): {n_zero}/{N} exact, ±1: {n_pm1}, ±2: {n_pm2}")

        corrections[label] = ulp_diff.astype(np.int8)

    # Verify consistency: n=0 vs n=1 (should be 100%)
    inputs_v = (np.float64(1.0) + fracs).astype(np.float32)
    gpu_v = _gpu_exp2_fast_math(inputs_v)
    exact_v = np.exp2(inputs_v.astype(np.float64)).astype(np.float32)
    corr_v = (gpu_v.view(np.int32) - exact_v.view(np.int32)).astype(np.int8)
    match = int(np.sum(corrections["positive"] == corr_v))
    print(f"  verify n=0 vs n=1: {match}/{N} ({match/N*100:.1f}%)")

    # Verify: n=-1 vs n=-2 (should be 100%)
    inputs_v2 = (np.float64(-2.0) + fracs).astype(np.float32)
    gpu_v2 = _gpu_exp2_fast_math(inputs_v2)
    exact_v2 = np.exp2(inputs_v2.astype(np.float64)).astype(np.float32)
    corr_v2 = (gpu_v2.view(np.int32) - exact_v2.view(np.int32)).astype(np.int8)
    match2 = int(np.sum(corrections["negative"] == corr_v2))
    print(f"  verify n=-1 vs n=-2: {match2}/{N} ({match2/N*100:.1f}%)")

    return corrections["positive"], corrections["negative"]


def _generate_ex2_full_table():
    """Generate a 4GB table: int8 correction for every possible float32 input.

    WHY a full 4B table instead of fraction-indexed 2^23:
    The hardware's EX2 correction depends only on the fractional part of x
    (verified: corrections at n=0 and n=5 match when properly aligned).
    However, a 2^23 fraction-indexed table fails for |x| < 1 because the
    fraction IS x itself, which has up to 24 significant mantissa bits —
    more than the 23-bit index can represent. Two distinct float32 values
    in [0,1) map to the same index, but the hardware distinguishes them.
    This caused ~32% error rate for inputs in [0, 1) and ~1% overall.
    A full uint32-indexed table eliminates all index logic and is exact
    for every possible input.

    Probes the actual MUFU.EX2 hardware (via PTX ex2.approx.ftz.f32)
    for all 2^32 possible float32 bit patterns. Stores the ULP difference
    between the hardware result and IEEE-correct exp2 as an int8 correction.

    For subnormal/overflow/NaN results, the int8 correction may be clamped
    and thus incorrect. This is safe because ex2() handles these cases
    explicitly after applying the correction:
    - Subnormal results: hardware returns 0 (FTZ). The clamped correction
      produces a wrong intermediate, but it stays in the subnormal range
      (max subnormal bits = 0x7FFFFF, minus 127 still has biased exp 0),
      so the FTZ check in ex2() overrides to 0.
    - Overflow/NaN: corrections for these are 0 (hw and exact agree on inf/nan).

    Returns: int8 array of shape (2^32,)
    """
    import torch

    # Compile PTX kernel
    _gpu_exp2_fast_math(np.array([0.0], dtype=np.float32))  # ensure kernel is compiled

    table = np.zeros(2**32, dtype=np.int8)
    chunk_size = 2**26  # 64M values per chunk — fits comfortably in GPU memory
    n_chunks = 2**32 // chunk_size

    import time
    t0 = time.time()
    for i in range(n_chunks):
        start = i * chunk_size
        end = start + chunk_size
        uint_vals = np.arange(start, end, dtype=np.uint32)
        float_vals = uint_vals.view(np.float32).copy()

        # Hardware MUFU.EX2
        hw_out = _gpu_exp2_fast_math(float_vals)

        # IEEE correctly-rounded exp2 (via float64)
        # Expected warnings on chunks containing large/NaN/inf bit patterns:
        # - overflow in exp2: exp2(3.4e38) overflows even float64 → inf → inf in fp32
        # - overflow in cast: float64 inf → float32 inf (fine)
        # - invalid in cast: NaN propagation (fine)
        # All produce inf/NaN in exact, matching the hardware output → correction = 0.
        # These inputs never occur in FA2 (attention scores are finite).
        exact = np.exp2(float_vals.astype(np.float64)).astype(np.float32)

        # Correction in ULPs
        hw_bits = hw_out.view(np.int32)
        exact_bits = exact.view(np.int32)
        corr = hw_bits.astype(np.int64) - exact_bits.astype(np.int64)

        # Clamp to int8: out-of-range means FTZ or special case, handled in code
        table[start:end] = np.clip(corr, -127, 127).astype(np.int8)

        if (i + 1) % 8 == 0:
            elapsed = time.time() - t0
            pct = (i + 1) / n_chunks * 100
            eta = elapsed / (i + 1) * (n_chunks - i - 1)
            print(f"    {pct:.0f}% ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"    Done ({elapsed:.0f}s)")
    return table


class MUFUEmulator:
    """
    Emulates MUFU special function unit instructions using probed correction tables.

    Supported instructions:
      - MUFU.RSQ: reciprocal square root (2 tables: even/odd exponent parity)
      - MUFU.EX2: base-2 exponential (1 table: indexed by fractional part)

    On first use, probes the GPU and caches the tables.
    On subsequent uses, loads from cache.

    HARDWARE-SPECIFIC: Different GPU architectures (V100, A100, H100, etc.)
    have different MUFU silicon and produce different corrections.  Tables
    must be re-probed for each architecture.
    """

    def __init__(self, gpu_name=None, cache_dir=CACHE_DIR):
        if gpu_name is None:
            gpu_name = _detect_gpu_name()
        if gpu_name is None:
            raise RuntimeError("Cannot detect GPU.  Pass gpu_name= explicitly.")

        self.gpu_name = gpu_name
        self.cache_path = os.path.join(cache_dir, gpu_name)

        # RSQ tables
        self.even_path = os.path.join(self.cache_path, "mufu_rsq_even.npy")
        self.odd_path = os.path.join(self.cache_path, "mufu_rsq_odd.npy")

        if os.path.exists(self.even_path) and os.path.exists(self.odd_path):
            self.corr_even = np.load(self.even_path)
            self.corr_odd = np.load(self.odd_path)
            print(f"MUFU.RSQ: loaded cached tables for {gpu_name}")
        else:
            print(f"MUFU.RSQ: no cache for {gpu_name}, probing hardware...")
            self.corr_even, self.corr_odd = _probe_mufu_rsq(gpu_name)
            os.makedirs(self.cache_path, exist_ok=True)
            np.save(self.even_path, self.corr_even)
            np.save(self.odd_path, self.corr_odd)
            print(f"MUFU.RSQ: cached to {self.cache_path}/")

        # EX2: full 4-billion-entry table (int8 correction for every float32 input)
        self.ex2_full_path = os.path.join(self.cache_path, "mufu_ex2_full.npy")

        if os.path.exists(self.ex2_full_path):
            try:
                self.ex2_full_table = np.load(self.ex2_full_path)
                if self.ex2_full_table.size != 2**32:
                    raise ValueError(f"corrupt table: {self.ex2_full_table.size} entries, expected {2**32}")
                print(f"MUFU.EX2: loaded full table for {gpu_name} ({self.ex2_full_table.nbytes / 1e9:.1f}GB)")
            except (ValueError, Exception) as e:
                print(f"MUFU.EX2: corrupt cache ({e}), regenerating...")
                os.remove(self.ex2_full_path)
                self.ex2_full_table = None

        if not hasattr(self, 'ex2_full_table') or self.ex2_full_table is None:
            print(f"MUFU.EX2: generating full table for {gpu_name} (4GB, ~60s)...")
            self.ex2_full_table = _generate_ex2_full_table()
            # Save to disk if possible, otherwise just keep in RAM.
            # np.save creates the file before writing — if disk is full,
            # it leaves an empty/corrupt file that crashes on next load.
            try:
                os.makedirs(self.cache_path, exist_ok=True)
                np.save(self.ex2_full_path, self.ex2_full_table)
                print(f"MUFU.EX2: cached to {self.ex2_full_path}")
            except Exception as e:
                # Clean up corrupt partial file
                if os.path.exists(self.ex2_full_path):
                    os.remove(self.ex2_full_path)
                print(f"MUFU.EX2: generated in RAM (save failed: {type(e).__name__}: {e})")

        # RCP table (single table, no exponent dependence)
        self.rcp_path = os.path.join(self.cache_path, "mufu_rcp.npy")

        if os.path.exists(self.rcp_path):
            self.corr_rcp = np.load(self.rcp_path)
            print(f"MUFU.RCP: loaded cached table for {gpu_name}")
        else:
            print(f"MUFU.RCP: no cache for {gpu_name}, probing hardware...")
            self.corr_rcp = _probe_mufu_rcp(gpu_name)
            os.makedirs(self.cache_path, exist_ok=True)
            np.save(self.rcp_path, self.corr_rcp)
            print(f"MUFU.RCP: cached to {self.cache_path}/")

    def rsq_scalar(self, x):
        """Emulate MUFU.RSQ for a single float32 value."""
        x = np.float32(x)
        bits = x.view(np.uint32)
        biased_exp = int((bits >> 23) & 0xFF)
        mantissa = int(bits & 0x7FFFFF)

        # Correctly-rounded rsqrt
        exact = np.float32(1.0 / np.sqrt(np.float64(x)))
        exact_bits = exact.view(np.int32)

        # Apply hardware correction based on biased exponent parity
        # "even" table was probed at biased_exp=127 (odd biased value)
        # "odd" table was probed at biased_exp=128 (even biased value)
        if biased_exp & 1:
            corr = np.int32(self.corr_even[mantissa])
        else:
            corr = np.int32(self.corr_odd[mantissa])

        result_bits = exact_bits + corr
        return result_bits.view(np.float32)

    def rsq(self, x_array):
        """
        Emulate MUFU.RSQ for an array of float32 values.
        Vectorized for performance.

        Args:
            x_array: numpy float32 array of any shape

        Returns:
            numpy float32 array, same shape, with MUFU.RSQ results
        """
        x = np.asarray(x_array, dtype=np.float32)
        orig_shape = x.shape
        x_flat = x.ravel()

        bits = x_flat.view(np.uint32)
        biased_exp = (bits >> 23) & 0xFF
        mantissa = (bits & 0x7FFFFF).astype(np.int64)  # index into LUT

        # Correctly-rounded rsqrt
        exact = (1.0 / np.sqrt(x_flat.astype(np.float64))).astype(np.float32)
        exact_bits = exact.view(np.int32)

        # Vectorized correction lookup
        corr = np.where(
            biased_exp & 1,
            self.corr_even[mantissa],  # biased odd → "even" table
            self.corr_odd[mantissa]    # biased even → "odd" table
        ).astype(np.int32)

        result_bits = exact_bits + corr
        return result_bits.view(np.float32).reshape(orig_shape)

    def ex2(self, x_array):
        """
        Emulate MUFU.EX2 for an array of float32 values.

        Uses a full 4-billion-entry table indexed by raw uint32 bits.
        No bit extraction, no fraction logic — just a direct lookup.

        WHY this approach: The hardware's correction depends on the full
        float32 representation, not just the fractional part. For |x| >= 1,
        a 2^23 fraction table works (verified). For |x| < 1, the mantissa
        carries more precision than the fraction index can capture. Rather
        than handling two regimes, we use one table for all inputs.
        """
        x = np.asarray(x_array, dtype=np.float32)
        orig_shape = x.shape
        x_flat = x.ravel()

        # IEEE correctly-rounded exp2 (via float64)
        exact = np.exp2(x_flat.astype(np.float64)).astype(np.float32)
        exact_bits = exact.view(np.int32).copy()

        # Look up correction by raw float32 bits
        idx = x_flat.view(np.uint32)
        corr = self.ex2_full_table[idx].astype(np.int32)
        result_bits = exact_bits + corr

        # FTZ: hardware flushes subnormal results to zero.
        # This check is LOAD-BEARING for correctness: when exp2(x) is subnormal,
        # the int8 correction may be clamped (true correction exceeds ±127),
        # making result_bits wrong. But the clamped result stays in the subnormal
        # range (max subnormal = 0x7FFFFF, minus 127 still has biased exp 0),
        # so this check catches it and overrides to 0.
        result = result_bits.view(np.float32)
        is_subnorm = (result_bits & 0x7F800000) == 0
        is_nonzero = result_bits & 0x7FFFFFFF
        result = np.where(is_subnorm & is_nonzero, np.float32(0.0), result)

        return result.reshape(orig_shape)

    def rcp(self, x_array):
        """
        Emulate MUFU.RCP for an array of float32 values.
        Vectorized for performance.

        MUFU.RCP computes 1/x. The error depends only on the mantissa
        (the exponent is simply negated, which is exact).
        rcp(2^e × 1.m) = 2^(-e) × rcp(1.m)

        Args:
            x_array: numpy float32 array of any shape

        Returns:
            numpy float32 array, same shape, with MUFU.RCP results
        """
        x = np.asarray(x_array, dtype=np.float32)
        orig_shape = x.shape
        x_flat = x.ravel()

        bits = x_flat.view(np.uint32)
        mantissa = (bits & 0x7FFFFF).astype(np.int64)  # index into LUT

        # Correctly-rounded reciprocal (via float64)
        exact = (1.0 / x_flat.astype(np.float64)).astype(np.float32)
        exact_bits = exact.view(np.int32).copy()

        # Apply correction (mantissa-only, no exponent dependence)
        corr = self.corr_rcp[mantissa].astype(np.int32)
        result_bits = exact_bits + corr

        # Handle special cases
        special = ~np.isfinite(x_flat) | (x_flat == 0)
        if np.any(special):
            result_bits[special] = exact_bits[special]

        return result_bits.view(np.float32).reshape(orig_shape)


if __name__ == "__main__":
    # Quick verification against saved GPU data
    import struct

    mufu = MUFUEmulator()

    if os.path.exists("ffn_chain_data/gpu_rms_rsqrt.bin"):
        var = np.fromfile("ffn_chain_data/gpu_rms_variance.bin", dtype=np.float32).reshape(-1, 1)
        gpu_rsqrt = np.fromfile("ffn_chain_data/gpu_rms_rsqrt.bin", dtype=np.float32).reshape(-1, 1)
        eps = np.float32(1e-6)
        vpe = (var + eps).astype(np.float32)

        emu = mufu.rsq(vpe)
        M = len(var)
        match = int(np.sum(emu.view(np.uint32) == gpu_rsqrt.view(np.uint32)))
        print(f"Verification: {match}/{M} FP32 bit-exact")
    else:
        print("No ffn_chain_data/ found, skipping verification")
        print("Run: /venv/main/bin/python mufu_emulator.py")
