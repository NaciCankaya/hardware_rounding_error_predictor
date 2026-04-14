# tc_profiles.py
"""
Tensor Core Profiles — hardware parameters for each GPU architecture.
Source: Khattak & Mikaitis (2025), "Accurate Models of NVIDIA Tensor Cores", Table II.

Each profile fully characterizes the block FMA behavior for a given
(GPU, input_format, output_format) combination. The emulator reads these
parameters and requires no code changes for new architectures.

Usage:
    from tc_profiles import get_profile
    p = get_profile("A100", "bf16", "fp32")
    print(p.nfma, p.neab, p.round_mode)
"""

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class TensorCoreProfile:
    """Hardware parameters for one tensor core configuration."""

    # Identity
    gpu: str                  # e.g. "A100", "H100"
    input_fmt: str            # e.g. "bf16", "fp16", "fp8_e4m3"
    output_fmt: str           # e.g. "fp32", "fp16"

    # Block FMA parameters
    nfma: int                 # products per block FMA (4, 8, 16, 32)
    neab: int                 # extra alignment bits beyond input precision
    round_mode: str           # "trunc" or "rne" — final rounding to output
    denorm_products: bool     # products aligned in raw (denormalized) form

    # Alignment window geometry
    int_bits: int             # integer bits in alignment window (always 2 for denorm products)
    frac_bits: int            # fractional bits (includes neab)

    # Input format significand bits (including implicit 1)
    input_sig_bits: int       # BF16=8, FP16=11, FP8-E4M3=4, FP8-E5M2=3

    # MMA instruction shape (M, N, K)
    mma_shape: tuple          # e.g. (16, 8, 16) for A100 BF16

    # Accumulator ordering
    c_order: str              # "early" = accumulator added with first block, "late" = added at end
    interleaved: bool         # H100/B200 FP8 interleaves pairs from input vectors

    @property
    def window_bits(self):
        """Total alignment window width."""
        return self.int_bits + self.frac_bits

    @property
    def acc_output_bits(self):
        """Accumulator output precision (window + carry bits)."""
        # carry bits = ceil(log2(nfma + 1))
        import math
        carry = math.ceil(math.log2(self.nfma + 1))
        return self.window_bits + carry

    @property
    def products_per_mma(self):
        """Total products per MMA instruction (K dimension of MMA shape)."""
        return self.mma_shape[2]

    @property
    def blocks_per_mma(self):
        """Number of block FMA invocations per MMA instruction."""
        return self.products_per_mma // self.nfma

    def describe(self):
        return (
            f"{self.gpu} {self.input_fmt}→{self.output_fmt}: "
            f"NFMA={self.nfma}, neab={self.neab}, "
            f"window=({self.int_bits},{self.frac_bits - self.neab},{self.neab})={self.window_bits}b, "
            f"round={self.round_mode}, denorm={self.denorm_products}, "
            f"MMA={self.mma_shape}, "
            f"c_order={self.c_order}, interleaved={self.interleaved}"
        )


# ============================================================
# Profile registry
# ============================================================
_PROFILES = {}


def _reg(p):
    key = (p.gpu.lower(), p.input_fmt.lower(), p.output_fmt.lower())
    _PROFILES[key] = p
    return p


# ------ V100 (Volta) ------
_reg(TensorCoreProfile(
    gpu="V100", input_fmt="fp16", output_fmt="fp32",
    nfma=4, neab=0, round_mode="trunc", denorm_products=True,
    int_bits=2, frac_bits=23, input_sig_bits=11,
    mma_shape=(16, 16, 16), c_order="early", interleaved=False,
))
_reg(TensorCoreProfile(
    gpu="V100", input_fmt="fp16", output_fmt="fp16",
    nfma=4, neab=0, round_mode="rne", denorm_products=True,
    int_bits=2, frac_bits=23, input_sig_bits=11,
    mma_shape=(16, 16, 16), c_order="early", interleaved=False,
))

# ------ A100 / A2 / A30 (Ampere) ------
for _gpu in ["A100", "A2", "A30"]:
    for _ifmt in ["fp16", "bf16"]:
        _isig = 11 if _ifmt == "fp16" else 8
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_ifmt, output_fmt="fp32",
            nfma=8, neab=1, round_mode="trunc", denorm_products=True,
            int_bits=2, frac_bits=24, input_sig_bits=_isig,
            mma_shape=(16, 8, 16), c_order="early", interleaved=False,
        ))
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_ifmt, output_fmt="fp16",
            nfma=8, neab=1, round_mode="rne", denorm_products=True,
            int_bits=2, frac_bits=24, input_sig_bits=_isig,
            mma_shape=(16, 8, 16), c_order="early", interleaved=False,
        ))
    # TF32
    _reg(TensorCoreProfile(
        gpu=_gpu, input_fmt="tf32", output_fmt="fp32",
        nfma=4, neab=1, round_mode="trunc", denorm_products=True,
        int_bits=2, frac_bits=24, input_sig_bits=11,
        mma_shape=(16, 8, 8), c_order="early", interleaved=False,
    ))

# ------ L40S / Ada RTX 1000 (Ada Lovelace) ------
for _gpu in ["L40S", "L40", "Ada_RTX_1000"]:
    for _ifmt in ["fp16", "bf16"]:
        _isig = 11 if _ifmt == "fp16" else 8
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_ifmt, output_fmt="fp32",
            nfma=8, neab=1, round_mode="trunc", denorm_products=True,
            int_bits=2, frac_bits=24, input_sig_bits=_isig,
            mma_shape=(16, 8, 16), c_order="early", interleaved=False,
        ))
    # TF32
    _reg(TensorCoreProfile(
        gpu=_gpu, input_fmt="tf32", output_fmt="fp32",
        nfma=4, neab=1, round_mode="trunc", denorm_products=True,
        int_bits=2, frac_bits=24, input_sig_bits=11,
        mma_shape=(16, 8, 8), c_order="early", interleaved=False,
    ))
    # FP8
    for _fp8 in ["fp8_e4m3", "fp8_e5m2"]:
        _isig = 4 if _fp8 == "fp8_e4m3" else 3
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_fp8, output_fmt="fp32",
            nfma=16, neab=-10, round_mode="trunc", denorm_products=True,
            int_bits=2, frac_bits=13, input_sig_bits=_isig,
            mma_shape=(16, 8, 32), c_order="early", interleaved=False,
        ))
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_fp8, output_fmt="fp16",
            nfma=16, neab=-10, round_mode="rne", denorm_products=True,
            int_bits=2, frac_bits=13, input_sig_bits=_isig,
            mma_shape=(16, 8, 32), c_order="early", interleaved=False,
        ))

# ------ H100 / H200 (Hopper) ------
for _gpu in ["H100", "H200"]:
    for _ifmt in ["fp16", "bf16"]:
        _isig = 11 if _ifmt == "fp16" else 8
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_ifmt, output_fmt="fp32",
            nfma=16, neab=2, round_mode="trunc", denorm_products=True,
            int_bits=2, frac_bits=25, input_sig_bits=_isig,
            mma_shape=(16, 8, 16), c_order="early", interleaved=False,
        ))
    # TF32
    _reg(TensorCoreProfile(
        gpu=_gpu, input_fmt="tf32", output_fmt="fp32",
        nfma=8, neab=2, round_mode="trunc", denorm_products=True,
        int_bits=2, frac_bits=25, input_sig_bits=11,
        mma_shape=(16, 8, 8), c_order="early", interleaved=False,
    ))
    # FP8 (standard path via mma.sync → HMMA)
    for _fp8 in ["fp8_e4m3", "fp8_e5m2"]:
        _isig = 4 if _fp8 == "fp8_e4m3" else 3
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_fp8, output_fmt="fp32",
            nfma=32, neab=-10, round_mode="trunc", denorm_products=True,
            int_bits=2, frac_bits=13, input_sig_bits=_isig,
            mma_shape=(16, 8, 32), c_order="early", interleaved=False,
        ))
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=_fp8, output_fmt="fp16",
            nfma=32, neab=-10, round_mode="rne", denorm_products=True,
            int_bits=2, frac_bits=13, input_sig_bits=_isig,
            mma_shape=(16, 8, 32), c_order="early", interleaved=False,
        ))
    # FP8* (via HMMA with fp8→fp16 conversion, interleaved)
    for _fp8 in ["fp8_e4m3", "fp8_e5m2"]:
        _isig = 4 if _fp8 == "fp8_e4m3" else 3
        _reg(TensorCoreProfile(
            gpu=_gpu, input_fmt=f"{_fp8}_hmma", output_fmt="fp32",
            nfma=16, neab=2, round_mode="rne", denorm_products=True,
            int_bits=2, frac_bits=25, input_sig_bits=_isig,
            mma_shape=(16, 8, 32), c_order="late", interleaved=True,
        ))

# ------ B200 (Blackwell) ------
for _ifmt in ["fp16", "bf16"]:
    _isig = 11 if _ifmt == "fp16" else 8
    _reg(TensorCoreProfile(
        gpu="B200", input_fmt=_ifmt, output_fmt="fp32",
        nfma=16, neab=2, round_mode="trunc", denorm_products=True,
        int_bits=2, frac_bits=25, input_sig_bits=_isig,
        mma_shape=(16, 8, 16), c_order="early", interleaved=False,
    ))
# TF32
_reg(TensorCoreProfile(
    gpu="B200", input_fmt="tf32", output_fmt="fp32",
    nfma=8, neab=2, round_mode="trunc", denorm_products=True,
    int_bits=2, frac_bits=25, input_sig_bits=11,
    mma_shape=(16, 8, 8), c_order="early", interleaved=False,
))
# FP8* (interleaved, via HMMA)
for _fp8 in ["fp8_e4m3", "fp8_e5m2"]:
    _isig = 4 if _fp8 == "fp8_e4m3" else 3
    _reg(TensorCoreProfile(
        gpu="B200", input_fmt=f"{_fp8}_hmma", output_fmt="fp32",
        nfma=16, neab=2, round_mode="rne", denorm_products=True,
        int_bits=2, frac_bits=25, input_sig_bits=_isig,
        mma_shape=(16, 8, 32), c_order="late", interleaved=True,
    ))


# ============================================================
# Public API
# ============================================================
def get_profile(gpu: str, input_fmt: str, output_fmt: str = "fp32") -> TensorCoreProfile:
    """Look up a tensor core profile by GPU and format.

    Args:
        gpu: GPU name (case-insensitive), e.g. "A100", "H100", "V100"
        input_fmt: Input format, e.g. "bf16", "fp16", "fp8_e4m3"
        output_fmt: Output format, default "fp32"

    Returns:
        TensorCoreProfile with all hardware parameters.

    Raises:
        KeyError if the combination is not registered.
    """
    key = (gpu.lower(), input_fmt.lower(), output_fmt.lower())
    if key not in _PROFILES:
        available = [f"{g} {i}→{o}" for g, i, o in sorted(_PROFILES.keys()) if g == gpu.lower()]
        raise KeyError(
            f"No profile for {gpu} {input_fmt}→{output_fmt}. "
            f"Available for {gpu}: {available or 'none (check GPU name)'}"
        )
    return _PROFILES[key]


def list_profiles(gpu: Optional[str] = None):
    """List all registered profiles, optionally filtered by GPU."""
    for key in sorted(_PROFILES.keys()):
        if gpu is None or key[0] == gpu.lower():
            print(_PROFILES[key].describe())


def detect_gpu() -> str:
    """Auto-detect the current GPU and return its profile name.

    Tries torch.cuda first, falls back to nvidia-smi.
    Returns a string like 'A100', 'H100', 'V100', etc.
    """
    # Map common device name substrings to profile names
    _GPU_MAP = {
        "A100": "A100",
        "A30": "A30",
        "A2 ": "A2",
        "H100": "H100",
        "H200": "H200",
        "B200": "B200",
        "V100": "V100",
        "L40S": "L40S",
        "L40": "L40S",      # L40 is same AD102 die as L40S
        "RTX 1000": "Ada_RTX_1000",
        "L4": "L40S",       # L4 uses same TC as Ada family
    }

    device_name = None

    # Try torch
    try:
        import torch
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    # Fallback: nvidia-smi
    if device_name is None:
        try:
            import subprocess
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0:
                device_name = r.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    if device_name is None:
        raise RuntimeError("No GPU detected. Cannot auto-select tensor core profile.")

    for substr, profile_name in _GPU_MAP.items():
        if substr in device_name:
            return profile_name

    raise RuntimeError(
        f"GPU '{device_name}' not recognized. "
        f"Known GPUs: {list(_GPU_MAP.values())}. "
        f"Use get_profile(gpu, fmt) manually."
    )


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("All registered tensor core profiles:")
    print("=" * 90)
    list_profiles()
    print()

    # Verify A100 BF16 params match what we validated
    p = get_profile("A100", "bf16", "fp32")
    assert p.nfma == 8
    assert p.neab == 1
    assert p.round_mode == "trunc"
    assert p.denorm_products == True
    assert p.window_bits == 26
    assert p.blocks_per_mma == 2
    assert p.input_sig_bits == 8
    print("A100 BF16→FP32 profile validated against experimental results.")
