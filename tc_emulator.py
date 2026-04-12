# tc_emulator.py
"""
Modular Tensor Core Emulator — Optimized

Generates and compiles a C emulator parameterized by a TensorCoreProfile.
No hardcoded GPU-specific constants — everything comes from the profile.

Optimizations (all numerics-preserving):
  1. OpenMP parallelization across output elements
  2. Pre-extracted input format parts (sign, exp, sig) — avoids redundant
     bit extraction in the inner loop
  3. B matrix transposed for cache-contiguous K-walk

The accumulation order within each output element is unchanged:
sequential K-walk in blocks of NFMA, same block FMA function.

Usage:
    from tc_profiles import get_profile
    from tc_emulator import TensorCoreEmulator

    profile = get_profile("A100", "bf16", "fp32")
    emu = TensorCoreEmulator(profile)
    C = emu.matmul(A_bf16, B_bf16)       # raw FP32 accumulator output
    C_bf16 = emu.matmul_bf16(A_bf16, B_bf16)  # with BF16 epilogue
"""

import numpy as np
import ctypes
import subprocess
import os
from tc_profiles import TensorCoreProfile, get_profile

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def _generate_c_source(profile: TensorCoreProfile) -> str:
    """Generate optimized C emulator source parameterized by profile."""

    neab = profile.neab
    window = profile.window_bits
    nfma = profile.nfma
    input_sig_bits = profile.input_sig_bits
    k_per_mma = profile.products_per_mma
    use_trunc = 1 if profile.round_mode == "trunc" else 0

    return r"""
#include <stdint.h>
#include <stdlib.h>
#include <math.h>

/* ================================================================
 * Profile parameters (from tc_profiles.py)
 * ================================================================ */
#define PARAM_NEAB      """ + str(neab) + r"""
#define PARAM_WINDOW    """ + str(window) + r"""
#define PARAM_NFMA      """ + str(nfma) + r"""
#define PARAM_SIG_BITS  """ + str(input_sig_bits) + r"""
#define PARAM_K_PER_MMA """ + str(k_per_mma) + r"""
#define PARAM_USE_TRUNC """ + str(use_trunc) + r"""
#define SHIFT_CONST     (24 - 2 * (PARAM_SIG_BITS - 1))
#define MANT_BITS       (PARAM_SIG_BITS - 1)

typedef union { float f; uint32_t u; } fp32_t;

/* Single-instruction MSB finding (replaces while loops) */
static inline int msb32(uint32_t x) { return 31 - __builtin_clz(x); }     /* x must be > 0 */
static inline int msb64(uint64_t x) { return 63 - __builtin_clzll(x); }   /* x must be > 0 */

/* ================================================================
 * FP32 helpers (for accumulator)
 * ================================================================ */
static void fp32_parts(float v, int *sign, int *exp, uint32_t *sig) {
    fp32_t fb; fb.f = v;
    *sign = (fb.u >> 31) & 1;
    int biased = (fb.u >> 23) & 0xFF;
    uint32_t frac = fb.u & 0x7FFFFF;
    if (biased == 0) {
        if (frac == 0) { *exp = -150; *sig = 0; return; }
        *exp = -126; *sig = frac;
    } else if (biased == 255) {
        *exp = 128; *sig = frac; return;
    } else {
        *exp = biased - 127;
        *sig = frac | 0x800000;
    }
}

/* Align a normal FP32 value (accumulator) */
static int64_t align_fp32(float v, int max_exp) {
    int sign, exp; uint32_t sig;
    fp32_parts(v, &sign, &exp, &sig);
    if (sig == 0) return 0;
    int msb_pos = 24 + (exp - max_exp);
    int lsb_pos = msb_pos - 23;
    if (msb_pos < 0) return 0;
    int64_t aligned;
    if (lsb_pos >= 0) aligned = (int64_t)sig << lsb_pos;
    else {
        int trunc = -lsb_pos;
        if (trunc >= 24) return 0;
        aligned = (int64_t)(sig >> trunc);
    }
    aligned &= ((int64_t)1 << PARAM_WINDOW) - 1;
    return sign ? -aligned : aligned;
}

/* Align a raw (denormalized) product */
static int64_t align_raw_product(int sign, int raw_exp, uint32_t raw_sig, int max_exp) {
    if (raw_sig == 0) return 0;
    int shift = raw_exp - max_exp + SHIFT_CONST;
    int msb = msb32(raw_sig);
    if (shift + msb < 0) return 0;
    int64_t aligned;
    if (shift >= 0) {
        aligned = (int64_t)raw_sig << shift;
    } else {
        int trunc = -shift;
        if (trunc > msb) return 0;
        aligned = (int64_t)(raw_sig >> trunc);
    }
    aligned &= ((int64_t)1 << PARAM_WINDOW) - 1;
    return sign ? -aligned : aligned;
}

/* ================================================================
 * Fixed-point to FP32 conversion
 * ================================================================ */
static float fixed_to_fp32_trunc(int64_t sum, int max_exp) {
    if (sum == 0) return 0.0f;
    int sign = 0; uint64_t mag;
    if (sum < 0) { sign = 1; mag = (uint64_t)(-sum); }
    else { mag = (uint64_t)sum; }
    int msb = msb64(mag);
    int result_exp = max_exp + (msb - 24);
    uint32_t result_sig;
    if (msb > 23) result_sig = (uint32_t)(mag >> (msb - 23));
    else result_sig = (uint32_t)(mag << (23 - msb));
    result_sig &= 0x7FFFFF;
    int biased = result_exp + 127;
    if (biased <= 0) return 0.0f;
    if (biased >= 255) { fp32_t fb; fb.u = ((uint32_t)sign << 31) | 0x7F800000; return fb.f; }
    fp32_t fb;
    fb.u = ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | result_sig;
    return fb.f;
}

static float fixed_to_fp32_rne(int64_t sum, int max_exp) {
    if (sum == 0) return 0.0f;
    int sign = 0; uint64_t mag;
    if (sum < 0) { sign = 1; mag = (uint64_t)(-sum); }
    else { mag = (uint64_t)sum; }
    int msb = msb64(mag);
    int result_exp = max_exp + (msb - 24);
    uint32_t result_sig;
    if (msb > 23) {
        int shift = msb - 23;
        result_sig = (uint32_t)(mag >> shift);
        uint64_t discarded = mag & (((uint64_t)1 << shift) - 1);
        uint64_t halfway = (uint64_t)1 << (shift - 1);
        if (discarded > halfway || (discarded == halfway && (result_sig & 1))) {
            result_sig++;
            if (result_sig >= 0x1000000) { result_sig >>= 1; result_exp++; }
        }
    } else {
        result_sig = (uint32_t)(mag << (23 - msb));
    }
    result_sig &= 0x7FFFFF;
    int biased = result_exp + 127;
    if (biased <= 0) return 0.0f;
    if (biased >= 255) { fp32_t fb; fb.u = ((uint32_t)sign << 31) | 0x7F800000; return fb.f; }
    fp32_t fb;
    fb.u = ((uint32_t)sign << 31) | ((uint32_t)biased << 23) | result_sig;
    return fb.f;
}

static float fixed_to_fp32(int64_t sum, int max_exp) {
#if PARAM_USE_TRUNC
    return fixed_to_fp32_trunc(sum, max_exp);
#else
    return fixed_to_fp32_rne(sum, max_exp);
#endif
}

/* ================================================================
 * Block FMA using pre-extracted parts
 * No per-element extraction in the hot loop.
 * ================================================================ */
static float hw_block_fma_pre(
    float acc,
    const int8_t *a_signs, const int16_t *a_exps, const uint16_t *a_sigs,
    const int8_t *b_signs, const int16_t *b_exps, const uint16_t *b_sigs,
    int n
) {
    int prod_signs[PARAM_NFMA];
    int raw_exps[PARAM_NFMA];
    uint32_t raw_sigs[PARAM_NFMA];

    for (int i = 0; i < n; i++) {
        prod_signs[i] = a_signs[i] ^ b_signs[i];
        raw_exps[i] = (int)a_exps[i] + (int)b_exps[i];
        raw_sigs[i] = (uint32_t)a_sigs[i] * (uint32_t)b_sigs[i];
    }

    int max_exp = -200;
    { int s,e; uint32_t sig; fp32_parts(acc, &s, &e, &sig); if (sig && e > max_exp) max_exp = e; }
    for (int i = 0; i < n; i++)
        if (raw_sigs[i] && raw_exps[i] > max_exp) max_exp = raw_exps[i];
    if (max_exp < -200) return 0.0f;

    int64_t sum = align_fp32(acc, max_exp);
    for (int i = 0; i < n; i++)
        sum += align_raw_product(prod_signs[i], raw_exps[i], raw_sigs[i], max_exp);

    return fixed_to_fp32(sum, max_exp);
}

/* ================================================================
 * Bulk extraction: input-format parts for an entire matrix
 * Called once before the matmul loops.
 * ================================================================ */
void extract_parts(
    const float *vals, int count,
    int8_t *signs, int16_t *exps, uint16_t *sigs
) {
    for (int i = 0; i < count; i++) {
        fp32_t fb; fb.f = vals[i];
        signs[i] = (int8_t)((fb.u >> 31) & 1);
        int biased = (fb.u >> 23) & 0xFF;
        uint32_t frac23 = fb.u & 0x7FFFFF;
        uint32_t mant = (frac23 >> (23 - MANT_BITS)) & ((1u << MANT_BITS) - 1);
        if (biased == 0) {
            exps[i] = -126;
            sigs[i] = (uint16_t)mant;
        } else if (biased == 255) {
            exps[i] = 128;
            sigs[i] = (uint16_t)mant;
        } else {
            exps[i] = (int16_t)(biased - 127);
            sigs[i] = (uint16_t)((1u << MANT_BITS) | mant);
        }
    }
}

/* ================================================================
 * Optimized matmul with:
 *   - Pre-extracted input parts (passed in from Python)
 *   - B transposed to [N,K] for contiguous K-access
 *   - OpenMP parallelization across output elements
 *
 * A: [M, K] row-major (pre-extracted as a_sign/exp/sig[M*K])
 * B_t: [N, K] row-major (pre-extracted as bt_sign/exp/sig[N*K])
 * C: [M, N] output
 * ================================================================ */
void tc_matmul_opt(
    const int8_t *a_sign, const int16_t *a_exp, const uint16_t *a_sig,
    const int8_t *bt_sign, const int16_t *bt_exp, const uint16_t *bt_sig,
    float *C,
    int M, int K, int N
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            /* A row i: a_*[i*K + k], B_t row j: bt_*[j*K + k] */
            const int8_t  *a_s = &a_sign[i * K];
            const int16_t *a_e = &a_exp[i * K];
            const uint16_t *a_g = &a_sig[i * K];
            const int8_t  *b_s = &bt_sign[j * K];
            const int16_t *b_e = &bt_exp[j * K];
            const uint16_t *b_g = &bt_sig[j * K];

            for (int k = 0; k < K; k += PARAM_K_PER_MMA) {
                int k_end_mma = k + PARAM_K_PER_MMA;
                if (k_end_mma > K) k_end_mma = K;

                for (int kb = k; kb < k_end_mma; kb += PARAM_NFMA) {
                    int end = kb + PARAM_NFMA;
                    if (end > k_end_mma) end = k_end_mma;
                    int n = end - kb;
                    acc = hw_block_fma_pre(
                        acc,
                        &a_s[kb], &a_e[kb], &a_g[kb],
                        &b_s[kb], &b_e[kb], &b_g[kb],
                        n
                    );
                }
            }
            C[i * N + j] = acc;
        }
    }
}

/* ================================================================
 * Non-optimized matmul (for reference / fallback)
 * Takes raw float arrays, extracts parts internally.
 * ================================================================ */
static void input_parts(float v, int *sign, int *exp_unbiased, uint32_t *sig_n) {
    fp32_t fb; fb.f = v;
    *sign = (fb.u >> 31) & 1;
    int biased = (fb.u >> 23) & 0xFF;
    uint32_t frac23 = fb.u & 0x7FFFFF;
    uint32_t mant = (frac23 >> (23 - MANT_BITS)) & ((1u << MANT_BITS) - 1);
    if (biased == 0) { *exp_unbiased = -126; *sig_n = mant; }
    else if (biased == 255) { *exp_unbiased = 128; *sig_n = mant; }
    else { *exp_unbiased = biased - 127; *sig_n = (1u << MANT_BITS) | mant; }
}

static float hw_block_fma(float acc, const float *a_vals, const float *b_vals, int n) {
    int prod_signs[PARAM_NFMA];
    int raw_exps[PARAM_NFMA];
    uint32_t raw_sigs[PARAM_NFMA];
    for (int i = 0; i < n; i++) {
        int sa, ea; uint32_t siga;
        int sb, eb; uint32_t sigb;
        input_parts(a_vals[i], &sa, &ea, &siga);
        input_parts(b_vals[i], &sb, &eb, &sigb);
        prod_signs[i] = sa ^ sb;
        raw_exps[i] = ea + eb;
        raw_sigs[i] = siga * sigb;
    }
    int max_exp = -200;
    { int s,e; uint32_t sig; fp32_parts(acc, &s, &e, &sig); if (sig && e > max_exp) max_exp = e; }
    for (int i = 0; i < n; i++)
        if (raw_sigs[i] && raw_exps[i] > max_exp) max_exp = raw_exps[i];
    if (max_exp < -200) return 0.0f;
    int64_t sum = align_fp32(acc, max_exp);
    for (int i = 0; i < n; i++)
        sum += align_raw_product(prod_signs[i], raw_exps[i], raw_sigs[i], max_exp);
    return fixed_to_fp32(sum, max_exp);
}

/* ================================================================
 * Batch block FMA: apply one block FMA to M*N independent accumulators.
 * acc[M,N] += A_block[M,n] * B_block[n,N]  (one NFMA-wide step)
 *
 * This is the hardware primitive — no K-walk, no matmul logic.
 * The caller manages the K-dimension iteration.
 *
 * WHY this exists: FA2's gemm_rs() accumulates P*V products into
 * the running O_acc register across KV tiles. The MMA instruction's
 * C operand IS O_acc — it does NOT start from zero. The standard
 * matmul() function always starts from zero, which gives different
 * FP32 results because the block FMA alignment window shifts when
 * the accumulator is nonzero. This primitive lets the FA2 emulator
 * pass O_acc as the initial accumulator, matching the hardware.
 * (Verified by reading gemm_rs in FA2's utils.h: cute::gemm passes
 * acc_o directly, no clear(acc).)
 *
 * matmul() is still correct for standalone GEMMs (Q/K/V/O projections)
 * where the hardware also starts from zero.
 * ================================================================ */
void tc_block_fma_batch(
    float *acc, const float *A_block, const float *B_block,
    int M, int n, int N
) {
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float a_vals[PARAM_NFMA], b_vals[PARAM_NFMA];
            for (int p = 0; p < n; p++) {
                a_vals[p] = A_block[i * n + p];
                b_vals[p] = B_block[p * N + j];
            }
            acc[i * N + j] = hw_block_fma(acc[i * N + j], a_vals, b_vals, n);
        }
    }
}

void tc_matmul(const float *A, const float *B, float *C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float acc = 0.0f;
            for (int k = 0; k < K; k += PARAM_K_PER_MMA) {
                int k_end_mma = k + PARAM_K_PER_MMA;
                if (k_end_mma > K) k_end_mma = K;
                for (int kb = k; kb < k_end_mma; kb += PARAM_NFMA) {
                    int end = kb + PARAM_NFMA;
                    if (end > k_end_mma) end = k_end_mma;
                    int n = end - kb;
                    float a_block[PARAM_NFMA], b_block[PARAM_NFMA];
                    for (int p = 0; p < n; p++) {
                        a_block[p] = A[i*K + kb + p];
                        b_block[p] = B[(kb + p)*N + j];
                    }
                    acc = hw_block_fma(acc, a_block, b_block, n);
                }
            }
            C[i*N + j] = acc;
        }
    }
}
"""


class TensorCoreEmulator:
    """Compiled C emulator for a specific tensor core profile."""

    def __init__(self, profile: TensorCoreProfile):
        self.profile = profile
        self._lib = None
        self._fn_slow = None
        self._fn_opt = None
        self._fn_extract = None
        self._fn_bfma = None
        self._compile()

    def _compile(self):
        src = _generate_c_source(self.profile)
        gpu = self.profile.gpu.lower()
        fmt = self.profile.input_fmt.lower()

        src_path = f"/tmp/_tc_emu_{gpu}_{fmt}.c"
        lib_path = f"/tmp/_tc_emu_{gpu}_{fmt}.so"

        with open(src_path, "w") as f:
            f.write(src)

        # Try with OpenMP first, fall back without
        r = subprocess.run(
            ["gcc", "-O3", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
             "-fopenmp", "-lm", "-o", lib_path, src_path],
            capture_output=True, text=True
        )
        if r.returncode != 0:
            # Retry without OpenMP
            r = subprocess.run(
                ["gcc", "-O3", "-shared", "-fPIC", "-ffp-contract=off", "-fno-fast-math",
                 "-lm", "-o", lib_path, src_path],
                capture_output=True, text=True
            )
            if r.returncode != 0:
                raise RuntimeError(f"Compile failed for {self.profile.gpu}: {r.stderr}")

        self._lib = ctypes.CDLL(lib_path)

        # Slow path (reference, takes raw floats)
        self._fn_slow = self._lib.tc_matmul
        self._fn_slow.restype = None
        self._fn_slow.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3

        # Fast path (pre-extracted parts + transposed B)
        self._fn_opt = self._lib.tc_matmul_opt
        self._fn_opt.restype = None
        self._fn_opt.argtypes = [ctypes.c_void_p] * 7 + [ctypes.c_int] * 3

        # Bulk extraction
        self._fn_extract = self._lib.extract_parts
        self._fn_extract.restype = None
        self._fn_extract.argtypes = [ctypes.c_void_p, ctypes.c_int] + [ctypes.c_void_p] * 3

        # Batch block FMA (hardware primitive)
        self._fn_bfma = self._lib.tc_block_fma_batch
        self._fn_bfma.restype = None
        self._fn_bfma.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_int] * 3

    def _extract(self, vals: np.ndarray):
        """Pre-extract input-format parts for an entire matrix."""
        flat = np.ascontiguousarray(vals.ravel(), dtype=np.float32)
        count = flat.size
        signs = np.zeros(count, dtype=np.int8)
        exps = np.zeros(count, dtype=np.int16)
        sigs = np.zeros(count, dtype=np.uint16)
        self._fn_extract(
            flat.ctypes.data, count,
            signs.ctypes.data, exps.ctypes.data, sigs.ctypes.data
        )
        return signs, exps, sigs

    def matmul(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Run matmul, return raw FP32 accumulator.

        Uses optimized path: pre-extraction + B transpose + OpenMP.
        Numerically identical to the slow path.

        A: [M, K] input-format values stored as FP32
        B: [K, N] input-format values stored as FP32
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"K mismatch: {K} vs {K2}"

        # Pre-extract A parts [M*K]
        a_sign, a_exp, a_sig = self._extract(A)

        # Transpose B to [N, K] and pre-extract
        B_t = np.ascontiguousarray(B.T, dtype=np.float32)  # [N, K]
        bt_sign, bt_exp, bt_sig = self._extract(B_t)

        # Run optimized matmul
        C = np.zeros((M, N), dtype=np.float32)
        self._fn_opt(
            a_sign.ctypes.data, a_exp.ctypes.data, a_sig.ctypes.data,
            bt_sign.ctypes.data, bt_exp.ctypes.data, bt_sig.ctypes.data,
            C.ctypes.data, M, K, N
        )
        return C

    def block_fma_batch(self, acc: np.ndarray, A_block: np.ndarray, B_block: np.ndarray) -> np.ndarray:
        """Apply one block FMA to M*N independent accumulators.

        acc[M,N] += A_block[M,n] * B_block[n,N]
        where n <= NFMA. This is the hardware primitive — caller manages the K-walk.
        Returns updated acc (new array).

        WHY: FA2's PV matmul accumulates across KV tiles into the same MMA
        register. Using matmul() (which starts from zero) then adding to O_acc
        produces different FP32 rounding. This function lets the caller pass
        the live O_acc as the initial accumulator. See gemm_rs() in FA2 utils.h.
        """
        M, N = acc.shape
        M2, n = A_block.shape
        n2, N2 = B_block.shape
        assert M == M2 and N == N2 and n == n2
        assert n <= self.profile.nfma, f"block width {n} > NFMA {self.profile.nfma}"

        out = np.ascontiguousarray(acc, dtype=np.float32).copy()
        A_c = np.ascontiguousarray(A_block, dtype=np.float32)
        B_c = np.ascontiguousarray(B_block, dtype=np.float32)
        self._fn_bfma(out.ctypes.data, A_c.ctypes.data, B_c.ctypes.data, M, n, N)
        return out

    def matmul_slow(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Run matmul using the reference (non-optimized) path.

        Useful for verifying the optimized path gives identical results.
        """
        M, K = A.shape
        K2, N = B.shape
        assert K == K2, f"K mismatch: {K} vs {K2}"
        C = np.zeros((M, N), dtype=np.float32)
        self._fn_slow(
            np.ascontiguousarray(A, dtype=np.float32).ctypes.data,
            np.ascontiguousarray(B, dtype=np.float32).ctypes.data,
            C.ctypes.data, M, K, N
        )
        return C

    def matmul_bf16(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Run matmul with BF16 epilogue (matches CUTLASS BF16 output)."""
        assert HAS_TORCH, "torch required for BF16 conversion"
        C_fp32 = self.matmul(A, B)
        return torch.tensor(C_fp32).bfloat16().float().numpy()

    def describe(self):
        print(self.profile.describe())
        print(f"  Blocks per MMA: {self.profile.blocks_per_mma}")
        print(f"  Window bits: {self.profile.window_bits}")
        print(f"  Acc output bits: {self.profile.acc_output_bits}")


# ============================================================
# Self-test
# ============================================================
if __name__ == "__main__":
    print("Testing A100 BF16→FP32 emulator...")
    profile = get_profile("A100", "bf16", "fp32")
    emu = TensorCoreEmulator(profile)
    emu.describe()

    rng = np.random.RandomState(42)
    A = rng.randn(4, 16).astype(np.float32)
    B = rng.randn(16, 4).astype(np.float32)

    if HAS_TORCH:
        A = torch.tensor(A).bfloat16().float().numpy()
        B = torch.tensor(B).bfloat16().float().numpy()

    C_fast = emu.matmul(A, B)
    C_slow = emu.matmul_slow(A, B)

    match = np.array_equal(C_fast.view(np.uint32), C_slow.view(np.uint32))
    print(f"\n  Fast vs slow: {'MATCH' if match else 'MISMATCH'}")
    print(f"  C[0,0] = {C_fast[0, 0]:.8e}")

    if not match:
        diffs = int(np.sum(C_fast.view(np.uint32) != C_slow.view(np.uint32)))
        print(f"  WARNING: {diffs} FP32 diffs between fast and slow paths!")

    print()
    print("Available profiles:")
    print("=" * 90)
    from tc_profiles import list_profiles
    list_profiles()
