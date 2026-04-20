"""
block_emulators.py — Pure CPU emulation of one transformer block.

Extracted from attn_chain_test.py and ffn_chain_test.py.
No file I/O, no prints in the run_* functions — call the diagnose_*
functions explicitly when you want output.

Public API
----------
run_attn_block(residual_bf16, weights, cos_rope, sin_rope, config, tc_emu, mufu)
    -> (attn_block_out, emu, emu_fp32)

run_ffn_block(residual_bf16, weights, config, tc_emu, mufu)
    -> (ffn_block_out, emu, emu_fp32)

diagnose_attn_block(emu, emu_fp32, captured, config)
    Full per-stage printout.  Mirrors Phase 3 of attn_chain_test.py.
    captured: dict of GPU-captured tensors (same keys as emu).

diagnose_ffn_block(emu, emu_fp32, captured, config)
    Full per-stage printout.  Mirrors Phase 3 of ffn_chain_test.py.

Config dict keys
----------------
Shared:
    H             int   hidden_dim
    eps           float rms_norm_eps
Attn only:
    num_heads     int
    num_kv_heads  int
    head_dim      int
    gqa_groups    int
    has_qk_norm   bool
    qk_eps        float  (only required when has_qk_norm)
FFN only:
    ffn_dim       int
"""

import math
import time

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Shared utilities (duplicated from chain tests to keep this module self-contained)
# ---------------------------------------------------------------------------

def to_bf16_f32(x):
    """Round to BF16 precision, keep as FP32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.float().bfloat16().float().numpy()
    return torch.tensor(x).bfloat16().float().numpy()


def count_bf16_diffs(a, b):
    if isinstance(a, np.ndarray):
        a = torch.tensor(a)
    if isinstance(b, np.ndarray):
        b = torch.tensor(b)
    a_bf16 = a.float().bfloat16()
    b_bf16 = b.float().bfloat16()
    return int(torch.sum(a_bf16 != b_bf16).item()), a_bf16.numel()


# ---------------------------------------------------------------------------
# FFN block
# ---------------------------------------------------------------------------

def run_ffn_block(residual_bf16, weights, config, tc_emu, mufu):
    """
    Parameters
    ----------
    residual_bf16 : np.ndarray [M, H], FP32 values at BF16 precision
    weights : dict
        ln_w   [H]      post_attention_layernorm weight
        gate_w [H, ffn_dim]
        up_w   [H, ffn_dim]
        down_w [ffn_dim, H]
        All FP32, BF16 precision.
    config : dict  — keys: H, ffn_dim, eps
    tc_emu : TensorCoreEmulator
    mufu   : MUFUEmulator

    Returns
    -------
    ffn_block_out : np.ndarray [M, H]
    emu           : dict  BF16-precision intermediates
    emu_fp32      : dict  raw FP32 accumulators
    """
    M = residual_bf16.shape[0]
    H = config["H"]
    ffn_dim = config["ffn_dim"]
    eps = config["eps"]

    ln_w = weights["ln_w"]
    gate_w = weights["gate_w"]
    up_w = weights["up_w"]
    down_w = weights["down_w"]

    emu = {}
    emu_fp32 = {}

    # 1. RMSNorm
    from emulate_pytorch_reduce import emulate_sum_reduce
    x_f32 = residual_bf16.astype(np.float32)
    emu_sumsq = emulate_sum_reduce(residual_bf16, warp_shfl_decreasing=True)
    variance = (emu_sumsq * np.float32(1.0 / H)).astype(np.float32)
    rsqrt_val = mufu.rsq((variance + np.float32(eps)).astype(np.float32))
    normed_f32 = (x_f32 * rsqrt_val).astype(np.float32)
    normed_bf16 = to_bf16_f32(normed_f32)
    rms_out = to_bf16_f32(normed_bf16 * ln_w)
    emu["rms_out"] = rms_out
    emu_fp32["rms_sumsq"] = emu_sumsq.copy()
    emu_fp32["rms_variance"] = variance.copy()
    emu_fp32["rms_rsqrt"] = rsqrt_val.copy()
    emu_fp32["rms_normed_f32"] = normed_f32.copy()
    emu_fp32["rms_normed_bf16"] = normed_bf16.copy()

    # 2. gate_proj
    gate_raw_fp32 = tc_emu.matmul(rms_out, gate_w)
    emu_fp32["gate_out"] = gate_raw_fp32.copy()
    gate_out = to_bf16_f32(gate_raw_fp32)
    emu["gate_out"] = gate_out

    # 3. up_proj
    up_raw_fp32 = tc_emu.matmul(rms_out, up_w)
    emu_fp32["up_out"] = up_raw_fp32.copy()
    up_out = to_bf16_f32(up_raw_fp32)
    emu["up_out"] = up_out

    # 4. SiLU(gate)
    gate_f32 = gate_out.astype(np.float32)
    with np.errstate(over="ignore"):
        sigmoid_val = 1.0 / (1.0 + np.exp(-gate_f32))
    silu_f32 = (gate_f32 * sigmoid_val).astype(np.float32)
    silu_gate = to_bf16_f32(silu_f32)
    emu["silu_gate"] = silu_gate

    # 5. SiLU(gate) * up
    down_input = to_bf16_f32(silu_gate.astype(np.float32) * up_out.astype(np.float32))
    emu["down_input"] = down_input

    # 6. down_proj
    down_raw_fp32 = tc_emu.matmul(down_input, down_w)
    emu_fp32["down_out"] = down_raw_fp32.copy()
    down_out = to_bf16_f32(down_raw_fp32)
    emu["down_out"] = down_out

    # 7. Residual add
    ffn_block_out = to_bf16_f32(residual_bf16 + down_out)
    emu["ffn_block_out"] = ffn_block_out

    return ffn_block_out, emu, emu_fp32


def diagnose_ffn_block(emu, emu_fp32, captured, config, label=""):
    """
    Mirrors the THREE-WAY DIAGNOSTIC section of ffn_chain_test.py Phase 3.
    captured: dict with same keys as emu, values are GPU-captured FP32 arrays.
    label: optional prefix for section headers (e.g. "L12")
    """
    M = captured.get("rms_out", next(iter(captured.values()))).shape[0]
    H = config["H"]
    ffn_dim = config["ffn_dim"]

    tag = f"[{label}] " if label else ""

    print("=" * 80)
    print(f"{tag}FFN BLOCK DIAGNOSTIC")
    print("=" * 80)
    print()

    stages = [
        ("RMSNorm out",   "rms_out",       (M, H)),
        ("gate_proj out", "gate_out",      (M, ffn_dim)),
        ("up_proj out",   "up_out",        (M, ffn_dim)),
        ("SiLU(gate)",    "silu_gate",     (M, ffn_dim)),
        ("SiLU*up",       "down_input",    (M, ffn_dim)),
        ("down_proj out", "down_out",      (M, H)),
        ("FFN block out", "ffn_block_out", (M, H)),
    ]

    col_w = 16
    header = f"  {'Stage':<18} {'Emu vs GPU':>{col_w}} {'Total elements':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    first_diff = None
    for label_s, key, shape in stages:
        emu_data = emu.get(key)
        cap_data = captured.get(key)
        if emu_data is None or cap_data is None:
            print(f"  {label_s:<18} {'N/A':>{col_w}} {'':>{col_w}}")
            continue
        d, t = count_bf16_diffs(emu_data, cap_data)
        pct = d / t * 100
        diff_str = (f"0/{t} ***" if d == 0 else f"{d}/{t} ({pct:.2f}%)").rjust(col_w)
        print(f"  {label_s:<18} {diff_str} {str(t):>{col_w}}")
        if d > 0 and first_diff is None:
            first_diff = label_s

    # RMSNorm FP32 intermediate diagnostic
    print()
    print("=" * 80)
    print("RMSNorm FP32 INTERMEDIATE DIAGNOSTIC")
    print("=" * 80)
    print()

    rms_fp32_stages = [
        ("sum_of_squares", "rms_sumsq",     (M, 1)),
        ("variance",       "rms_variance",  (M, 1)),
        ("rsqrt",          "rms_rsqrt",     (M, 1)),
        ("normed_f32",     "rms_normed_f32",(M, H)),
        ("normed_bf16",    "rms_normed_bf16",(M, H)),
    ]
    cap_rms = {k: captured.get(f"rms_{k.split('_',1)[-1]}") for _, k, _ in rms_fp32_stages}

    print(f"  {'Stage':<18} {'FP32 diffs':>16} {'BF16 diffs':>16} {'rows affected':>16}")
    print("  " + "-" * 70)

    for stage_label, emu_key, shape in rms_fp32_stages:
        emu_data = emu_fp32.get(emu_key)
        cap_key = f"gpu_{emu_key}"
        cap_data = captured.get(cap_key)
        if emu_data is None or cap_data is None:
            print(f"  {stage_label:<18} {'N/A':>50}")
            continue

        total = emu_data.size
        fp32_diffs = int(np.sum(emu_data.view(np.uint32) != cap_data.view(np.uint32)))
        bf16_d, _ = count_bf16_diffs(emu_data, cap_data)
        if len(shape) == 2 and shape[1] > 1:
            row_mask = np.any(emu_data.view(np.uint32) != cap_data.view(np.uint32), axis=1)
            rows_str = f"{int(np.sum(row_mask))}/{shape[0]}"
        else:
            rows_str = f"{fp32_diffs}/{shape[0]}"
        print(f"  {stage_label:<18} {str(fp32_diffs)+'/'+str(total):>16} "
              f"{str(bf16_d)+'/'+str(total):>16} {rows_str:>16}")
        if fp32_diffs > 0:
            diff_mask = (emu_data.view(np.uint32) != cap_data.view(np.uint32))
            diffs_abs = np.abs(emu_data.ravel()[diff_mask.ravel()] - cap_data.ravel()[diff_mask.ravel()])
            vals_abs = np.maximum(np.abs(emu_data.ravel()[diff_mask.ravel()]),
                                  np.abs(cap_data.ravel()[diff_mask.ravel()]))
            rel_diffs = diffs_abs / np.maximum(vals_abs, 1e-30)
            print(f"    {'':18} abs: min={diffs_abs.min():.2e} max={diffs_abs.max():.2e} "
                  f"med={np.median(diffs_abs):.2e}")
            print(f"    {'':18} rel: min={rel_diffs.min():.2e} max={rel_diffs.max():.2e} "
                  f"med={np.median(rel_diffs):.2e}")

    # Detailed diff positions for small diff counts
    if first_diff is not None:
        print()
        print("DIFF DIAGNOSIS")
        print("-" * 60)
        for label_s, key, shape in stages:
            emu_data = emu.get(key)
            cap_data = captured.get(key)
            if emu_data is None or cap_data is None:
                continue
            d, t = count_bf16_diffs(emu_data, cap_data)
            if d == 0:
                continue
            if d <= 20:
                print(f"  {label_s}: {d} diffs")
                e_bf16 = torch.tensor(emu_data).bfloat16()
                c_bf16 = torch.tensor(cap_data).bfloat16()
                diff_mask = (e_bf16 != c_bf16)
                for idx in diff_mask.nonzero().tolist()[:20]:
                    i, j = idx
                    ev, cv = emu_data[i, j], cap_data[i, j]
                    eb = e_bf16[i, j].float().item()
                    cb = c_bf16[i, j].float().item()
                    print(f"    [{i},{j}]: emu_f32={ev:.8e} gpu_f32={cv:.8e} "
                          f"emu_bf16={eb:.8e} gpu_bf16={cb:.8e}")
            else:
                print(f"  {label_s}: {d}/{t} diffs (too many to list)")


# ---------------------------------------------------------------------------
# Attention block
# ---------------------------------------------------------------------------

def _fa2_core(fa2_q, fa2_k, fa2_v, num_heads, num_kv_heads, gqa_groups,
               head_dim, M, tc_emu, mufu):
    """
    FlashAttention-2 core emulation.
    Inputs: fa2_q [M, num_heads, head_dim],
            fa2_k [M, num_kv_heads, head_dim],
            fa2_v [M, num_kv_heads, head_dim]  — all BF16-snapped FP32.
    Returns: fa2_out [M, num_heads, head_dim] FP32
    """
    kBlockM = 128
    kBlockN = 64
    softmax_scale = np.float32(1.0 / math.sqrt(head_dim))
    scale_log2 = np.float32(softmax_scale * math.log2(math.e))

    thread_cols = []
    for t in range(4):
        cols = []
        for atom in range(8):
            cols.extend([atom * 8 + 2 * t, atom * 8 + 2 * t + 1])
        thread_cols.append(cols)

    def fma_f32_vec(a, b, c):
        return (a.astype(np.float64) * np.float64(b) + c.astype(np.float64)).astype(np.float32)

    def allreduce4_max(vals):
        return np.maximum(np.maximum(vals[:, 0], vals[:, 2]),
                          np.maximum(vals[:, 1], vals[:, 3]))

    def apply_causal_mask(S, m_start, n_start, m_size, n_size):
        rows = np.arange(m_start, m_start + m_size)[:, None]
        cols = np.arange(n_start, n_start + n_size)[None, :]
        S[rows < cols] = -np.inf

    n_q_tiles = (M + kBlockM - 1) // kBlockM
    n_kv_tiles = (M + kBlockN - 1) // kBlockN
    fa2_out = np.zeros((M, num_heads, head_dim), dtype=np.float32)

    t0 = time.time()
    for qh in range(num_heads):
        kvh = qh // gqa_groups
        Q_head = fa2_q[:, qh, :]
        K_head = fa2_k[:, kvh, :]
        V_head = fa2_v[:, kvh, :]

        if qh == 0:
            print(f"    Head 0/{num_heads}...", end="", flush=True)
        elif qh % 8 == 0:
            elapsed = time.time() - t0
            eta = elapsed / qh * (num_heads - qh)
            print(f"\n    Head {qh}/{num_heads} ({elapsed:.0f}s, ETA {eta:.0f}s)...",
                  end="", flush=True)

        for m_block in range(n_q_tiles):
            m_start = m_block * kBlockM
            m_end = min(m_start + kBlockM, M)
            m_size = m_end - m_start
            Q_tile = Q_head[m_start:m_end]

            row_max = np.full((m_size, 4), -np.inf, dtype=np.float32)
            row_sum = np.zeros((m_size, 4), dtype=np.float32)
            O_acc = np.zeros((m_size, head_dim), dtype=np.float32)

            n_block_max = min((m_end + kBlockN - 1) // kBlockN, n_kv_tiles)
            n_blocks = list(range(n_block_max - 1, -1, -1))

            is_first = True
            rescale = None
            for n_block in n_blocks:
                n_start = n_block * kBlockN
                n_end = min(n_start + kBlockN, M)
                n_size = n_end - n_start
                K_tile = K_head[n_start:n_end]
                V_tile = V_head[n_start:n_end]

                S = tc_emu.matmul(Q_tile, K_tile.T)
                apply_causal_mask(S, m_start, n_start, m_size, n_size)

                if n_size < kBlockN:
                    Sp = np.full((m_size, kBlockN), -np.inf, dtype=np.float32)
                    Sp[:, :n_size] = S
                else:
                    Sp = S

                if is_first:
                    for t in range(4):
                        tmax = Sp[:, thread_cols[t][0]].copy()
                        for c in thread_cols[t][1:]:
                            tmax = np.maximum(tmax, Sp[:, c])
                        row_max[:, t] = tmax
                    gmax = allreduce4_max(row_max)
                    row_max[:, :] = gmax[:, None]

                    mscaled = np.where(gmax == -np.inf, np.float32(0.0),
                                       (gmax * scale_log2).astype(np.float32))
                    for j in range(kBlockN):
                        Sp[:, j] = mufu.ex2(
                            fma_f32_vec(Sp[:, j], scale_log2, -mscaled)
                        ).astype(np.float32)

                    for t in range(4):
                        row_sum[:, t] = Sp[:, thread_cols[t][0]].copy()
                        for c in thread_cols[t][1:]:
                            row_sum[:, t] = (row_sum[:, t] + Sp[:, c]).astype(np.float32)

                    is_first = False
                else:
                    prev_max = row_max[:, 0].copy()

                    for t in range(4):
                        tmax = row_max[:, t].copy()
                        for c in thread_cols[t]:
                            tmax = np.maximum(tmax, Sp[:, c])
                        row_max[:, t] = tmax
                    gmax = allreduce4_max(row_max)
                    row_max[:, :] = gmax[:, None]

                    cur_max = np.where(gmax == -np.inf, np.float32(0.0), gmax)
                    rescale = mufu.ex2(
                        ((prev_max - cur_max) * scale_log2).astype(np.float32)
                    ).astype(np.float32)
                    O_acc = (O_acc * rescale[:, None]).astype(np.float32)

                    mscaled = np.where(gmax == -np.inf, np.float32(0.0),
                                       (gmax * scale_log2).astype(np.float32))
                    for j in range(kBlockN):
                        Sp[:, j] = mufu.ex2(
                            fma_f32_vec(Sp[:, j], scale_log2, -mscaled)
                        ).astype(np.float32)

                    for t in range(4):
                        first_col = thread_cols[t][0]
                        row_sum[:, t] = (
                            row_sum[:, t].astype(np.float64)
                            * rescale.astype(np.float64)
                            + Sp[:, first_col].astype(np.float64)
                        ).astype(np.float32)
                        for c in thread_cols[t][1:]:
                            row_sum[:, t] = (row_sum[:, t] + Sp[:, c]).astype(np.float32)

                P = to_bf16_f32(Sp[:, :n_size])
                nfma = tc_emu.profile.nfma
                k_per_mma = tc_emu.profile.products_per_mma
                for k in range(0, n_size, k_per_mma):
                    k_end = min(k + k_per_mma, n_size)
                    for kb in range(k, k_end, nfma):
                        bend = min(kb + nfma, k_end)
                        O_acc = tc_emu.block_fma_batch(
                            O_acc, P[:, kb:bend], V_tile[kb:bend, :])

            # normalize_softmax_lse
            s02 = (row_sum[:, 0] + row_sum[:, 2]).astype(np.float32)
            s13 = (row_sum[:, 1] + row_sum[:, 3]).astype(np.float32)
            total_sum = (s02 + s13).astype(np.float32)
            inv_sum = np.where(
                (total_sum == 0) | np.isnan(total_sum),
                np.float32(1.0),
                mufu.rcp(total_sum),
            )
            fa2_out[m_start:m_end, qh, :] = (O_acc * inv_sum[:, None]).astype(np.float32)

    elapsed = time.time() - t0
    print(f"\n    FA2 core done ({elapsed:.1f}s)")
    return fa2_out


def run_attn_block(residual_bf16, weights, cos_rope, sin_rope, config, tc_emu, mufu):
    """
    Parameters
    ----------
    residual_bf16 : np.ndarray [M, H], FP32 values at BF16 precision
    weights : dict
        ln_w      [H]
        q_w       [H, num_heads*head_dim]
        k_w       [H, num_kv_heads*head_dim]
        v_w       [H, num_kv_heads*head_dim]
        o_w       [num_heads*head_dim, H]
        q_norm_w  [head_dim]  (optional, required when has_qk_norm)
        k_norm_w  [head_dim]  (optional, required when has_qk_norm)
        All FP32, BF16 precision.
    cos_rope, sin_rope : np.ndarray [M, head_dim]
        BF16-cast FP32, already duplicated as [cos,cos] / [sin,sin]
        for rotate_half style.  Precomputed by caller.
    config : dict
        H, num_heads, num_kv_heads, head_dim, eps, gqa_groups,
        has_qk_norm, qk_eps
    tc_emu : TensorCoreEmulator
    mufu   : MUFUEmulator

    Returns
    -------
    attn_block_out : np.ndarray [M, H]
    emu            : dict  BF16-precision intermediates
    emu_fp32       : dict  raw FP32 accumulators / sub-intermediates
    """
    from emulate_pytorch_reduce import emulate_sum_reduce

    M = residual_bf16.shape[0]
    H = config["H"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    eps = config["eps"]
    gqa_groups = config["gqa_groups"]
    has_qk_norm = config["has_qk_norm"]
    qk_eps = config.get("qk_eps", eps)
    half_dim = head_dim // 2

    ln_w = weights["ln_w"]
    q_w = weights["q_w"]
    k_w = weights["k_w"]
    v_w = weights["v_w"]
    o_w = weights["o_w"]

    emu = {}
    emu_fp32 = {}

    # 1. RMSNorm (input_layernorm)
    x_f32 = residual_bf16.astype(np.float32)
    emu_sumsq = emulate_sum_reduce(residual_bf16, warp_shfl_decreasing=True)
    variance = (emu_sumsq * np.float32(1.0 / H)).astype(np.float32)
    rsqrt_val = mufu.rsq((variance + np.float32(eps)).astype(np.float32))
    normed_f32 = (x_f32 * rsqrt_val).astype(np.float32)
    normed_bf16 = to_bf16_f32(normed_f32)
    rms_out = to_bf16_f32(normed_bf16 * ln_w)
    emu["rms_out"] = rms_out
    emu_fp32["rms_sumsq"] = emu_sumsq.copy()
    emu_fp32["rms_variance"] = variance.copy()
    emu_fp32["rms_rsqrt"] = rsqrt_val.copy()
    emu_fp32["rms_normed_f32"] = normed_f32.copy()
    emu_fp32["rms_normed_bf16"] = normed_bf16.copy()

    # 2. Q/K/V projections
    for name, w, out_dim in [
        ("q_proj_out", q_w, num_heads * head_dim),
        ("k_proj_out", k_w, num_kv_heads * head_dim),
        ("v_proj_out", v_w, num_kv_heads * head_dim),
    ]:
        raw = tc_emu.matmul(rms_out, w)
        emu_fp32[name] = raw.copy()
        emu[name] = to_bf16_f32(raw)

    # 3. Reshape to per-head (exact)
    q_heads = emu["q_proj_out"].reshape(M, num_heads, head_dim)
    k_heads = emu["k_proj_out"].reshape(M, num_kv_heads, head_dim)
    v_heads = emu["v_proj_out"].reshape(M, num_kv_heads, head_dim)

    # 4. QK-norm
    if has_qk_norm:
        q_norm_w = weights["q_norm_w"]
        k_norm_w = weights["k_norm_w"]
        q_normed = np.zeros_like(q_heads)
        k_normed = np.zeros_like(k_heads)
        for h in range(num_heads):
            qh = q_heads[:, h, :]
            ss = emulate_sum_reduce(qh, warp_shfl_decreasing=True)
            var = (ss * np.float32(1.0 / head_dim)).astype(np.float32)
            rs = mufu.rsq((var + np.float32(qk_eps)).astype(np.float32))
            nf = (qh.astype(np.float32) * rs).astype(np.float32)
            nb = to_bf16_f32(nf)
            q_normed[:, h, :] = to_bf16_f32(nb * q_norm_w)
        for h in range(num_kv_heads):
            kh = k_heads[:, h, :]
            ss = emulate_sum_reduce(kh, warp_shfl_decreasing=True)
            var = (ss * np.float32(1.0 / head_dim)).astype(np.float32)
            rs = mufu.rsq((var + np.float32(qk_eps)).astype(np.float32))
            nf = (kh.astype(np.float32) * rs).astype(np.float32)
            nb = to_bf16_f32(nf)
            k_normed[:, h, :] = to_bf16_f32(nb * k_norm_w)
        emu["q_normed"] = q_normed
        emu["k_normed"] = k_normed
    else:
        q_normed = q_heads
        k_normed = k_heads

    # 5. RoPE (rotate_half style, BF16 cast after each multiply)
    def apply_rope_head(x, cos, sin):
        x1 = x[:, :half_dim]
        x2 = x[:, half_dim:]
        rotated = np.concatenate([-x2, x1], axis=-1)
        tmp1 = to_bf16_f32(x * cos)
        tmp2 = to_bf16_f32(rotated * sin)
        return to_bf16_f32(tmp1 + tmp2)

    q_roped = np.zeros_like(q_normed)
    k_roped = np.zeros_like(k_normed)
    for h in range(num_heads):
        q_roped[:, h, :] = apply_rope_head(q_normed[:, h, :], cos_rope, sin_rope)
    for h in range(num_kv_heads):
        k_roped[:, h, :] = apply_rope_head(k_normed[:, h, :], cos_rope, sin_rope)
    v_roped = v_heads

    emu["q_roped"] = q_roped
    emu["k_roped"] = k_roped

    # 6. FA2 core (snaps to BF16 grid at load boundary)
    fa2_q = to_bf16_f32(q_roped)
    fa2_k = to_bf16_f32(k_roped)
    fa2_v = to_bf16_f32(v_roped)

    print(f"  [6/8] FA2 attention core...", flush=True)
    fa2_out = _fa2_core(fa2_q, fa2_k, fa2_v,
                        num_heads, num_kv_heads, gqa_groups, head_dim, M,
                        tc_emu, mufu)
    fa2_flat = fa2_out.reshape(M, num_heads * head_dim)
    emu["fa2_out"] = to_bf16_f32(fa2_flat)
    emu_fp32["fa2_out"] = fa2_flat.copy()

    # 7. O projection (uses emulated FA2 output — fully chained)
    o_input = emu["fa2_out"]
    o_raw_fp32 = tc_emu.matmul(o_input, o_w)
    emu_fp32["o_proj_out"] = o_raw_fp32.copy()
    emu["o_proj_out"] = to_bf16_f32(o_raw_fp32)

    # 8. Residual add
    attn_block_out = to_bf16_f32(residual_bf16 + emu["o_proj_out"])
    emu["attn_block_out"] = attn_block_out

    return attn_block_out, emu, emu_fp32


def diagnose_attn_block(emu, emu_fp32, captured, config, label=""):
    """
    Mirrors the THREE-WAY DIAGNOSTIC section of attn_chain_test.py Phase 3.
    captured: dict with same keys as emu, values are GPU-captured FP32 arrays.
    label: optional prefix for section headers (e.g. "L12")
    """
    M = captured.get("rms_out", next(iter(captured.values()))).shape[0]
    H = config["H"]
    num_heads = config["num_heads"]
    num_kv_heads = config["num_kv_heads"]
    head_dim = config["head_dim"]
    has_qk_norm = config["has_qk_norm"]

    tag = f"[{label}] " if label else ""

    print("=" * 80)
    print(f"{tag}ATTENTION BLOCK DIAGNOSTIC")
    print("=" * 80)
    print()

    stages = [
        ("RMSNorm out",    "rms_out",       (M, H)),
        ("Q projection",   "q_proj_out",    (M, num_heads * head_dim)),
        ("K projection",   "k_proj_out",    (M, num_kv_heads * head_dim)),
        ("V projection",   "v_proj_out",    (M, num_kv_heads * head_dim)),
        ("FA2 out",        "fa2_out",       (M, num_heads * head_dim)),
        ("O projection",   "o_proj_out",    (M, H)),
        ("Attn block out", "attn_block_out",(M, H)),
    ]
    if has_qk_norm:
        stages_qknorm = [
            ("Q normed",   "q_normed", (M, num_heads, head_dim)),
            ("K normed",   "k_normed", (M, num_kv_heads, head_dim)),
        ]
    else:
        stages_qknorm = []

    col_w = 16
    header = f"  {'Stage':<18} {'Emu vs GPU':>{col_w}} {'Total elements':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    first_diff = None
    for label_s, key, shape in stages:
        emu_data = emu.get(key)
        cap_data = captured.get(key)
        if emu_data is None or cap_data is None:
            print(f"  {label_s:<18} {'N/A':>{col_w}} {'':>{col_w}}")
            continue
        d, t = count_bf16_diffs(emu_data, cap_data)
        pct = d / t * 100
        diff_str = (f"0/{t} ***" if d == 0 else f"{d}/{t} ({pct:.2f}%)").rjust(col_w)
        print(f"  {label_s:<18} {diff_str} {str(t):>{col_w}}")
        if d > 0 and first_diff is None:
            first_diff = label_s

    if stages_qknorm:
        print()
        print("QK-NORM DIAGNOSTIC")
        print("-" * 60)
        for label_s, key, shape in stages_qknorm:
            emu_data = emu.get(key)
            cap_data = captured.get(key)
            if emu_data is None or cap_data is None:
                print(f"  {label_s}: N/A")
                continue
            d, t = count_bf16_diffs(emu_data, cap_data)
            tag2 = "***" if d == 0 else f"({d/t*100:.2f}%)"
            print(f"  {label_s}: {d}/{t} {tag2}")

    # RMSNorm FP32 intermediate diagnostic
    print()
    print("=" * 80)
    print("RMSNorm FP32 INTERMEDIATE DIAGNOSTIC")
    print("=" * 80)
    print()

    rms_fp32_stages = [
        ("sum_of_squares", "rms_sumsq",      (M, 1)),
        ("variance",       "rms_variance",   (M, 1)),
        ("rsqrt",          "rms_rsqrt",      (M, 1)),
        ("normed_f32",     "rms_normed_f32", (M, H)),
        ("normed_bf16",    "rms_normed_bf16",(M, H)),
    ]

    print(f"  {'Stage':<18} {'FP32 diffs':>16} {'BF16 diffs':>16} {'rows affected':>16}")
    print("  " + "-" * 70)

    for stage_label, emu_key, shape in rms_fp32_stages:
        emu_data = emu_fp32.get(emu_key)
        cap_data = captured.get(f"gpu_{emu_key}")
        if emu_data is None or cap_data is None:
            print(f"  {stage_label:<18} {'N/A':>50}")
            continue
        total = emu_data.size
        fp32_diffs = int(np.sum(emu_data.view(np.uint32) != cap_data.view(np.uint32)))
        bf16_d, _ = count_bf16_diffs(emu_data, cap_data)
        if len(shape) == 2 and shape[1] > 1:
            row_mask = np.any(emu_data.view(np.uint32) != cap_data.view(np.uint32), axis=1)
            rows_str = f"{int(np.sum(row_mask))}/{shape[0]}"
        else:
            rows_str = f"{fp32_diffs}/{shape[0]}"
        print(f"  {stage_label:<18} {str(fp32_diffs)+'/'+str(total):>16} "
              f"{str(bf16_d)+'/'+str(total):>16} {rows_str:>16}")
        if fp32_diffs > 0:
            diff_mask = (emu_data.view(np.uint32) != cap_data.view(np.uint32))
            diffs_abs = np.abs(emu_data.ravel()[diff_mask.ravel()] - cap_data.ravel()[diff_mask.ravel()])
            vals_abs = np.maximum(np.abs(emu_data.ravel()[diff_mask.ravel()]),
                                  np.abs(cap_data.ravel()[diff_mask.ravel()]))
            rel_diffs = diffs_abs / np.maximum(vals_abs, 1e-30)
            print(f"    {'':18} abs: min={diffs_abs.min():.2e} max={diffs_abs.max():.2e} "
                  f"med={np.median(diffs_abs):.2e}")
            print(f"    {'':18} rel: min={rel_diffs.min():.2e} max={rel_diffs.max():.2e} "
                  f"med={np.median(rel_diffs):.2e}")

    if first_diff is not None:
        print()
        print("DIFF DIAGNOSIS")
        print("-" * 60)
        for label_s, key, shape in stages:
            emu_data = emu.get(key)
            cap_data = captured.get(key)
            if emu_data is None or cap_data is None:
                continue
            d, t = count_bf16_diffs(emu_data, cap_data)
            if d == 0:
                continue
            if d <= 20:
                print(f"  {label_s}: {d} diffs")
                e_bf16 = torch.tensor(emu_data).bfloat16()
                c_bf16 = torch.tensor(cap_data).bfloat16()
                diff_mask = (e_bf16 != c_bf16)
                for idx in diff_mask.nonzero().tolist()[:20]:
                    i, j = idx
                    ev = float(emu_data.ravel()[i * emu_data.shape[1] + j]
                               if emu_data.ndim == 2 else emu_data.flat[i])
                    cv = float(cap_data.ravel()[i * cap_data.shape[1] + j]
                               if cap_data.ndim == 2 else cap_data.flat[i])
                    eb = e_bf16.reshape(-1)[i * (emu_data.shape[1] if emu_data.ndim == 2 else 1) + j].float().item()
                    cb = c_bf16.reshape(-1)[i * (cap_data.shape[1] if cap_data.ndim == 2 else 1) + j].float().item()
                    print(f"    [{i},{j}]: emu_f32={ev:.8e} gpu_f32={cv:.8e} "
                          f"emu_bf16={eb:.8e} gpu_bf16={cb:.8e}")
            else:
                print(f"  {label_s}: {d}/{t} diffs (too many to list)")
