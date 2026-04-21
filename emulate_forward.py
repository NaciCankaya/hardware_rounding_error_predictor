#!/usr/bin/env python3
"""
emulate_forward.py — CPU-only full forward pass emulation.

Reads the capture archive produced by capture_forward.py, then runs
the bit-exact emulator layer by layer.  Prints a one-line status per
layer.  On the first mismatch, dumps the full per-stage diagnostic for
that block and exits non-zero.

Usage:
    python3 emulate_forward.py [capture_dir]

    capture_dir   directory written by capture_forward.py
                  (default ./forward_capture)

No GPU required.  Loads model weights from HuggingFace (CPU, ~8 GB RAM).
"""

import json
import math
import os
import sys
import time

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

from tc_profiles import get_profile
from tc_emulator import TensorCoreEmulator
from mufu_emulator import MUFUEmulator
from block_emulators import (
    run_attn_block, run_ffn_block,
    diagnose_attn_block, diagnose_ffn_block,
    to_bf16_f32, count_bf16_diffs,
)

DEFAULT_CAPTURE_DIR = "forward_capture"
INPUT_FMT = "bf16"
OUTPUT_FMT = "fp32"


def load_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def load_layer_tensor(cap_dir, layer_idx, key, shape):
    path = os.path.join(cap_dir, f"L{layer_idx:02d}_{key}.bin")
    if not os.path.exists(path):
        return None
    return load_bin(path, shape)


def _bf16_snap(arr):
    """Return arr with values snapped to BF16 precision (FP32 container)."""
    return to_bf16_f32(arr)


def emulate(cap_dir=DEFAULT_CAPTURE_DIR):
    meta_path = os.path.join(cap_dir, "meta.json")
    if not os.path.exists(meta_path):
        print(f"ERROR: {meta_path} not found.  Run capture_forward.py first.")
        sys.exit(1)

    with open(meta_path) as f:
        meta = json.load(f)

    n_layers = meta["n_layers"]
    H = meta["H"]
    ffn_dim = meta["ffn_dim"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]
    gqa_groups = meta["gqa_groups"]
    eps = meta["eps"]
    has_qk_norm = meta["has_qk_norm"]
    qk_eps = meta.get("qk_eps", eps)
    rope_theta = meta["rope_theta"]
    rope_type = meta.get("rope_type", "default")
    hidden_act = meta.get("hidden_act", "silu")
    capture_path = meta.get("capture_path", "cublas-legacy")
    seq_len = meta["seq_len"]
    model_name = meta["model"]

    # The emulator targets CUTLASS tile config.  If the capture came from
    # a cuBLAS-path run (hooked model forward), every matmul will show
    # ~0.2% diffs and FA2 will amplify them by ~50× — not a regression,
    # just an impossible comparison.  Refuse to run against such captures.
    assert capture_path == "cutlass", (
        f"capture_path={capture_path!r}; emulator only validates against "
        f"CUTLASS-path captures.  Re-run capture_forward.py."
    )

    # Which GPU the capture ran on — must come from the capture, not from
    # the current host (this script is intended to run on a CPU-only box
    # where detect_gpu() would raise).
    if "gpu" not in meta:
        raise RuntimeError(
            "meta.json has no 'gpu' field.  Re-run capture_forward.py — the "
            "emulator can't pick a tensor-core profile without knowing which "
            "GPU produced the capture."
        )
    gpu = meta["gpu"]

    # The emulator hardcodes Qwen3-family assumptions.  Bail early if the
    # capture came from a model that violates them — silent misbehaviour
    # would be far worse than a clear abort.
    assert rope_type == "default", (
        f"block_emulators.run_attn_block implements rotate_half RoPE with "
        f"inv_freq = 1/theta^(2i/d).  Capture has rope_type={rope_type!r}, "
        f"which needs a different formula."
    )
    assert hidden_act == "silu", (
        f"block_emulators.run_ffn_block implements SiLU(gate) * up.  "
        f"Capture has hidden_act={hidden_act!r}, which decomposes differently."
    )

    print("=" * 80)
    print("FULL FORWARD PASS EMULATION")
    print(f"  capture: {cap_dir}")
    print(f"  model:   {model_name}")
    print(f"  seq_len: {seq_len}  layers: {n_layers}  hidden: {H}")
    print("=" * 80)
    print()

    # -----------------------------------------------------------------------
    # Emulator setup (uses GPU name from capture, not from current host)
    # -----------------------------------------------------------------------
    profile = get_profile(gpu, INPUT_FMT, OUTPUT_FMT)
    tc_emu = TensorCoreEmulator(profile)
    mufu = MUFUEmulator(gpu_name=gpu)
    print(f"Emulating {gpu}  —  {profile.describe()}")
    print()

    # -----------------------------------------------------------------------
    # Load model weights (CPU only)
    # -----------------------------------------------------------------------
    print(f"Loading {model_name} weights on CPU (this takes a moment)...")
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, device_map="cpu"
    )
    model.eval()
    print("done.")
    print()

    # -----------------------------------------------------------------------
    # Config dicts for block_emulators
    # -----------------------------------------------------------------------
    attn_cfg = dict(H=H, num_heads=num_heads, num_kv_heads=num_kv_heads,
                    head_dim=head_dim, gqa_groups=gqa_groups, eps=eps,
                    has_qk_norm=has_qk_norm, qk_eps=qk_eps)
    ffn_cfg = dict(H=H, ffn_dim=ffn_dim, eps=eps)

    # Weights come out of HuggingFace as BF16 tensors; torch.Tensor.numpy()
    # refuses BF16 because numpy has no native BF16 dtype.  Cast to FP32
    # first (values are preserved exactly — BF16 is a strict FP32 subset),
    # then to_bf16_f32 rounds back to BF16 precision in an FP32 container.
    def _bf16_weight(t):
        return to_bf16_f32(t.detach().float().numpy())

    def _bf16_weight_T(t):
        return to_bf16_f32(t.detach().float().t().numpy())

    def layer_weights_attn(i):
        layer = model.model.layers[i]
        attn = layer.self_attn
        w = {
            "ln_w":  _bf16_weight(layer.input_layernorm.weight),
            "q_w":   _bf16_weight_T(attn.q_proj.weight),
            "k_w":   _bf16_weight_T(attn.k_proj.weight),
            "v_w":   _bf16_weight_T(attn.v_proj.weight),
            "o_w":   _bf16_weight_T(attn.o_proj.weight),
        }
        if has_qk_norm:
            w["q_norm_w"] = _bf16_weight(attn.q_norm.weight)
            w["k_norm_w"] = _bf16_weight(attn.k_norm.weight)
        return w

    def layer_weights_ffn(i):
        layer = model.model.layers[i]
        mlp = layer.mlp
        return {
            "ln_w":   _bf16_weight(layer.post_attention_layernorm.weight),
            "gate_w": _bf16_weight_T(mlp.gate_proj.weight),
            "up_w":   _bf16_weight_T(mlp.up_proj.weight),
            "down_w": _bf16_weight_T(mlp.down_proj.weight),
        }

    # -----------------------------------------------------------------------
    # Load RoPE cos/sin (GPU-computed, pre-BF16-cast)
    # -----------------------------------------------------------------------
    print("Loading RoPE cos/sin...")
    cos_raw = np.load(os.path.join(cap_dir, "rope_cos_gpu.npy"))  # [seq_len, head_dim]
    sin_raw = np.load(os.path.join(cap_dir, "rope_sin_gpu.npy"))
    # Snap to BF16 (transformers casts cos/sin to model dtype before RoPE apply)
    cos_rope = to_bf16_f32(cos_raw)  # [seq_len, head_dim], already duplicated [cos,cos]
    sin_rope = to_bf16_f32(sin_raw)
    print(f"  cos/sin shape: {cos_rope.shape}")
    print()

    # -----------------------------------------------------------------------
    # Helper: load captured tensors for one layer into a dict
    # -----------------------------------------------------------------------
    def load_attn_captured(i):
        cap = {}
        for key, shape in [
            ("q_proj_out",  (seq_len, num_heads * head_dim)),
            ("k_proj_out",  (seq_len, num_kv_heads * head_dim)),
            ("v_proj_out",  (seq_len, num_kv_heads * head_dim)),
            ("o_proj_input",(seq_len, num_heads * head_dim)),
            ("o_proj_out",  (seq_len, H)),
            ("fa2_q",       (seq_len, num_heads, head_dim)),
            ("fa2_k",       (seq_len, num_kv_heads, head_dim)),
            ("fa2_v",       (seq_len, num_kv_heads, head_dim)),
            ("attn_residual",(seq_len, H)),
            ("rms_out",     (seq_len, H)),
            ("attn_block_out",(seq_len, H)),
        ]:
            cap[key] = load_layer_tensor(cap_dir, i, key, shape)

        if has_qk_norm:
            cap["q_normed"] = load_layer_tensor(cap_dir, i, "q_normed",
                                                (seq_len, num_heads, head_dim))
            cap["k_normed"] = load_layer_tensor(cap_dir, i, "k_normed",
                                                (seq_len, num_kv_heads, head_dim))

        # RMSNorm FP32 sub-intermediates
        for key, shape in [
            ("gpu_rms_sumsq",      (seq_len, 1)),
            ("gpu_rms_variance",   (seq_len, 1)),
            ("gpu_rms_rsqrt",      (seq_len, 1)),
            ("gpu_rms_normed_f32", (seq_len, H)),
            ("gpu_rms_normed_bf16",(seq_len, H)),
        ]:
            cap[key] = load_layer_tensor(cap_dir, i, key, shape)

        return {k: v for k, v in cap.items() if v is not None}

    def load_ffn_captured(i):
        cap = {}
        # Note: the FFN RMSNorm output is saved as L{i}_ffn_rms_out.bin on
        # disk because the attention block's input_layernorm output already
        # claimed the bare "rms_out" key in capture_forward.py.  Store it
        # here under "rms_out" so diagnose_ffn_block can find it.
        ffn_rms = load_layer_tensor(cap_dir, i, "ffn_rms_out", (seq_len, H))
        if ffn_rms is not None:
            cap["rms_out"] = ffn_rms
        for key, shape in [
            ("ffn_residual",  (seq_len, H)),
            ("gate_out",      (seq_len, ffn_dim)),
            ("up_out",        (seq_len, ffn_dim)),
            ("down_input",    (seq_len, ffn_dim)),
            ("down_out",      (seq_len, H)),
            ("ffn_block_out", (seq_len, H)),
        ]:
            cap[key] = load_layer_tensor(cap_dir, i, key, shape)

        for key, shape in [
            ("ffn_gpu_rms_sumsq",      (seq_len, 1)),
            ("ffn_gpu_rms_variance",   (seq_len, 1)),
            ("ffn_gpu_rms_rsqrt",      (seq_len, 1)),
            ("ffn_gpu_rms_normed_f32", (seq_len, H)),
            ("ffn_gpu_rms_normed_bf16",(seq_len, H)),
        ]:
            val = load_layer_tensor(cap_dir, i, key, shape)
            if val is not None:
                # Rename to the key diagnose_ffn_block expects (gpu_rms_*)
                cap[key.replace("ffn_", "")] = val

        return {k: v for k, v in cap.items() if v is not None}

    # -----------------------------------------------------------------------
    # Helper: check a block's output tensor against captured ground truth
    # -----------------------------------------------------------------------
    def check(emu_out, cap_out, label):
        if cap_out is None:
            return True, 0
        d, t = count_bf16_diffs(emu_out, cap_out)
        return d == 0, d

    # -----------------------------------------------------------------------
    # Layer loop
    # -----------------------------------------------------------------------
    print("LAYER EMULATION")
    print("-" * 60)
    print(f"  {'Layer':>6}  {'Attn':>12}  {'FFN':>12}  {'Time':>8}")
    print("  " + "-" * 46)

    # Starting residual: token embedding output snapped to BF16
    embed_out = load_bin(os.path.join(cap_dir, "embed_out.bin"), (seq_len, H))
    x = _bf16_snap(embed_out)

    t_total = time.time()

    for i in range(n_layers):
        t_layer = time.time()

        # --- Attention block ---
        attn_w = layer_weights_attn(i)
        attn_cap = load_attn_captured(i)

        # Run emulator (prints FA2 head progress)
        print(f"\n  Layer {i:2d} attn:")
        attn_out, attn_emu, attn_emu_fp32 = run_attn_block(
            x, attn_w, cos_rope, sin_rope, attn_cfg, tc_emu, mufu
        )

        attn_ok, attn_diffs = check(attn_out, attn_cap.get("attn_block_out"), "attn_block_out")

        if not attn_ok:
            print(f"\n  *** MISMATCH at layer {i} attention block ({attn_diffs} BF16 diffs) ***")
            diagnose_attn_block(attn_emu, attn_emu_fp32, attn_cap, attn_cfg, label=f"L{i}")
            sys.exit(1)

        # --- FFN block ---
        ffn_w = layer_weights_ffn(i)
        ffn_cap = load_ffn_captured(i)

        ffn_out, ffn_emu, ffn_emu_fp32 = run_ffn_block(
            attn_out, ffn_w, ffn_cfg, tc_emu, mufu
        )

        ffn_ok, ffn_diffs = check(ffn_out, ffn_cap.get("ffn_block_out"), "ffn_block_out")

        elapsed = time.time() - t_layer
        attn_str = ("0 ***" if attn_ok else f"{attn_diffs} DIFFS").rjust(12)
        ffn_str  = ("0 ***" if ffn_ok  else f"{ffn_diffs} DIFFS").rjust(12)
        print(f"  Layer {i:2d}  {attn_str}  {ffn_str}  {elapsed:>7.1f}s")

        if not ffn_ok:
            print(f"\n  *** MISMATCH at layer {i} FFN block ({ffn_diffs} BF16 diffs) ***")
            diagnose_ffn_block(ffn_emu, ffn_emu_fp32, ffn_cap, ffn_cfg, label=f"L{i}")
            sys.exit(1)

        x = ffn_out

    # -----------------------------------------------------------------------
    # Final norm + LM head
    # -----------------------------------------------------------------------
    print()
    print("FINAL NORM + LM HEAD")
    print("-" * 60)

    final_norm_cap = load_bin(os.path.join(cap_dir, "final_norm_out.bin"), (seq_len, H))

    # Final RMSNorm
    from emulate_pytorch_reduce import emulate_sum_reduce
    x_f32 = x.astype(np.float32)
    final_ln_w = _bf16_weight(model.model.norm.weight)
    emu_sumsq = emulate_sum_reduce(x, warp_shfl_decreasing=True)
    variance = (emu_sumsq * np.float32(1.0 / H)).astype(np.float32)
    rsqrt_val = mufu.rsq((variance + np.float32(eps)).astype(np.float32))
    normed_f32 = (x_f32 * rsqrt_val).astype(np.float32)
    normed_bf16 = to_bf16_f32(normed_f32)
    final_norm_out = to_bf16_f32(normed_bf16 * final_ln_w)

    d_norm, t_norm = count_bf16_diffs(final_norm_out, final_norm_cap)
    norm_str = f"0/{t_norm} ***" if d_norm == 0 else f"{d_norm}/{t_norm} DIFFS"
    print(f"  Final RMSNorm:  {norm_str}")

    if d_norm > 0:
        print(f"\n  *** MISMATCH in final RMSNorm ({d_norm} BF16 diffs) ***")
        # Show first few diff positions
        e_bf16 = torch.tensor(final_norm_out).bfloat16()
        c_bf16 = torch.tensor(final_norm_cap).bfloat16()
        diff_mask = (e_bf16 != c_bf16)
        for idx in diff_mask.nonzero().tolist()[:10]:
            i_r, j = idx
            print(f"    [{i_r},{j}]: emu={final_norm_out[i_r,j]:.8e}  gpu={final_norm_cap[i_r,j]:.8e}")
        sys.exit(1)

    # LM head
    lm_head_w = _bf16_weight(model.lm_head.weight)
    logits_cap = load_bin(os.path.join(cap_dir, "logits.bin"),
                          (seq_len, lm_head_w.shape[0]))

    print(f"  LM head [{seq_len},{H}]x[{H},{lm_head_w.shape[0]}]...", end=" ", flush=True)
    t0 = time.time()
    logits_emu_fp32 = tc_emu.matmul(final_norm_out, lm_head_w.T)
    logits_emu = to_bf16_f32(logits_emu_fp32)
    elapsed = time.time() - t0
    print(f"done ({elapsed:.1f}s)")

    d_logits, t_logits = count_bf16_diffs(logits_emu, logits_cap)
    logits_str = f"0/{t_logits} ***" if d_logits == 0 else f"{d_logits}/{t_logits} DIFFS"
    print(f"  LM head logits: {logits_str}")

    if d_logits > 0:
        print(f"\n  *** MISMATCH in LM head ({d_logits} BF16 diffs) ***")
        e_bf16 = torch.tensor(logits_emu).bfloat16()
        c_bf16 = torch.tensor(logits_cap).bfloat16()
        diff_mask = (e_bf16 != c_bf16)
        for idx in diff_mask.nonzero().tolist()[:10]:
            i_r, j = idx
            print(f"    [{i_r},{j}]: emu={logits_emu[i_r,j]:.8e}  gpu={logits_cap[i_r,j]:.8e}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    total_elapsed = time.time() - t_total
    print()
    print("=" * 80)
    print("ALL LAYERS: 0 BF16 DIFFS")
    print(f"  {n_layers} layers × (attn + ffn) + final norm + LM head")
    print(f"  seq_len={seq_len}  hidden={H}  total time: {total_elapsed:.0f}s")
    print()
    print("The CPU emulator predicts every BF16 bit of the full forward pass")
    print(f"of {model_name} on {seq_len} real tokens.")
    print("=" * 80)


if __name__ == "__main__":
    cap_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CAPTURE_DIR
    emulate(cap_dir=cap_dir)
