#!/usr/bin/env python3
"""
Attention Block Chain Test — Instrumented Version

Mirrors ffn_chain_test.py but for the self-attention block.
Hooks every intermediate tensor from RMSNorm through O-projection,
then compares against CPU emulation.

SOFTWARE STACK: Same as ffn_chain_test.py.
  1. HuggingFace transformers (eager mode)
  2. PyTorch (ATen CUDA kernels)
  3. FlashAttention-2 for the attention core
  4. CUTLASS for matmul ground truth

Pipeline:
  1.   Tokenize → full model prefill → hook attention block at target layer
  2.   Run CUTLASS matmuls for Q/K/V/O projections
  1.5. Feed CUTLASS projections back through GPU QK-norm → RoPE → FA2
  3.   Run CPU emulator chain including FA2 core emulation
  4.   Three-way comparison at EVERY step

Usage:
  python3 attn_chain_test.py extract [seq_len]   # Phase 1
  python3 attn_chain_test.py cutlass              # Phase 2
  python3 attn_chain_test.py rerun                # Phase 1.5
  python3 attn_chain_test.py compare              # Phase 3
  python3 attn_chain_test.py all [seq_len]        # All phases
"""

import numpy as np
import torch
import torch.nn.functional as F
import subprocess
import os
import sys
import time
import json
import glob
import math

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ============================================================
# SOFTWARE STACK DECLARATION (see ffn_chain_test.py for details)
# ============================================================

DEVICE = "cuda"
DATA_DIR = "attn_chain_data"
CUTLASS_BIN = "./cutlass_gemm_flex"
MODEL_NAME = "Qwen/Qwen3-4B"
LAYER = 20

MAX_SEQ_LEN = 256


# ============================================================
# Utilities (same as ffn_chain_test.py)
# ============================================================
def to_bf16_f32(x):
    """Round to BF16 precision, keep as FP32 numpy array."""
    if isinstance(x, torch.Tensor):
        return x.float().bfloat16().float().numpy()
    return torch.tensor(x).bfloat16().float().numpy()


def count_bf16_diffs(a, b):
    if isinstance(a, np.ndarray): a = torch.tensor(a)
    if isinstance(b, np.ndarray): b = torch.tensor(b)
    a_bf16 = a.float().bfloat16()
    b_bf16 = b.float().bfloat16()
    return int(torch.sum(a_bf16 != b_bf16).item()), a_bf16.numel()


def save_bin(path, data):
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy()
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)


def load_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


def extract_pdf_text():
    """Find and extract text from available PDFs."""
    pdf_paths = []
    for d in ["/workspace", "/mnt/user-data/uploads", "/mnt/project", "/mnt/data"]:
        if os.path.isdir(d):
            pdf_paths.extend(glob.glob(os.path.join(d, "**/*.pdf"), recursive=True))
    for pdf_path in sorted(set(pdf_paths), key=os.path.getsize, reverse=True):
        try:
            result = subprocess.run(
                ["pdftotext", pdf_path, "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and len(result.stdout.strip()) > 500:
                print(f"  Using PDF (pdftotext): {pdf_path}")
                return result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        try:
            import PyPDF2
            with open(pdf_path, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                text = "".join(p.extract_text() or "" for p in reader.pages)
            if len(text.strip()) > 500:
                print(f"  Using PDF (PyPDF2): {pdf_path}")
                return text
        except Exception:
            pass
    return None


# ============================================================
# Emulator imports
# ============================================================
from tc_profiles import get_profile, detect_gpu
from tc_emulator import TensorCoreEmulator
from emulate_pytorch_reduce import emulate_sum_reduce
from mufu_emulator import MUFUEmulator


INPUT_FMT = "bf16"
OUTPUT_FMT = "fp32"


# ============================================================
# PHASE 1: Instrumented attention forward pass
# ============================================================
def phase_extract():
    print("=" * 80)
    print("PHASE 1: INSTRUMENTED ATTENTION FORWARD PASS")
    print("=" * 80)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    assert LAYER < n_layers, f"Layer {LAYER} out of range (model has {n_layers} layers)"
    print(f"Loaded. {n_layers} layers, targeting layer {LAYER}.")
    print()

    # ============================================================
    # Architecture inspection — log EVERYTHING about the attention block
    # ============================================================
    print("ARCHITECTURE INSPECTION")
    print("-" * 60)

    layer = model.model.layers[LAYER]
    attn = layer.self_attn
    ln = layer.input_layernorm  # pre-attention layernorm

    hidden_dim = model.config.hidden_size
    num_heads = model.config.num_attention_heads
    num_kv_heads = model.config.num_key_value_heads
    head_dim = model.config.head_dim
    eps = model.config.rms_norm_eps
    gqa_groups = num_heads // num_kv_heads

    print(f"  hidden_size:         {hidden_dim}")
    print(f"  num_attention_heads: {num_heads}")
    print(f"  num_key_value_heads: {num_kv_heads}")
    print(f"  head_dim:            {head_dim}")
    print(f"  GQA groups:          {gqa_groups}")
    print(f"  rms_norm_eps:        {eps}")
    print()

    print(f"  LayerNorm type:  {type(ln).__name__}")
    print(f"  Attention type:  {type(attn).__name__}")
    print(f"  q_proj:  {attn.q_proj.weight.shape}  bias={attn.q_proj.bias is not None}")
    print(f"  k_proj:  {attn.k_proj.weight.shape}  bias={attn.k_proj.bias is not None}")
    print(f"  v_proj:  {attn.v_proj.weight.shape}  bias={attn.v_proj.bias is not None}")
    print(f"  o_proj:  {attn.o_proj.weight.shape}  bias={attn.o_proj.bias is not None}")
    print()

    # QK-norm detection
    q_norm = getattr(attn, "q_norm", None)
    k_norm = getattr(attn, "k_norm", None)
    has_qk_norm = q_norm is not None
    print(f"  QK-norm present: {has_qk_norm}")
    if has_qk_norm:
        print(f"    q_norm type: {type(q_norm).__name__}")
        print(f"    q_norm weight: {q_norm.weight.shape}")
        qk_eps = getattr(q_norm, "variance_epsilon", getattr(q_norm, "eps", None))
        print(f"    q_norm eps: {qk_eps}")
        print(f"    k_norm type: {type(k_norm).__name__}")
        print(f"    k_norm weight: {k_norm.weight.shape}")
    print()

    # RoPE detection
    rotary = getattr(attn, "rotary_emb", None)
    print(f"  RoPE type: {type(rotary).__name__ if rotary else 'NONE'}")
    if rotary:
        rope_config = getattr(model.config, "rope_scaling", None)
        print(f"    rope_scaling: {rope_config}")
    print()

    # Print full submodule tree
    print("  Full attention submodule tree:")
    for name, mod in attn.named_modules():
        if name:
            print(f"    {name}: {type(mod).__name__}")
    print()

    # ============================================================
    # Tokenize
    # ============================================================
    print("TOKENIZATION")
    print("-" * 60)
    text = extract_pdf_text()
    if text:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    else:
        print("  No PDF found, using repeated text")
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")
    print()

    # ============================================================
    # Hook EVERYTHING in the attention block
    # ============================================================
    print("INSTRUMENTED FORWARD PASS")
    print("-" * 60)

    captures = {}
    hooks = []

    # 1. Input to input_layernorm = attention residual
    def hook_ln_pre(module, args):
        x = args[0]
        if isinstance(x, tuple): x = x[0]
        captures["attn_residual"] = x.detach().squeeze(0).clone()
    hooks.append(ln.register_forward_pre_hook(hook_ln_pre))

    # 2. Output of input_layernorm = RMSNorm output
    def hook_ln_post(module, args, output):
        captures["rms_out"] = output.detach().squeeze(0).clone()
    hooks.append(ln.register_forward_hook(hook_ln_post))

    # 3. Q/K/V projection outputs
    def hook_proj(name):
        def hook(module, args, output):
            captures[name] = output.detach().squeeze(0).clone()
        return hook
    hooks.append(attn.q_proj.register_forward_hook(hook_proj("q_proj_out")))
    hooks.append(attn.k_proj.register_forward_hook(hook_proj("k_proj_out")))
    hooks.append(attn.v_proj.register_forward_hook(hook_proj("v_proj_out")))

    # 4. QK-norm outputs (if present)
    if has_qk_norm:
        hooks.append(q_norm.register_forward_hook(hook_proj("q_normed")))
        hooks.append(k_norm.register_forward_hook(hook_proj("k_normed")))

    # 4b. Capture post-RoPE Q/K/V by monkey-patching flash_attn_func
    # These are the actual inputs to the FA2 kernel — after proj, reshape, QK-norm, RoPE.
    try:
        import flash_attn.flash_attn_interface as _fa_iface
        _orig_flash_attn = _fa_iface._flash_attn_forward

        def _capture_flash_attn(*args, **kwargs):
            # flash_attn_func passes (q, k, v, ...) to _flash_attn_forward
            q_arg, k_arg, v_arg = args[0], args[1], args[2]
            # q: [batch, seqlen, nheads, headdim], k/v: [batch, seqlen, nkv, headdim]
            captures["fa2_q"] = q_arg.detach().squeeze(0).clone()  # [seq, nheads, headdim]
            captures["fa2_k"] = k_arg.detach().squeeze(0).clone()  # [seq, nkv, headdim]
            captures["fa2_v"] = v_arg.detach().squeeze(0).clone()  # [seq, nkv, headdim]
            return _orig_flash_attn(*args, **kwargs)

        _fa_iface._flash_attn_forward = _capture_flash_attn
        _patched_fa = True
    except (ImportError, AttributeError):
        try:
            import flash_attn
            _orig_fa_func = flash_attn.flash_attn_func

            def _capture_fa_func(q, k, v, *args, **kwargs):
                captures["fa2_q"] = q.detach().squeeze(0).clone()
                captures["fa2_k"] = k.detach().squeeze(0).clone()
                captures["fa2_v"] = v.detach().squeeze(0).clone()
                return _orig_fa_func(q, k, v, *args, **kwargs)

            flash_attn.flash_attn_func = _capture_fa_func
            _patched_fa = True
        except (ImportError, AttributeError):
            _patched_fa = False
            print("  WARNING: Could not monkey-patch flash_attn to capture post-RoPE Q/K/V")

    # 5. O projection input and output
    def hook_o_pre(module, args):
        x = args[0]
        if isinstance(x, tuple): x = x[0]
        captures["o_proj_input"] = x.detach().squeeze(0).clone()
    hooks.append(attn.o_proj.register_forward_pre_hook(hook_o_pre))
    hooks.append(attn.o_proj.register_forward_hook(hook_proj("o_proj_out")))

    # 6. Full attention block output (self_attn module output)
    def hook_attn_out(module, args, output):
        out = output[0] if isinstance(output, tuple) else output
        captures["attn_module_out"] = out.detach().squeeze(0).clone()
    hooks.append(attn.register_forward_hook(hook_attn_out))

    # Run the model
    print("  Running model forward pass...", end=" ", flush=True)
    with torch.no_grad():
        _ = model(input_ids)
    print("done.")

    # Remove hooks and restore monkey-patches
    for h in hooks:
        h.remove()
    try:
        if _patched_fa:
            import flash_attn.flash_attn_interface as _fa_iface
            if hasattr(_fa_iface, '_flash_attn_forward'):
                _fa_iface._flash_attn_forward = _orig_flash_attn
    except Exception:
        pass

    # ============================================================
    # Print captured tensors
    # ============================================================
    expected_base = ["attn_residual", "rms_out", "q_proj_out", "k_proj_out", "v_proj_out",
                     "o_proj_input", "o_proj_out", "attn_module_out"]
    if has_qk_norm:
        expected_base += ["q_normed", "k_normed"]

    fa2_captured = all(k in captures for k in ["fa2_q", "fa2_k", "fa2_v"])
    if fa2_captured:
        expected_base += ["fa2_q", "fa2_k", "fa2_v"]
    else:
        print("  WARNING: Post-RoPE FA2 inputs not captured")

    for name in expected_base:
        if name not in captures:
            print(f"  ERROR: Failed to capture '{name}'")
            return
        t = captures[name]
        print(f"  {name:<20} {str(list(t.shape)):>20}  dtype={t.dtype}  "
              f"range=[{t.float().min().item():.4f}, {t.float().max().item():.4f}]")

    print()

    # ============================================================
    # Architecture verification
    # ============================================================
    print("ARCHITECTURE VERIFICATION")
    print("-" * 60)

    # Q/K/V are plain linears?
    for proj_name in ["q_proj", "k_proj", "v_proj"]:
        proj = getattr(attn, proj_name)
        rms = captures["rms_out"]
        with torch.no_grad():
            recomputed = proj(rms.unsqueeze(0).to(DEVICE)).squeeze(0)
        d, t = count_bf16_diffs(recomputed.float().cpu().numpy(),
                                captures[f"{proj_name}_out"].float().cpu().numpy())
        status = "*** CONFIRMED ***" if d == 0 else f"WARNING: {d} diffs"
        print(f"  {proj_name} is plain linear? {d}/{t}  {status}")

    # O projection is plain linear?
    with torch.no_grad():
        o_recomp = attn.o_proj(captures["o_proj_input"].unsqueeze(0).to(DEVICE)).squeeze(0)
    d, t = count_bf16_diffs(o_recomp.float().cpu().numpy(),
                            captures["o_proj_out"].float().cpu().numpy())
    status = "*** CONFIRMED ***" if d == 0 else f"WARNING: {d} diffs"
    print(f"  o_proj is plain linear? {d}/{t}  {status}")

    # Reshape verification: q_proj_out → per-head
    q_flat = captures["q_proj_out"]  # [seq, num_heads * head_dim]
    q_heads = q_flat.view(seq_len, num_heads, head_dim)
    print(f"  Q reshape: [{seq_len}, {num_heads * head_dim}] → [{seq_len}, {num_heads}, {head_dim}] (view, exact)")

    k_flat = captures["k_proj_out"]
    k_heads = k_flat.view(seq_len, num_kv_heads, head_dim)
    print(f"  K reshape: [{seq_len}, {num_kv_heads * head_dim}] → [{seq_len}, {num_kv_heads}, {head_dim}] (view, exact)")

    # QK-norm verification
    if has_qk_norm:
        # Test: does RMSNorm on reshaped Q match q_normed?
        q_normed_cap = captures["q_normed"]
        # q_normed might be [seq, dim] or [seq, heads, dim] depending on how it's hooked
        print(f"  q_normed shape: {list(q_normed_cap.shape)}")
        print(f"  k_normed shape: {list(captures['k_normed'].shape)}")

    # RoPE verification: element-wise?
    # The o_proj_input is the attention output BEFORE o_proj.
    # It should be [seq, num_heads * head_dim] = [seq, 4096]
    print(f"  o_proj_input shape: {list(captures['o_proj_input'].shape)}")
    print(f"    (this is the raw attention output before O projection)")

    # Attention block residual
    attn_residual = captures["attn_residual"]
    o_proj_out = captures["o_proj_out"]
    attn_block_out = (attn_residual.float() + o_proj_out.float()).bfloat16()

    # Does self_attn output == o_proj output? (i.e., no extra ops after o_proj)
    d, t = count_bf16_diffs(captures["attn_module_out"].float().cpu().numpy(),
                            o_proj_out.float().cpu().numpy())
    print(f"  attn_module_out == o_proj_out? {d}/{t}",
          "*** YES ***" if d == 0 else f"NO — {d} diffs, there are extra ops")

    print()

    # ============================================================
    # Detect which attention kernel is used
    # ============================================================
    print("ATTENTION KERNEL DETECTION")
    print("-" * 60)
    print(f"  flash_sdp_enabled: {torch.backends.cuda.flash_sdp_enabled()}")
    try:
        import flash_attn
        print(f"  flash_attn installed: v{flash_attn.__version__}")
    except ImportError:
        print(f"  flash_attn: NOT installed")

    # Check what the model's forward actually dispatches
    attn_impl = getattr(model.config, "_attn_implementation", "unknown")
    print(f"  config._attn_implementation: {attn_impl}")
    print()

    # ============================================================
    # RoPE element-wise validation
    # ============================================================
    print("ROPE VALIDATION")
    print("-" * 60)

    # We can validate RoPE by re-running it and checking for element-wise behavior
    # RoPE takes (q_normed_reshaped, k_normed_reshaped, position_ids) → (q_roped, k_roped)
    # We need to capture the RoPE inputs and outputs separately
    # For now, just report what we know
    print("  RoPE is typically element-wise (sin/cos multiply).")
    print("  Full validation will be done in Phase 3 by reconstructing Q_roped from Q_normed.")
    print()

    # ============================================================
    # RMSNorm FP32 intermediate capture (for reduction tree diagnosis)
    # ============================================================
    print("RMSNorm FP32 INTERMEDIATE CAPTURE")
    print("-" * 60)
    with torch.no_grad():
        x_f32_gpu = captures["attn_residual"].float()
        gpu_sumsq = x_f32_gpu.pow(2).sum(-1, keepdim=True)
        gpu_variance = x_f32_gpu.pow(2).mean(-1, keepdim=True)
        gpu_rsqrt = torch.rsqrt(gpu_variance + eps)
        gpu_normed_f32 = x_f32_gpu * gpu_rsqrt
        gpu_normed_bf16 = gpu_normed_f32.bfloat16().float()
    print(f"  gpu_sumsq:      [{seq_len}, 1]  range=[{gpu_sumsq.min().item():.4f}, {gpu_sumsq.max().item():.4f}]")
    print(f"  gpu_variance:   [{seq_len}, 1]  range=[{gpu_variance.min().item():.6f}, {gpu_variance.max().item():.6f}]")
    print(f"  gpu_rsqrt:      [{seq_len}, 1]  range=[{gpu_rsqrt.min().item():.6f}, {gpu_rsqrt.max().item():.6f}]")
    print()

    # ============================================================
    # Save everything
    # ============================================================
    print("SAVING DATA")
    print("-" * 60)

    for name in expected_base:
        path = f"{DATA_DIR}/model_{name}.bin"
        save_bin(path, captures[name])
        print(f"  {path}  {list(captures[name].shape)}")

    save_bin(f"{DATA_DIR}/model_attn_block_out.bin", attn_block_out)
    print(f"  {DATA_DIR}/model_attn_block_out.bin  {list(attn_block_out.shape)}")

    # Save weights
    save_bin(f"{DATA_DIR}/ln_weight.bin", ln.weight.detach())
    save_bin(f"{DATA_DIR}/q_proj_w.bin", attn.q_proj.weight.detach().t())  # [in, out]
    save_bin(f"{DATA_DIR}/k_proj_w.bin", attn.k_proj.weight.detach().t())
    save_bin(f"{DATA_DIR}/v_proj_w.bin", attn.v_proj.weight.detach().t())
    save_bin(f"{DATA_DIR}/o_proj_w.bin", attn.o_proj.weight.detach().t())
    print(f"  {DATA_DIR}/ln_weight.bin  {list(ln.weight.shape)}")
    print(f"  {DATA_DIR}/q_proj_w.bin  [{hidden_dim}, {num_heads * head_dim}]")
    print(f"  {DATA_DIR}/k_proj_w.bin  [{hidden_dim}, {num_kv_heads * head_dim}]")
    print(f"  {DATA_DIR}/v_proj_w.bin  [{hidden_dim}, {num_kv_heads * head_dim}]")
    print(f"  {DATA_DIR}/o_proj_w.bin  [{num_heads * head_dim}, {hidden_dim}]")

    if has_qk_norm:
        save_bin(f"{DATA_DIR}/q_norm_weight.bin", q_norm.weight.detach())
        save_bin(f"{DATA_DIR}/k_norm_weight.bin", k_norm.weight.detach())
        print(f"  {DATA_DIR}/q_norm_weight.bin  {list(q_norm.weight.shape)}")
        print(f"  {DATA_DIR}/k_norm_weight.bin  {list(k_norm.weight.shape)}")

    # RMSNorm FP32 intermediates
    save_bin(f"{DATA_DIR}/gpu_rms_sumsq.bin", gpu_sumsq.cpu())
    save_bin(f"{DATA_DIR}/gpu_rms_variance.bin", gpu_variance.cpu())
    save_bin(f"{DATA_DIR}/gpu_rms_rsqrt.bin", gpu_rsqrt.cpu())
    save_bin(f"{DATA_DIR}/gpu_rms_normed_f32.bin", gpu_normed_f32.cpu())
    save_bin(f"{DATA_DIR}/gpu_rms_normed_bf16.bin", gpu_normed_bf16.cpu())
    print(f"  {DATA_DIR}/gpu_rms_sumsq.bin       [{seq_len}, 1]")
    print(f"  {DATA_DIR}/gpu_rms_variance.bin     [{seq_len}, 1]")
    print(f"  {DATA_DIR}/gpu_rms_rsqrt.bin        [{seq_len}, 1]")
    print(f"  {DATA_DIR}/gpu_rms_normed_f32.bin   [{seq_len}, {hidden_dim}]")
    print(f"  {DATA_DIR}/gpu_rms_normed_bf16.bin  [{seq_len}, {hidden_dim}]")

    rope_config = getattr(model.config, "rope_scaling", None) or {}
    rope_theta = getattr(model.config, "rope_theta", rope_config.get("rope_theta", 10000.0))
    rope_type = rope_config.get("rope_type", "default")

    meta = {
        "model": MODEL_NAME,
        "layer": LAYER,
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "gqa_groups": gqa_groups,
        "eps": eps,
        "has_qk_norm": has_qk_norm,
        "qk_eps": qk_eps if has_qk_norm else None,
        "fa2_captured": fa2_captured,
        "rope_theta": rope_theta,
        "rope_type": rope_type,
    }
    with open(f"{DATA_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {DATA_DIR}/meta.json")

    print()
    print(f"Phase 1 complete. Captured {len(expected_base)} tensors from layer {LAYER}.")
    print(f"Next: python3 {sys.argv[0]} cutlass")


# ============================================================
# PHASE 2: CUTLASS matmuls for projections
# ============================================================
def phase_cutlass():
    print("=" * 80)
    print("PHASE 2: CUTLASS MATMULS FOR PROJECTIONS")
    print("=" * 80)
    print()

    if not os.path.exists(CUTLASS_BIN):
        print(f"ERROR: {CUTLASS_BIN} not found.")
        return False

    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)

    M = meta["seq_len"]
    H = meta["hidden_dim"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]

    def run_cutlass(name, m, k, n, a_path, b_path, d_path, fp32=False):
        print(f"  {name}: [{m},{k}]x[{k},{n}]...", end=" ", flush=True)
        cmd = [CUTLASS_BIN, str(m), str(k), str(n), a_path, b_path, d_path]
        if fp32:
            cmd.append("fp32")
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f"FAILED: {result.stderr}")
            return False
        print("OK")
        return True

    projections = [
        ("q_proj", M, H, num_heads * head_dim, "model_rms_out", "q_proj_w", "cutlass_q_proj_out"),
        ("k_proj", M, H, num_kv_heads * head_dim, "model_rms_out", "k_proj_w", "cutlass_k_proj_out"),
        ("v_proj", M, H, num_kv_heads * head_dim, "model_rms_out", "v_proj_w", "cutlass_v_proj_out"),
        ("o_proj", M, num_heads * head_dim, H, "model_o_proj_input", "o_proj_w", "cutlass_o_proj_out"),
    ]

    for name, m, k, n, a_file, b_file, d_file in projections:
        ok = run_cutlass(name, m, k, n,
                         f"{DATA_DIR}/{a_file}.bin", f"{DATA_DIR}/{b_file}.bin",
                         f"{DATA_DIR}/{d_file}.bin")
        if not ok:
            return False
        # Also FP32 output
        run_cutlass(f"{name} (FP32)", m, k, n,
                    f"{DATA_DIR}/{a_file}.bin", f"{DATA_DIR}/{b_file}.bin",
                    f"{DATA_DIR}/{d_file}_fp32.bin", fp32=True)

    print()
    print("Phase 2 complete. CUTLASS projection intermediates saved.")
    print(f"Next: python3 {sys.argv[0]} compare")
    return True


# ============================================================
# PHASE 3: CPU emulator chain + three-way diagnostic
# ============================================================
def phase_compare():
    print("=" * 80)
    print("PHASE 3: CPU EMULATOR + THREE-WAY DIAGNOSTIC")
    print("=" * 80)
    print()

    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)

    M = meta["seq_len"]
    H = meta["hidden_dim"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]
    gqa_groups = meta["gqa_groups"]
    eps = meta["eps"]
    has_qk_norm = meta["has_qk_norm"]
    qk_eps = meta.get("qk_eps", eps)

    print(f"Layer {meta['layer']}, seq_len={M}, hidden={H}")
    print(f"Heads: {num_heads}Q / {num_kv_heads}KV, head_dim={head_dim}, GQA={gqa_groups}")
    print()

    gpu = detect_gpu()
    profile = get_profile(gpu, INPUT_FMT, OUTPUT_FMT)
    tc_emu = TensorCoreEmulator(profile)
    mufu = MUFUEmulator(gpu_name=gpu)
    print(f"GPU detected: {gpu}")
    print(f"Emulator: {profile.describe()}")

    # Load inputs
    residual = load_bin(f"{DATA_DIR}/model_attn_residual.bin", (M, H))
    ln_w = load_bin(f"{DATA_DIR}/ln_weight.bin", (H,))
    q_w = load_bin(f"{DATA_DIR}/q_proj_w.bin", (H, num_heads * head_dim))
    k_w = load_bin(f"{DATA_DIR}/k_proj_w.bin", (H, num_kv_heads * head_dim))
    v_w = load_bin(f"{DATA_DIR}/v_proj_w.bin", (H, num_kv_heads * head_dim))
    o_w = load_bin(f"{DATA_DIR}/o_proj_w.bin", (num_heads * head_dim, H))

    residual_bf16 = to_bf16_f32(residual)
    ln_w_bf16 = to_bf16_f32(ln_w)
    q_w_bf16 = to_bf16_f32(q_w)
    k_w_bf16 = to_bf16_f32(k_w)
    v_w_bf16 = to_bf16_f32(v_w)
    o_w_bf16 = to_bf16_f32(o_w)

    emu = {}
    emu_fp32 = {}
    timings = {}

    print()
    print("EMULATOR CHAIN")
    print("-" * 60)

    # ============================================================
    # Step 1: RMSNorm (same as FFN, proven 0-diff)
    # ============================================================
    # SOFTWARE STACK ASSUMPTION: Same RMSNorm as FFN (HuggingFace Qwen3RMSNorm)
    print(f"  [1/8] RMSNorm...", end=" ", flush=True)
    t0 = time.time()
    x_f32 = residual_bf16.astype(np.float32)
    emu_sumsq = emulate_sum_reduce(residual_bf16, warp_shfl_decreasing=True)
    variance = (emu_sumsq * np.float32(1.0 / H)).astype(np.float32)
    rsqrt_val = mufu.rsq((variance + np.float32(eps)).astype(np.float32))
    normed_f32 = (x_f32 * rsqrt_val).astype(np.float32)
    normed_bf16 = to_bf16_f32(normed_f32)
    rms_out = to_bf16_f32(normed_bf16 * ln_w_bf16)
    emu["rms_out"] = rms_out
    emu_fp32["rms_sumsq"] = emu_sumsq.copy()
    emu_fp32["rms_variance"] = variance.copy()
    emu_fp32["rms_rsqrt"] = rsqrt_val.copy()
    emu_fp32["rms_normed_f32"] = normed_f32.copy()
    emu_fp32["rms_normed_bf16"] = normed_bf16.copy()
    timings["rms"] = time.time() - t0
    print(f"done ({timings['rms']:.1f}s)")

    # ============================================================
    # Step 2: Q/K/V projections (matmuls, proven 0-diff)
    # ============================================================
    for proj_name, weight, out_dim in [
        ("q_proj", q_w_bf16, num_heads * head_dim),
        ("k_proj", k_w_bf16, num_kv_heads * head_dim),
        ("v_proj", v_w_bf16, num_kv_heads * head_dim),
    ]:
        print(f"  [2/8] {proj_name} [{M},{H}]x[{H},{out_dim}]...", end=" ", flush=True)
        t0 = time.time()
        raw_fp32 = tc_emu.matmul(rms_out, weight)
        emu_fp32[f"{proj_name}_out"] = raw_fp32.copy()
        emu[f"{proj_name}_out"] = to_bf16_f32(raw_fp32)
        timings[proj_name] = time.time() - t0
        print(f"done ({timings[proj_name]:.1f}s)")

    # ============================================================
    # Step 3: Reshape to per-head (exact, no rounding)
    # ============================================================
    print(f"  [3/8] Reshape to per-head...", end=" ", flush=True)
    q_heads = emu["q_proj_out"].reshape(M, num_heads, head_dim)
    k_heads = emu["k_proj_out"].reshape(M, num_kv_heads, head_dim)
    v_heads = emu["v_proj_out"].reshape(M, num_kv_heads, head_dim)
    print("done (exact)")

    # ============================================================
    # Step 4: QK-norm (RMSNorm per head, head_dim=128)
    # ============================================================
    if has_qk_norm:
        print(f"  [4/8] QK-norm (head_dim={head_dim})...", end=" ", flush=True)
        t0 = time.time()

        q_norm_w = to_bf16_f32(load_bin(f"{DATA_DIR}/q_norm_weight.bin", (head_dim,)))
        k_norm_w = to_bf16_f32(load_bin(f"{DATA_DIR}/k_norm_weight.bin", (head_dim,)))

        # RMSNorm each head independently
        # SOFTWARE STACK ASSUMPTION: QK-norm is per-head RMSNorm with head_dim=128.
        # compute_block_shape(128, M*num_heads) will determine the reduction tree params.
        q_normed = np.zeros_like(q_heads)
        k_normed = np.zeros_like(k_heads)

        for h in range(num_heads):
            qh = q_heads[:, h, :]  # [M, head_dim]
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
        timings["qk_norm"] = time.time() - t0
        print(f"done ({timings['qk_norm']:.1f}s)")
    else:
        q_normed = q_heads
        k_normed = k_heads
        print(f"  [4/8] QK-norm: SKIPPED (not present)")

    # ============================================================
    # Step 5: RoPE (element-wise)
    # ============================================================
    # MODEL-SPECIFIC: rotate_half style (split dim in half: (-x2, x1)).
    #   This is the LLaMA/Qwen family pattern. GPT-J/GPT-NeoX use interleaved.
    #   Determined by: transformers model source (modeling_qwen3.py rotate_half).
    # MODEL-SPECIFIC: rope_theta, rope_type from model config.
    # GENERAL: inv_freq = 1/(theta^(2i/d)) for rope_type="default".
    # INFERENCE-DTYPE-SPECIFIC: BF16 cast after each multiply.
    #   When model runs in BF16, all tensors are BF16. PyTorch BF16 × BF16
    #   promotes to FP32 internally, casts result back to BF16 before the add.
    #   Verified empirically: 0 diffs with cast-after-each-mul, 18% without.
    #   Would be different for FP16 or FP32 inference.
    print(f"  [5/8] RoPE...", end=" ", flush=True)
    t0 = time.time()

    rope_theta = meta.get("rope_theta", 10000.0)
    rope_type = meta.get("rope_type", "default")
    assert rope_type == "default", f"Only 'default' rope_type supported, got '{rope_type}'"

    half_dim = head_dim // 2

    # Compute inverse frequencies: inv_freq[i] = 1 / (theta ^ (2i / head_dim))
    # GENERAL for rope_type="default". inv_freq computed in FP32.
    inv_freq = np.float32(1.0) / (np.float32(rope_theta) ** (
        np.arange(0, head_dim, 2, dtype=np.float32) / np.float32(head_dim)))

    # Compute cos/sin for each position
    positions = np.arange(M, dtype=np.float32)
    freqs = np.outer(positions, inv_freq).astype(np.float32)  # [M, head_dim/2]
    cos_f32 = np.cos(freqs).astype(np.float32)
    sin_f32 = np.sin(freqs).astype(np.float32)

    # INFERENCE-DTYPE-SPECIFIC: transformers casts cos/sin to model dtype (BF16)
    # before passing to apply_rotary_pos_emb.
    # MODEL-SPECIFIC: cos/sin are duplicated [cos, cos] for rotate_half style.
    cos_rope = to_bf16_f32(np.concatenate([cos_f32, cos_f32], axis=-1))  # [M, head_dim]
    sin_rope = to_bf16_f32(np.concatenate([sin_f32, sin_f32], axis=-1))  # [M, head_dim]

    def apply_rope_head(x, cos, sin):
        """Apply rotate_half RoPE to x: [M, head_dim].
        x, cos, sin are BF16-rounded FP32.
        MODEL-SPECIFIC: rotate_half = (-x2, x1) (LLaMA/Qwen family).
        INFERENCE-DTYPE-SPECIFIC: BF16 cast after each multiply, then add.
        """
        x1 = x[:, :half_dim]
        x2 = x[:, half_dim:]
        rotated = np.concatenate([-x2, x1], axis=-1)  # rotate_half
        tmp1 = to_bf16_f32(x * cos)       # cast to model dtype after q * cos
        tmp2 = to_bf16_f32(rotated * sin)  # cast to model dtype after rot * sin
        return to_bf16_f32(tmp1 + tmp2)    # cast to model dtype after add

    # Apply RoPE to all Q and K heads
    q_roped = np.zeros_like(q_normed)
    k_roped = np.zeros_like(k_normed)

    for h in range(num_heads):
        q_roped[:, h, :] = apply_rope_head(q_normed[:, h, :], cos_rope, sin_rope)
    for h in range(num_kv_heads):
        k_roped[:, h, :] = apply_rope_head(k_normed[:, h, :], cos_rope, sin_rope)

    # V does NOT get RoPE
    v_roped = v_heads  # just the reshaped v_proj output

    emu["q_roped"] = q_roped
    emu["k_roped"] = k_roped
    emu["v_roped"] = v_roped

    timings["rope"] = time.time() - t0
    print(f"done ({timings['rope']:.1f}s)")

    # ============================================================
    # Step 6: FA2 core
    # ============================================================
    # SOFTWARE STACK ASSUMPTION: FlashAttention-2 v2.8.3, sm_80 (A100)
    # Tile config: kBlockM=128, kBlockN=64, kNWarps=4
    # MMA: SM80_16x8x16_F32BF16BF16F32_TN
    # Warp layout: 4 warps in M, 1 in N
    # Per thread: 4 rows × 16 columns of the 128×64 S tile
    # Thread column assignment: within each 8-col MMA atom, thread t owns
    #   cols {2t, 2t+1}. Across 8 atoms: thread 0 → {0,1,8,9,16,17,...,56,57}
    # reduce_max = thread_reduce + Allreduce<4> (every tile)
    # reduce_sum = thread_reduce ONLY (deferred to normalize_softmax_lse)
    # Allreduce<4> = shfl_xor(2) then shfl_xor(1)
    # exp2 argument uses FMA: exp2f(x * scale - max_scaled) = one rounding
    # Causal: reverse tile iteration
    # P cast to BF16 between QK^T and PV matmuls
    # NOTE: PV matmul is computed fresh and added to O_acc. The hardware
    #   accumulates INTO O_acc within the MMA. This may cause FP32 diffs
    #   due to blockFMA alignment, but the effect should be small.
    # NOTE: exp2f on CUDA compiles to MUFU.EX2 with --use_fast_math.
    #   We use mufu.ex2() which applies probed hardware corrections.

    print(f"  [6/8] FA2 attention core...", flush=True)
    t0 = time.time()

    fa2_captured = meta.get("fa2_captured", False)

    kBlockM = 128
    kBlockN = 64
    softmax_scale = np.float32(1.0 / math.sqrt(head_dim))
    scale_log2 = np.float32(softmax_scale * math.log2(math.e))

    # Thread column indices: within each 8-col MMA atom, thread t owns
    # cols {2t, 2t+1}. With 8 atoms across kBlockN=64:
    thread_cols = []
    for t in range(4):
        cols = []
        for atom in range(8):
            cols.extend([atom * 8 + 2 * t, atom * 8 + 2 * t + 1])
        thread_cols.append(cols)

    def fma_f32_vec(a, b, c):
        """Vectorized FMA: a[:]*b + c[:] with single rounding to FP32."""
        return (a.astype(np.float64) * np.float64(b) + c.astype(np.float64)).astype(np.float32)

    def allreduce4_max(vals):
        """Allreduce<4> for max: shfl_xor(2) then shfl_xor(1)."""
        return np.maximum(np.maximum(vals[:, 0], vals[:, 2]),
                          np.maximum(vals[:, 1], vals[:, 3]))

    def apply_causal_mask(S, m_start, n_start, m_size, n_size):
        """Set S[i,j] = -inf where row i attends to future column j."""
        rows = np.arange(m_start, m_start + m_size)[:, None]
        cols = np.arange(n_start, n_start + n_size)[None, :]
        S[rows < cols] = -np.inf

    # Use emulated post-RoPE Q/K/V from step 5
    fa2_q = q_roped
    fa2_k = k_roped
    fa2_v = v_roped

    fa2_out = np.zeros((M, num_heads, head_dim), dtype=np.float32)
    n_q_tiles = (M + kBlockM - 1) // kBlockM
    n_kv_tiles = (M + kBlockN - 1) // kBlockN

    # Per-tile diagnostic: store O_acc and rescale after each tile
    # Keyed by (qh, m_block) -> list of dicts per tile
    tile_diag = {}

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
            print(f"\n    Head {qh}/{num_heads} ({elapsed:.0f}s, ETA {eta:.0f}s)...", end="", flush=True)

        for m_block in range(n_q_tiles):
            m_start = m_block * kBlockM
            m_end = min(m_start + kBlockM, M)
            m_size = m_end - m_start
            Q_tile = Q_head[m_start:m_end]

            # Per-thread state: 4 threads each track independent row_sum
            row_max = np.full((m_size, 4), -np.inf, dtype=np.float32)
            row_sum = np.zeros((m_size, 4), dtype=np.float32)
            O_acc = np.zeros((m_size, head_dim), dtype=np.float32)

            # Causal: iterate KV tiles in reverse
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

                # QK^T via tensor core emulator
                S = tc_emu.matmul(Q_tile, K_tile.T)

                # Causal mask
                apply_causal_mask(S, m_start, n_start, m_size, n_size)

                # Pad to kBlockN if needed (partial last tile)
                if n_size < kBlockN:
                    Sp = np.full((m_size, kBlockN), -np.inf, dtype=np.float32)
                    Sp[:, :n_size] = S
                else:
                    Sp = S

                if is_first:
                    # --- reduce_max<zero_init=true> ---
                    # thread_reduce_: sequential max over thread's columns
                    for t in range(4):
                        tmax = Sp[:, thread_cols[t][0]].copy()
                        for c in thread_cols[t][1:]:
                            tmax = np.maximum(tmax, Sp[:, c])
                        row_max[:, t] = tmax
                    # quad_allreduce_ for max
                    gmax = allreduce4_max(row_max)
                    row_max[:, :] = gmax[:, None]

                    # --- scale_apply_exp2 via FMA ---
                    mscaled = np.where(gmax == -np.inf, np.float32(0.0),
                                       (gmax * scale_log2).astype(np.float32))
                    for j in range(kBlockN):
                        Sp[:, j] = mufu.ex2(fma_f32_vec(Sp[:, j], scale_log2, -mscaled)).astype(np.float32)

                    # --- reduce_sum<zero_init=true> (NO quad allreduce) ---
                    # WHY no allreduce: FA2's reduce_sum calls only thread_reduce_,
                    # not quad_allreduce_. The cross-thread sum allreduce is deferred
                    # to normalize_softmax_lse at the end. Verified in softmax.h:
                    # reduce_max calls reduce_() = thread_reduce_ + quad_allreduce_,
                    # but reduce_sum calls only thread_reduce_.
                    # thread_reduce_: sequential sum, init from first column
                    for t in range(4):
                        row_sum[:, t] = Sp[:, thread_cols[t][0]].copy()
                        for c in thread_cols[t][1:]:
                            row_sum[:, t] = (row_sum[:, t] + Sp[:, c]).astype(np.float32)

                    is_first = False
                else:
                    # --- save previous max (already allreduced) ---
                    prev_max = row_max[:, 0].copy()

                    # --- reduce_max<zero_init=false> ---
                    # thread_reduce_: merge current scores with existing max
                    for t in range(4):
                        tmax = row_max[:, t].copy()
                        for c in thread_cols[t]:
                            tmax = np.maximum(tmax, Sp[:, c])
                        row_max[:, t] = tmax
                    # quad_allreduce_ for max
                    gmax = allreduce4_max(row_max)
                    row_max[:, :] = gmax[:, None]

                    # --- rescale row_sum and O_acc ---
                    cur_max = np.where(gmax == -np.inf, np.float32(0.0), gmax)
                    rescale = mufu.ex2(((prev_max - cur_max) * scale_log2).astype(np.float32)).astype(np.float32)
                    for t in range(4):
                        row_sum[:, t] = (row_sum[:, t] * rescale).astype(np.float32)
                    O_acc = (O_acc * rescale[:, None]).astype(np.float32)

                    # --- scale_apply_exp2 via FMA ---
                    mscaled = np.where(gmax == -np.inf, np.float32(0.0),
                                       (gmax * scale_log2).astype(np.float32))
                    for j in range(kBlockN):
                        Sp[:, j] = mufu.ex2(fma_f32_vec(Sp[:, j], scale_log2, -mscaled)).astype(np.float32)

                    # --- reduce_sum<zero_init=false> (NO quad allreduce) ---
                    # thread_reduce_: accumulate DIRECTLY into row_sum
                    for t in range(4):
                        for c in thread_cols[t]:
                            row_sum[:, t] = (row_sum[:, t] + Sp[:, c]).astype(np.float32)

                # --- P → BF16 cast, then PV matmul (K-walk accumulating into O_acc) ---
                # WHY block_fma_batch instead of tc_emu.matmul:
                # FA2's gemm_rs() passes acc_o as the live MMA accumulator — the
                # PV matmul does NOT start from zero, it accumulates into the running
                # O_acc from previous KV tiles. matmul() starts from 0.0f, which gives
                # different block FMA alignment windows and thus different FP32 results.
                # Fixing this reduced FA2 diffs from 72 to 28.
                # (Verified: gemm_rs in utils.h calls cute::gemm(mma, A, B, acc) with
                # no clear(acc). SASS confirms acc_o flows through as MMA C operand.)
                P = to_bf16_f32(Sp[:, :n_size])
                nfma = tc_emu.profile.nfma
                k_per_mma = tc_emu.profile.products_per_mma
                for k in range(0, n_size, k_per_mma):
                    k_end = min(k + k_per_mma, n_size)
                    for kb in range(k, k_end, nfma):
                        bend = min(kb + nfma, k_end)
                        O_acc = tc_emu.block_fma_batch(
                            O_acc, P[:, kb:bend], V_tile[kb:bend, :])

                # --- tile diagnostic snapshot ---
                key = (qh, m_block)
                if key not in tile_diag:
                    tile_diag[key] = []
                tile_diag[key].append({
                    'n_block': n_block,
                    'O_acc': O_acc.copy(),
                    'rescale': rescale.copy() if rescale is not None else None,
                    'row_max': row_max[:, 0].copy(),
                    'row_sum': row_sum.copy(),
                })

            # --- normalize_softmax_lse: allreduce<4> sum, then divide ---
            # Allreduce order: shfl_xor(2) then shfl_xor(1), giving (t0+t2)+(t1+t3).
            # Verified against Allreduce<4>::run in FA2 utils.h:
            #   OFFSET=2: x = x + shfl_xor(x, 2) → thread 0 gets t0+t2
            #   Allreduce<2>: x = x + shfl_xor(x, 1) → thread 0 gets (t0+t2)+(t1+t3)
            # FA2 uses MUFU.RCP for 1/sum (not IEEE division). Under --use_fast_math,
            # 1.f/x compiles to rcp.approx.ftz.f32. MUFU.RCP differs from IEEE on
            # 13.2% of inputs on A100. Using IEEE division here causes ~8 BF16 diffs.
            s02 = (row_sum[:, 0] + row_sum[:, 2]).astype(np.float32)
            s13 = (row_sum[:, 1] + row_sum[:, 3]).astype(np.float32)
            total_sum = (s02 + s13).astype(np.float32)
            inv_sum = np.where((total_sum == 0) | np.isnan(total_sum),
                               np.float32(1.0),
                               mufu.rcp(total_sum))
            fa2_out[m_start:m_end, qh, :] = (O_acc * inv_sum[:, None]).astype(np.float32)

    elapsed = time.time() - t0
    print(f"\n    FA2 core done ({elapsed:.1f}s)")
    timings["fa2"] = elapsed

    # Compare against model's o_proj_input
    fa2_flat = fa2_out.reshape(M, num_heads * head_dim)
    emu["fa2_out"] = to_bf16_f32(fa2_flat)
    emu_fp32["fa2_out"] = fa2_flat.copy()
    model_fa2_out = load_bin(f"{DATA_DIR}/model_o_proj_input.bin", (M, num_heads * head_dim))
    d_fa2, t_fa2 = count_bf16_diffs(fa2_flat, model_fa2_out)
    pct_fa2 = d_fa2 / t_fa2 * 100
    print(f"    FA2 emulator vs model (cuBLAS path): {d_fa2}/{t_fa2} BF16 diffs ({pct_fa2:.2f}%)")

    # Compare against CUTLASS-path FA2 ground truth (from Phase 1.5)
    cutlass_fa2_path = f"{DATA_DIR}/cutlass_fa2_out.bin"
    if os.path.exists(cutlass_fa2_path):
        cutlass_fa2_out = load_bin(cutlass_fa2_path, (M, num_heads * head_dim))
        d_cut, t_cut = count_bf16_diffs(fa2_flat, cutlass_fa2_out)
        pct_cut = d_cut / t_cut * 100
        print(f"    FA2 emulator vs CUTLASS path:        {d_cut}/{t_cut} BF16 diffs ({pct_cut:.2f}%)")
        if d_cut == 0:
            print(f"    *** FA2 CORE: 0 DIFFS vs CUTLASS PATH ***")
    else:
        print(f"    (no CUTLASS-path FA2 ground truth — run 'rerun' phase)")

    # --- Per-tile diagnostic for FA2 diffs ---
    if os.path.exists(cutlass_fa2_path) and d_cut > 0:
        import torch as _th
        cut_3d = cutlass_fa2_out.reshape(M, num_heads, head_dim)
        emu_3d = fa2_out
        e16 = _th.tensor(emu_3d).bfloat16().view(_th.uint16).numpy()
        c16 = _th.tensor(cut_3d).bfloat16().view(_th.uint16).numpy()
        diff_mask = (e16 != c16)  # [M, num_heads, head_dim]
        print(f"\n    TILE DIAGNOSTIC for {d_cut} FA2 diffs:")
        # For each head with diffs, show per-tile O_acc progression at diff positions
        for qh in range(num_heads):
            head_diffs = diff_mask[:, qh, :]
            if not head_diffs.any():
                continue
            rows, cols = np.where(head_diffs)
            # Pick first diff position in this head
            r, c = rows[0], cols[0]
            m_block = r // kBlockM
            local_r = r - m_block * kBlockM
            key = (qh, m_block)
            tiles = tile_diag.get(key, [])
            print(f"    Head {qh}, row {r}, col {c} ({head_diffs.sum()} diffs in head):")
            print(f"      CUTLASS final = {cut_3d[r, qh, c]:.8e},  Emu final = {emu_3d[r, qh, c]:.8e}")
            for ti, snap in enumerate(tiles):
                o_val = snap['O_acc'][local_r, c]
                rmax = snap['row_max'][local_r]
                rs = f"{snap['rescale'][local_r]:.8e}" if snap['rescale'] is not None else "N/A (first)"
                print(f"      tile {ti} (n_block={snap['n_block']}): O_acc={o_val:.8e}  row_max={rmax:.8e}  rescale={rs}")

    # ============================================================
    # Step 7: O projection
    # ============================================================
    # Uses the MODEL's o_proj_input (captured from the actual forward pass)
    # to validate the o_proj matmul independently of FA2 emulation
    print(f"  [7/8] O projection [{M},{num_heads*head_dim}]x[{num_heads*head_dim},{H}]...", end=" ", flush=True)
    t0 = time.time()
    o_input = to_bf16_f32(load_bin(f"{DATA_DIR}/model_o_proj_input.bin", (M, num_heads * head_dim)))
    o_raw_fp32 = tc_emu.matmul(o_input, o_w_bf16)
    emu_fp32["o_proj_out"] = o_raw_fp32.copy()
    emu["o_proj_out"] = to_bf16_f32(o_raw_fp32)
    timings["o_proj"] = time.time() - t0
    print(f"done ({timings['o_proj']:.1f}s)")

    # ============================================================
    # Step 8: Residual add
    # ============================================================
    print(f"  [8/8] Residual add...", end=" ", flush=True)
    t0 = time.time()
    attn_block_out = to_bf16_f32(residual_bf16 + emu["o_proj_out"])
    emu["attn_block_out"] = attn_block_out
    timings["residual"] = time.time() - t0
    print(f"done ({timings['residual']:.1f}s)")

    total_time = sum(timings.values())
    print(f"\n  Total: {total_time:.1f}s")

    # Save emulator outputs for post-hoc diagnosis
    print(f"\n  Saving emulator intermediates...")
    for key, data in emu.items():
        save_bin(f"{DATA_DIR}/emu_{key}.bin", data)
    for key, data in emu_fp32.items():
        save_bin(f"{DATA_DIR}/emu_{key}_rawfp32.bin", data)
    print(f"  Saved {len(emu)} BF16 + {len(emu_fp32)} raw FP32 tensors to {DATA_DIR}/emu_*.bin")
    print()

    # ============================================================
    # THREE-WAY COMPARISON AT EVERY STAGE
    # ============================================================
    print("=" * 80)
    print("THREE-WAY DIAGNOSTIC (every intermediate)")
    print("=" * 80)
    print()

    stages = [
        ("RMSNorm out",   "rms_out",      None,                  "model_rms_out",      (M, H)),
        ("Q projection",  "q_proj_out",   "cutlass_q_proj_out",   "model_q_proj_out",   (M, num_heads * head_dim)),
        ("K projection",  "k_proj_out",   "cutlass_k_proj_out",   "model_k_proj_out",   (M, num_kv_heads * head_dim)),
        ("V projection",  "v_proj_out",   "cutlass_v_proj_out",   "model_v_proj_out",   (M, num_kv_heads * head_dim)),
        ("O projection",  "o_proj_out",   "cutlass_o_proj_out",   "model_o_proj_out",   (M, H)),
    ]

    col_w = 16
    header = f"  {'Stage':<18} {'Emu vs CUT':>{col_w}} {'Emu vs Model':>{col_w}} {'CUT vs Model':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_emu_vs_cut_zero = True
    first_diff_stage = None

    for label, emu_key, cut_file, model_file, shape in stages:
        emu_data = emu.get(emu_key)

        cut_data = None
        model_data = None
        if cut_file and os.path.exists(f"{DATA_DIR}/{cut_file}.bin"):
            cut_data = load_bin(f"{DATA_DIR}/{cut_file}.bin", shape)
        if model_file and os.path.exists(f"{DATA_DIR}/{model_file}.bin"):
            model_data = load_bin(f"{DATA_DIR}/{model_file}.bin", shape)

        def fmt(a, b):
            if a is None or b is None:
                return "N/A".center(col_w)
            d, t = count_bf16_diffs(a, b)
            pct = d / t * 100
            if d == 0:
                return f"0/{t} ***".rjust(col_w)
            return f"{d}/{t} ({pct:.2f}%)".rjust(col_w)

        evc = fmt(emu_data, cut_data)
        evm = fmt(emu_data, model_data)
        cvm = fmt(cut_data, model_data)

        if cut_data is not None and emu_data is not None:
            d, _ = count_bf16_diffs(emu_data, cut_data)
            if d > 0:
                if all_emu_vs_cut_zero:
                    first_diff_stage = label
                all_emu_vs_cut_zero = False

        print(f"  {label:<18} {evc} {evm} {cvm}")

    # QK-norm diagnostic
    if meta["has_qk_norm"] and "q_normed" in emu:
        print()
        print("QK-NORM DIAGNOSTIC")
        print("-" * 60)
        for name, key, shape in [
            ("Q normed", "q_normed", (M, num_heads, head_dim)),
            ("K normed", "k_normed", (M, num_kv_heads, head_dim)),
        ]:
            model_path = f"{DATA_DIR}/model_{key}.bin"
            cutlass_path = f"{DATA_DIR}/cutlass_{key}.bin"
            parts = []

            if os.path.exists(cutlass_path):
                cut_data = load_bin(cutlass_path, shape)
                d, t = count_bf16_diffs(emu[key], cut_data)
                tag = "***" if d == 0 else f"({d/t*100:.2f}%)"
                parts.append(f"Emu vs CUT: {d}/{t} {tag}")

            if os.path.exists(model_path):
                model_data = load_bin(model_path, shape)
                d, t = count_bf16_diffs(emu[key], model_data)
                tag = "***" if d == 0 else f"({d/t*100:.2f}%)"
                parts.append(f"Emu vs Model: {d}/{t} {tag}")

            print(f"  {name}: {', '.join(parts)}")

    # ============================================================
    # RMSNorm FP32 INTERMEDIATE DIAGNOSTIC
    # ============================================================
    print()
    print("=" * 80)
    print("RMSNorm FP32 INTERMEDIATE DIAGNOSTIC")
    print("=" * 80)
    print()

    rms_fp32_stages = [
        ("sum_of_squares", "rms_sumsq",      "gpu_rms_sumsq",      (M, 1)),
        ("variance",       "rms_variance",    "gpu_rms_variance",   (M, 1)),
        ("rsqrt",          "rms_rsqrt",       "gpu_rms_rsqrt",      (M, 1)),
        ("normed_f32",     "rms_normed_f32",  "gpu_rms_normed_f32", (M, H)),
        ("normed_bf16",    "rms_normed_bf16", "gpu_rms_normed_bf16",(M, H)),
    ]

    print(f"  {'Stage':<18} {'FP32 diffs':>16} {'BF16 diffs':>16} {'rows affected':>16}")
    print("  " + "-" * 70)

    for label, emu_key, gpu_file, shape in rms_fp32_stages:
        gpu_path = f"{DATA_DIR}/{gpu_file}.bin"
        if not os.path.exists(gpu_path):
            print(f"  {label:<18} {'N/A (run extract phase)':>50}")
            continue
        if emu_key not in emu_fp32:
            print(f"  {label:<18} {'N/A (no emulator data)':>50}")
            continue

        gpu_data = load_bin(gpu_path, shape)
        emu_data = emu_fp32[emu_key]
        total = emu_data.size

        fp32_diffs = int(np.sum(emu_data.view(np.uint32) != gpu_data.view(np.uint32)))
        bf16_d, _ = count_bf16_diffs(emu_data, gpu_data)

        if len(shape) == 2 and shape[1] > 1:
            row_mask = np.any(emu_data.view(np.uint32) != gpu_data.view(np.uint32), axis=1)
            n_rows = int(np.sum(row_mask))
            rows_str = f"{n_rows}/{shape[0]}"
        else:
            n_rows = fp32_diffs
            rows_str = f"{n_rows}/{shape[0]}"

        fp32_str = f"{fp32_diffs}/{total}".rjust(16)
        bf16_str = f"{bf16_d}/{total}".rjust(16)
        rows_str = rows_str.rjust(16)
        print(f"  {label:<18} {fp32_str} {bf16_str} {rows_str}")

        if fp32_diffs > 0:
            diff_mask = (emu_data.view(np.uint32) != gpu_data.view(np.uint32))
            diffs_abs = np.abs(emu_data.ravel()[diff_mask.ravel()] - gpu_data.ravel()[diff_mask.ravel()])
            vals_abs = np.maximum(np.abs(emu_data.ravel()[diff_mask.ravel()]),
                                  np.abs(gpu_data.ravel()[diff_mask.ravel()]))
            rel_diffs = diffs_abs / np.maximum(vals_abs, 1e-30)
            print(f"    {'':18} abs: min={diffs_abs.min():.2e} max={diffs_abs.max():.2e} med={np.median(diffs_abs):.2e}")
            print(f"    {'':18} rel: min={rel_diffs.min():.2e} max={rel_diffs.max():.2e} med={np.median(rel_diffs):.2e}")

    # Show which rows are affected
    sumsq_gpu_path = f"{DATA_DIR}/gpu_rms_sumsq.bin"
    if os.path.exists(sumsq_gpu_path) and "rms_sumsq" in emu_fp32:
        gpu_sumsq = load_bin(sumsq_gpu_path, (M, 1))
        emu_sumsq = emu_fp32["rms_sumsq"]
        bad_rows = np.where(gpu_sumsq.view(np.uint32).ravel() != emu_sumsq.view(np.uint32).ravel())[0]
        if len(bad_rows) > 0 and len(bad_rows) <= 20:
            print(f"\n  Rows with sum-of-squares FP32 mismatch: {bad_rows.tolist()}")
            for r in bad_rows[:5]:
                eg = emu_sumsq[r, 0]
                gg = gpu_sumsq[r, 0]
                print(f"    row {r}: emu={eg:.8e} gpu={gg:.8e} diff={abs(eg-gg):.2e}")
        elif len(bad_rows) > 20:
            print(f"\n  {len(bad_rows)}/{M} rows have sum-of-squares FP32 mismatch")

    # ============================================================
    # FP32 RAW ACCUMULATOR COMPARISON (projections only)
    # ============================================================
    print()
    print("=" * 80)
    print("FP32 RAW ACCUMULATOR DIAGNOSTIC (before BF16 epilogue)")
    print("=" * 80)
    print()

    fp32_stages = [
        ("q_proj",  "q_proj_out",  "cutlass_q_proj_out_fp32",  (M, num_heads * head_dim)),
        ("k_proj",  "k_proj_out",  "cutlass_k_proj_out_fp32",  (M, num_kv_heads * head_dim)),
        ("v_proj",  "v_proj_out",  "cutlass_v_proj_out_fp32",  (M, num_kv_heads * head_dim)),
        ("o_proj",  "o_proj_out",  "cutlass_o_proj_out_fp32",  (M, H)),
    ]

    print(f"  {'Stage':<14} {'FP32 bit-exact':>16} {'Hidden by BF16':>16} {'BF16 diffs':>12}")
    print("  " + "-" * 62)

    for label, emu_key, cut_fp32_file, shape in fp32_stages:
        fp32_path = f"{DATA_DIR}/{cut_fp32_file}.bin"
        if not os.path.exists(fp32_path):
            print(f"  {label:<14} {'N/A (run cutlass phase)':>48}")
            continue
        if emu_key not in emu_fp32:
            print(f"  {label:<14} {'N/A (no raw FP32 emulator data)':>48}")
            continue

        cut_fp32 = load_bin(fp32_path, shape)
        emu_raw = emu_fp32[emu_key]

        fp32_diffs = int(np.sum(emu_raw.view(np.uint32) != cut_fp32.view(np.uint32)))
        total = emu_raw.size
        bf16_diffs, _ = count_bf16_diffs(emu_raw, cut_fp32)
        hidden = fp32_diffs - bf16_diffs

        fp32_str = f"{fp32_diffs}/{total}".rjust(16)
        hidden_str = f"{hidden}".rjust(16)
        bf16_str = f"{bf16_diffs}".rjust(12)
        print(f"  {label:<14} {fp32_str} {hidden_str} {bf16_str}")

        if fp32_diffs > 0 and fp32_diffs <= 50:
            diff_mask = (emu_raw.view(np.uint32) != cut_fp32.view(np.uint32))
            indices = np.argwhere(diff_mask)
            for idx in indices[:10]:
                i, j = idx
                ev, cv = emu_raw[i,j], cut_fp32[i,j]
                rel = abs(ev - cv) / max(abs(ev), abs(cv), 1e-30)
                print(f"    [{i},{j}]: emu={ev: .8e} cut={cv: .8e} rel_diff={rel:.2e}")
        elif fp32_diffs > 50:
            diff_mask = (emu_raw.view(np.uint32) != cut_fp32.view(np.uint32))
            diffs_abs = np.abs(emu_raw[diff_mask] - cut_fp32[diff_mask])
            vals_abs = np.maximum(np.abs(emu_raw[diff_mask]), np.abs(cut_fp32[diff_mask]))
            rel_diffs = diffs_abs / np.maximum(vals_abs, 1e-30)
            print(f"    abs diff:  min={diffs_abs.min():.2e} max={diffs_abs.max():.2e} median={np.median(diffs_abs):.2e}")
            print(f"    rel diff:  min={rel_diffs.min():.2e} max={rel_diffs.max():.2e} median={np.median(rel_diffs):.2e}")

    print()
    print("=" * 80)

    if all_emu_vs_cut_zero:
        total_el = sum(emu[k].size for _, k, cf, _, _ in stages if cf and os.path.exists(f"{DATA_DIR}/{cf}.bin"))
        print(f"ALL EMULATOR vs CUTLASS COMPARISONS: 0 DIFFS")
        print(f"  {total_el:,} total elements checked across {sum(1 for _,_,cf,_,_ in stages if cf)} stages")
        print()
        print(f"The CPU emulator predicts every BF16 bit of the pre/post-attention projections")
        print(f"at layer {meta['layer']} of {meta['model']} on {M} real tokens.")
    else:
        print(f"FIRST DIFF at: {first_diff_stage}")
        print()
        print("DIFF DIAGNOSIS")
        print("-" * 60)
        for label, emu_key, cut_file, model_file, shape in stages:
            if cut_file and os.path.exists(f"{DATA_DIR}/{cut_file}.bin"):
                cut_data = load_bin(f"{DATA_DIR}/{cut_file}.bin", shape)
                emu_data = emu[emu_key]
                d, t = count_bf16_diffs(emu_data, cut_data)
                if d > 0 and d <= 20:
                    print(f"  {label}: {d} diffs")
                    e_bf16 = torch.tensor(emu_data).bfloat16()
                    c_bf16 = torch.tensor(cut_data).bfloat16()
                    diff_mask = (e_bf16 != c_bf16)
                    indices = diff_mask.nonzero().tolist()
                    for idx in indices[:20]:
                        i, j = idx
                        ev = emu_data[i, j]
                        cv = cut_data[i, j]
                        eb = e_bf16[i, j].float().item()
                        cb = c_bf16[i, j].float().item()
                        print(f"    [{i},{j}]: emu_f32={ev:.8e} cut_f32={cv:.8e} emu_bf16={eb:.8e} cut_bf16={cb:.8e}")
                elif d > 20:
                    print(f"  {label}: {d} diffs (too many to list)")

    print()
    print("=" * 80)
    print("NOTE: FA2 attention core emulation not yet implemented.")
    print("Steps 1-4 (RMSNorm, projections, reshape, QK-norm) use proven")
    print("components from ffn_chain_test.py. Step 6 (FA2) is the remaining work.")
    print("=" * 80)


# ============================================================
# PHASE 1.5: Re-run attention internals with CUTLASS projections
# ============================================================
def phase_rerun():
    """Feed CUTLASS projection outputs through GPU's actual QK-norm → RoPE → FA2.

    This produces CUTLASS-consistent ground truth at every stage downstream
    of the projections, so Phase 3 comparisons are meaningful.

    Depends on: Phase 1 (meta.json, model weights) and Phase 2 (cutlass_*_proj_out.bin).
    Produces: cutlass_q_normed.bin, cutlass_k_normed.bin, cutlass_q_roped.bin,
              cutlass_k_roped.bin, cutlass_fa2_out.bin, cutlass_o_proj_from_fa2.bin
    """
    print("=" * 80)
    print("PHASE 1.5: CUTLASS-PATH GROUND TRUTH")
    print("=" * 80)
    print()

    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)

    M = meta["seq_len"]
    H = meta["hidden_dim"]
    num_heads = meta["num_heads"]
    num_kv_heads = meta["num_kv_heads"]
    head_dim = meta["head_dim"]
    has_qk_norm = meta["has_qk_norm"]

    # Verify CUTLASS projection outputs exist
    for name in ["cutlass_q_proj_out", "cutlass_k_proj_out", "cutlass_v_proj_out"]:
        if not os.path.exists(f"{DATA_DIR}/{name}.bin"):
            print(f"ERROR: {DATA_DIR}/{name}.bin not found. Run Phase 2 first.")
            return False

    # Load CUTLASS projection outputs → GPU BF16
    cut_q = torch.tensor(load_bin(f"{DATA_DIR}/cutlass_q_proj_out.bin",
                (M, num_heads * head_dim))).bfloat16().to(DEVICE)
    cut_k = torch.tensor(load_bin(f"{DATA_DIR}/cutlass_k_proj_out.bin",
                (M, num_kv_heads * head_dim))).bfloat16().to(DEVICE)
    cut_v = torch.tensor(load_bin(f"{DATA_DIR}/cutlass_v_proj_out.bin",
                (M, num_kv_heads * head_dim))).bfloat16().to(DEVICE)
    print(f"Loaded CUTLASS projections: Q{list(cut_q.shape)} K{list(cut_k.shape)} V{list(cut_v.shape)}")

    # Reshape to per-head
    q_heads = cut_q.view(M, num_heads, head_dim)
    k_heads = cut_k.view(M, num_kv_heads, head_dim)
    v_heads = cut_v.view(M, num_kv_heads, head_dim)

    # Load model
    from transformers import AutoModelForCausalLM
    print(f"Loading {MODEL_NAME}...", end=" ", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda",
        attn_implementation="flash_attention_2"
    )
    model.eval()
    attn = model.model.layers[LAYER].self_attn
    print("done.")

    with torch.no_grad():
        # QK-norm
        if has_qk_norm:
            q_normed = attn.q_norm(q_heads)
            k_normed = attn.k_norm(k_heads)
            save_bin(f"{DATA_DIR}/cutlass_q_normed.bin", q_normed)
            save_bin(f"{DATA_DIR}/cutlass_k_normed.bin", k_normed)
            print(f"  QK-norm saved.")
        else:
            q_normed = q_heads
            k_normed = k_heads

        # RoPE — model uses [batch, heads, seq, dim] internally
        position_ids = torch.arange(M, device=DEVICE).unsqueeze(0)
        cos, sin = model.model.rotary_emb(v_heads, position_ids)
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb
        # q_normed is [seq, heads, dim] → [1, heads, seq, dim] for RoPE
        q_for_rope = q_normed.unsqueeze(0).transpose(1, 2)
        k_for_rope = k_normed.unsqueeze(0).transpose(1, 2)
        q_roped, k_roped = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos, sin)
        # Back to [seq, heads, dim]
        q_roped = q_roped.squeeze(0).transpose(0, 1)
        k_roped = k_roped.squeeze(0).transpose(0, 1)
        save_bin(f"{DATA_DIR}/cutlass_q_roped.bin", q_roped)
        save_bin(f"{DATA_DIR}/cutlass_k_roped.bin", k_roped)
        print(f"  RoPE saved.")

        # FA2 — needs [batch, seq, heads, dim]
        from flash_attn import flash_attn_func
        fa2_out = flash_attn_func(
            q_roped.unsqueeze(0), k_roped.unsqueeze(0), v_heads.unsqueeze(0),
            causal=True
        ).squeeze(0)  # [seq, heads, dim]
        fa2_flat = fa2_out.view(M, num_heads * head_dim)
        save_bin(f"{DATA_DIR}/cutlass_fa2_out.bin", fa2_flat)
        print(f"  FA2 saved: {list(fa2_flat.shape)}")

        # O projection using CUTLASS-path FA2 output
        o_out = attn.o_proj(fa2_flat.unsqueeze(0).to(DEVICE)).squeeze(0)
        save_bin(f"{DATA_DIR}/cutlass_o_proj_from_fa2.bin", o_out)
        print(f"  O proj (from CUTLASS FA2) saved.")

    print()
    print("Phase 1.5 complete. CUTLASS-path ground truth saved.")
    return True


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    global MAX_SEQ_LEN
    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        MAX_SEQ_LEN = int(sys.argv[2])
        print(f"Sequence length override: {MAX_SEQ_LEN}")

    if cmd == "extract":
        phase_extract()
    elif cmd == "cutlass":
        phase_cutlass()
    elif cmd == "rerun":
        phase_rerun()
    elif cmd == "compare":
        phase_compare()
    elif cmd == "all":
        phase_extract()
        print("\n" + "=" * 80 + "\n")
        ok = phase_cutlass()
        if ok:
            print("\n" + "=" * 80 + "\n")
            phase_rerun()
            print("\n" + "=" * 80 + "\n")
            phase_compare()
    else:
        print(f"Unknown command: {cmd}")
        print(f"Usage: python3 {sys.argv[0]} [extract|cutlass|rerun|compare|all] [seq_len]")


if __name__ == "__main__":
    main()
