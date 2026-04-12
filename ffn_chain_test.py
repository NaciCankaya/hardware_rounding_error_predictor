#!/usr/bin/env python3
"""
FFN Block Chain Test — Instrumented Version

Strategy: Don't reconstruct the architecture. OBSERVE it.
Hook into the model's actual forward pass, capture every intermediate
tensor, then see if the emulator can reproduce them.

Pipeline:
  1. Tokenize a PDF (like previous FA2 experiments)
  2. Run full model prefill to get realistic layer-20 activations
  3. Hook every operation inside the FFN block and capture intermediates
  4. Run CUTLASS matmuls on the captured RMSNorm output
  5. Run CPU emulator chain on the captured FFN residual
  6. Three-way comparison at EVERY step:
     - Emulator vs CUTLASS (target: 0 diffs)
     - Emulator vs Model (shows cuBLAS vs CUTLASS gap)
     - CUTLASS vs Model (confirms kernel difference is the only source)

Usage:
  python3 ffn_chain_test.py extract    # Phase 1: model forward + hook capture
  python3 ffn_chain_test.py cutlass    # Phase 2: CUTLASS matmuls
  python3 ffn_chain_test.py compare    # Phase 3: CPU emulator + three-way compare
  python3 ffn_chain_test.py all        # All phases
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

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ============================================================
# SOFTWARE STACK DECLARATION
# ============================================================
# This emulator assumes the following software stack.  If ANY of
# these change, the bit-exact results may break.  Each downstream
# assumption is marked with "# SOFTWARE STACK ASSUMPTION:" at
# the point of use.
#
# 1. INFERENCE FRAMEWORK: HuggingFace transformers (eager mode)
#    - Determines: cast ordering in RMSNorm, SiLU decomposition,
#      element-wise vs fused kernel boundaries
#    - Alternative stacks (vLLM, TensorRT, torch.compile) may
#      fuse operations differently
#
# 2. COMPUTE LIBRARY: PyTorch (ATen CUDA kernels)
#    - Determines: reduction tree topology (Reduce.cuh heuristics),
#      element-wise kernel behavior, type promotion rules
#    - The reduction tree depends on PyTorch VERSION because the
#      warp shuffle direction changed (~Oct 2025)
#
# 3. MATMUL KERNEL: CUTLASS (for emulator) vs cuBLAS (model)
#    - The emulator matches CUTLASS tile config, NOT cuBLAS
#    - "Emu vs Model" diffs on matmuls are expected (different kernel)
#    - "Emu vs CUTLASS" must be 0 — that's the validation target
#
# Hardware is parameterized through tc_profiles.py and is NOT
# declared here — swap GPUs by changing the profile.
# ============================================================

DEVICE = "cuda"
DATA_DIR = "ffn_chain_data"
CUTLASS_BIN = "./cutlass_gemm_flex"
MODEL_NAME = "Qwen/Qwen3-4B"
LAYER = 20  # deep layer for realistic activations

# Start short, prove the chain, then scale up.
# 64 tokens: ~30s per matmul on CPU.  2048 tokens: ~30 min per matmul.
MAX_SEQ_LEN = 64


# ============================================================
# Utilities
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
    """Find and extract text from available PDFs (same as probe_flash_attn.py)."""
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
# Tensor core emulator — uses modular tc_emulator.py
# ============================================================
from tc_profiles import get_profile, detect_gpu
from tc_emulator import TensorCoreEmulator
from emulate_pytorch_reduce import emulate_sum_reduce
from mufu_emulator import MUFUEmulator

INPUT_FMT = "bf16"   # Change for FP16/FP8 workloads
OUTPUT_FMT = "fp32"  # FP32 accumulation (standard for inference)


# ============================================================
# PHASE 1: Instrumented model forward pass
# ============================================================
def phase_extract():
    print("=" * 80)
    print("PHASE 1: INSTRUMENTED MODEL FORWARD PASS")
    print("=" * 80)
    print()

    os.makedirs(DATA_DIR, exist_ok=True)

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map="cuda"
    )
    model.eval()

    n_layers = model.config.num_hidden_layers
    assert LAYER < n_layers, f"Layer {LAYER} out of range (model has {n_layers} layers)"
    print(f"Loaded. {n_layers} layers, targeting layer {LAYER}.")
    print()

    # ============================================================
    # Architecture inspection — log EVERYTHING about the FFN
    # ============================================================
    print("ARCHITECTURE INSPECTION")
    print("-" * 60)
    print(f"  model.config.hidden_size:       {model.config.hidden_size}")
    print(f"  model.config.intermediate_size:  {model.config.intermediate_size}")
    print(f"  model.config.hidden_act:         {model.config.hidden_act}")
    print(f"  model.config.rms_norm_eps:       {model.config.rms_norm_eps}")

    layer = model.model.layers[LAYER]
    mlp = layer.mlp
    ln = layer.post_attention_layernorm

    print(f"  LayerNorm type:  {type(ln).__name__}")
    print(f"  MLP type:        {type(mlp).__name__}")
    print(f"  act_fn type:     {type(mlp.act_fn).__name__}")
    print(f"  gate_proj:       {mlp.gate_proj.weight.shape}  bias={mlp.gate_proj.bias is not None}")
    print(f"  up_proj:         {mlp.up_proj.weight.shape}  bias={mlp.up_proj.bias is not None}")
    print(f"  down_proj:       {mlp.down_proj.weight.shape}  bias={mlp.down_proj.bias is not None}")
    print(f"  ln weight shape: {ln.weight.shape}")
    print()

    hidden_dim = model.config.hidden_size
    ffn_dim = model.config.intermediate_size
    eps = model.config.rms_norm_eps

    # ============================================================
    # Tokenize PDF (like previous FA2 experiments)
    # ============================================================
    print("TOKENIZATION")
    print("-" * 60)
    text = extract_pdf_text()
    if text:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)
    else:
        print("  No PDF found, using repeated text (like FA2 fallback)")
        text = "The quick brown fox jumps over the lazy dog. " * 100
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN)

    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")
    print()

    # ============================================================
    # Hook EVERYTHING in the FFN block
    # ============================================================
    print("INSTRUMENTED FORWARD PASS")
    print("-" * 60)

    captures = {}
    hooks = []

    # 1. Input to post_attention_layernorm = FFN residual
    def hook_ln_pre(module, args):
        x = args[0]
        if isinstance(x, tuple): x = x[0]
        captures["ffn_residual"] = x.detach().squeeze(0).clone()
    hooks.append(ln.register_forward_pre_hook(hook_ln_pre))

    # 2. Output of post_attention_layernorm = RMSNorm output
    def hook_ln_post(module, args, output):
        captures["rms_out"] = output.detach().squeeze(0).clone()
    hooks.append(ln.register_forward_hook(hook_ln_post))

    # 3. gate_proj output
    def hook_gate_post(module, args, output):
        captures["gate_out"] = output.detach().squeeze(0).clone()
    hooks.append(mlp.gate_proj.register_forward_hook(hook_gate_post))

    # 4. up_proj output
    def hook_up_post(module, args, output):
        captures["up_out"] = output.detach().squeeze(0).clone()
    hooks.append(mlp.up_proj.register_forward_hook(hook_up_post))

    # 5. Input to down_proj = SiLU(gate) * up (the actual intermediate the model computed)
    def hook_down_pre(module, args):
        x = args[0]
        if isinstance(x, tuple): x = x[0]
        captures["down_input"] = x.detach().squeeze(0).clone()
    hooks.append(mlp.down_proj.register_forward_pre_hook(hook_down_pre))

    # 6. down_proj output = MLP output
    def hook_down_post(module, args, output):
        captures["down_out"] = output.detach().squeeze(0).clone()
    hooks.append(mlp.down_proj.register_forward_hook(hook_down_post))

    # Run the model
    print("  Running model forward pass...", end=" ", flush=True)
    with torch.no_grad():
        _ = model(input_ids)
    print("done.")

    # Remove hooks
    for h in hooks:
        h.remove()

    # Verify we captured everything
    expected = ["ffn_residual", "rms_out", "gate_out", "up_out", "down_input", "down_out"]
    for name in expected:
        if name not in captures:
            print(f"  ERROR: Failed to capture '{name}'")
            return
        t = captures[name]
        print(f"  {name:<16} {str(list(t.shape)):>20}  dtype={t.dtype}  "
              f"range=[{t.float().min().item():.4f}, {t.float().max().item():.4f}]")

    print()

    # ============================================================
    # Architecture verification: is the intermediate what we expect?
    # ============================================================
    print("ARCHITECTURE VERIFICATION")
    print("-" * 60)

    gate_bf16 = captures["gate_out"]
    up_bf16 = captures["up_out"]
    down_input = captures["down_input"]

    # Test: does SiLU(gate) * up == what down_proj actually received?
    with torch.no_grad():
        silu_recon = F.silu(gate_bf16.float()).bfloat16()
        intermediate_recon = (silu_recon.float() * up_bf16.float()).bfloat16()

    recon_diffs, recon_total = count_bf16_diffs(
        intermediate_recon.float().cpu().numpy(),
        down_input.float().cpu().numpy()
    )
    print(f"  SiLU(gate_out) * up_out == down_proj input?")
    print(f"    {recon_diffs}/{recon_total} BF16 diffs", end="")
    if recon_diffs == 0:
        print("  *** CONFIRMED: act_fn is SiLU, no hidden casts or fused kernels ***")
    else:
        print()
        print(f"    WARNING: {recon_diffs} diffs! The model does something different.")
        print(f"    Check if act_fn is actually SiLU, or if there's a fused SwiGLU kernel.")
        print(f"    Try: F.silu(gate.float()) * up.float() vs gate.float().silu() * up.float()")

        # Additional diagnosis: try different activation functions
        with torch.no_grad():
            gelu_recon = (F.gelu(gate_bf16.float()).bfloat16().float() * up_bf16.float()).bfloat16()
            relu_recon = (F.relu(gate_bf16.float()).bfloat16().float() * up_bf16.float()).bfloat16()
        gelu_d, _ = count_bf16_diffs(gelu_recon.float().cpu().numpy(), down_input.float().cpu().numpy())
        relu_d, _ = count_bf16_diffs(relu_recon.float().cpu().numpy(), down_input.float().cpu().numpy())
        print(f"    GELU(gate)*up vs down_input: {gelu_d} diffs")
        print(f"    ReLU(gate)*up vs down_input: {relu_d} diffs")

    # Also verify: does down_out == what we'd get from down_proj on this input?
    # (Verifies that down_proj doesn't have bias or other hidden transforms)
    with torch.no_grad():
        down_verify = mlp.down_proj(down_input.unsqueeze(0).to(DEVICE)).squeeze(0)
    down_verify_diffs, _ = count_bf16_diffs(
        down_verify.float().cpu().numpy(),
        captures["down_out"].float().cpu().numpy()
    )
    print(f"  down_proj(captured_input) == captured_output?")
    print(f"    {down_verify_diffs} diffs", end="")
    if down_verify_diffs == 0:
        print("  *** CONFIRMED: down_proj is a plain linear, no hidden ops ***")
    else:
        print(f"  WARNING: down_proj output differs on re-execution ({down_verify_diffs} diffs)")

    # Derive ffn_block_out
    ffn_residual = captures["ffn_residual"]
    down_out = captures["down_out"]
    ffn_block_out = (ffn_residual.float() + down_out.float()).bfloat16()

    print()

    # ============================================================
    # SiLU element-wise validation: GPU vs CPU vs numpy
    # ============================================================
    print("ELEMENT-WISE PRECISION VALIDATION")
    print("-" * 60)

    # GPU SiLU
    with torch.no_grad():
        silu_gpu = F.silu(gate_bf16.float().to(DEVICE)).bfloat16().float().cpu()

    # CPU torch SiLU
    silu_cpu_torch = F.silu(gate_bf16.float().cpu()).bfloat16().float()

    # CPU numpy SiLU (what the emulator will use)
    gate_np = gate_bf16.float().cpu().numpy()
    with np.errstate(over='ignore'):
        sig_np = 1.0 / (1.0 + np.exp(-gate_np.astype(np.float32)))
    silu_np = (gate_np * sig_np).astype(np.float32)

    d1, t1 = count_bf16_diffs(silu_gpu.numpy(), silu_cpu_torch.numpy())
    d2, t2 = count_bf16_diffs(silu_gpu.numpy(), silu_np)
    d3, t3 = count_bf16_diffs(silu_cpu_torch.numpy(), silu_np)

    print(f"  SiLU GPU torch vs CPU torch:    {d1}/{t1} BF16 diffs")
    print(f"  SiLU GPU torch vs CPU numpy:    {d2}/{t2} BF16 diffs")
    print(f"  SiLU CPU torch vs CPU numpy:    {d3}/{t3} BF16 diffs")

    if d2 > 0:
        pct = d2 / t2 * 100
        print(f"  WARNING: numpy SiLU differs from GPU on {d2} elements ({pct:.4f}%).")
        print(f"  The emulator uses numpy. This will propagate to down_proj.")
    else:
        print(f"  *** All three agree at BF16. Element-wise is safe. ***")

    print()

    # ============================================================
    # RMSNorm FP32 intermediate capture (for reduction tree diagnosis)
    # ============================================================
    print("RMSNorm FP32 INTERMEDIATE CAPTURE")
    print("-" * 60)
    with torch.no_grad():
        x_f32_gpu = captures["ffn_residual"].float()
        gpu_sumsq = x_f32_gpu.pow(2).sum(-1, keepdim=True)         # [M, 1] — raw sum of squares
        gpu_variance = x_f32_gpu.pow(2).mean(-1, keepdim=True)     # [M, 1] — variance = sumsq / H
        gpu_rsqrt = torch.rsqrt(gpu_variance + eps)                # [M, 1] — rsqrt(var + eps)
        gpu_normed_f32 = x_f32_gpu * gpu_rsqrt                     # [M, H] — normalized, FP32
        gpu_normed_bf16 = gpu_normed_f32.bfloat16().float()        # [M, H] — after BF16 cast
    print(f"  gpu_sumsq:      [{seq_len}, 1]  range=[{gpu_sumsq.min().item():.4f}, {gpu_sumsq.max().item():.4f}]")
    print(f"  gpu_variance:   [{seq_len}, 1]  range=[{gpu_variance.min().item():.6f}, {gpu_variance.max().item():.6f}]")
    print(f"  gpu_rsqrt:      [{seq_len}, 1]  range=[{gpu_rsqrt.min().item():.6f}, {gpu_rsqrt.max().item():.6f}]")

    print()

    # ============================================================
    # Save everything
    # ============================================================
    print("SAVING DATA")
    print("-" * 60)

    # Model intermediates (cuBLAS ground truth)
    for name in expected:
        path = f"{DATA_DIR}/model_{name}.bin"
        save_bin(path, captures[name])
        print(f"  {path}")

    save_bin(f"{DATA_DIR}/model_ffn_block_out.bin", ffn_block_out)
    save_bin(f"{DATA_DIR}/model_silu_gate.bin", silu_gpu)
    print(f"  {DATA_DIR}/model_ffn_block_out.bin")
    print(f"  {DATA_DIR}/model_silu_gate.bin")

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

    # Weights (transposed for A @ W^T = [M,K] @ [K,N])
    save_bin(f"{DATA_DIR}/ln_weight.bin", ln.weight.detach())
    save_bin(f"{DATA_DIR}/gate_w.bin", mlp.gate_proj.weight.detach().T.contiguous())
    save_bin(f"{DATA_DIR}/up_w.bin", mlp.up_proj.weight.detach().T.contiguous())
    save_bin(f"{DATA_DIR}/down_w.bin", mlp.down_proj.weight.detach().T.contiguous())
    print(f"  {DATA_DIR}/ln_weight.bin")
    print(f"  {DATA_DIR}/gate_w.bin  [{hidden_dim}, {ffn_dim}]")
    print(f"  {DATA_DIR}/up_w.bin    [{hidden_dim}, {ffn_dim}]")
    print(f"  {DATA_DIR}/down_w.bin  [{ffn_dim}, {hidden_dim}]")

    meta = {
        "seq_len": seq_len,
        "hidden_dim": hidden_dim,
        "ffn_dim": ffn_dim,
        "layer": LAYER,
        "model": MODEL_NAME,
        "eps": eps,
        "hidden_act": model.config.hidden_act,
        "arch_check_diffs": recon_diffs,
    }
    with open(f"{DATA_DIR}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  {DATA_DIR}/meta.json")

    print()
    print(f"Phase 1 complete. {len(expected)+2} tensors captured from layer {LAYER}.")
    print(f"Next: python3 {sys.argv[0]} cutlass")


# ============================================================
# PHASE 2: CUTLASS matmuls + chained ground truth
# ============================================================
def phase_cutlass():
    print("=" * 80)
    print("PHASE 2: CUTLASS MATMULS + CHAINED GROUND TRUTH")
    print("=" * 80)
    print()

    if not os.path.exists(CUTLASS_BIN):
        print(f"ERROR: {CUTLASS_BIN} not found. Compile with:")
        print(f"  cp cutlass_gemm_flex.txt cutlass_gemm_flex.cu")
        print(f"  nvcc -o cutlass_gemm_flex cutlass_gemm_flex.cu \\")
        print(f"    -I /workspace/cutlass/include \\")
        print(f"    -I /workspace/cutlass/tools/util/include \\")
        print(f"    -arch=sm_80 -std=c++17 -O2")
        return False

    with open(f"{DATA_DIR}/meta.json") as f:
        meta = json.load(f)
    M = meta["seq_len"]
    H = meta["hidden_dim"]
    F_dim = meta["ffn_dim"]

    rms_out_path = f"{DATA_DIR}/model_rms_out.bin"
    if not os.path.exists(rms_out_path):
        print(f"ERROR: {rms_out_path} not found. Run 'extract' first.")
        return False

    def run_cutlass(label, m, k, n, a_path, b_path, d_path, fp32=False):
        mode = "fp32" if fp32 else ""
        suffix = " (FP32)" if fp32 else ""
        print(f"  {label}{suffix}: [{m},{k}]x[{k},{n}]...", end=" ", flush=True)
        cmd = [CUTLASS_BIN, str(m), str(k), str(n), a_path, b_path, d_path]
        if fp32:
            cmd.append("fp32")
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"FAILED: {r.stderr.strip()}")
            return False
        print("OK")
        return True

    # Step 1: CUTLASS gate_proj (BF16 + FP32)
    ok = run_cutlass("gate_proj", M, H, F_dim,
                     rms_out_path, f"{DATA_DIR}/gate_w.bin",
                     f"{DATA_DIR}/cutlass_gate_out.bin")
    if not ok: return False
    run_cutlass("gate_proj", M, H, F_dim,
                rms_out_path, f"{DATA_DIR}/gate_w.bin",
                f"{DATA_DIR}/cutlass_gate_out_fp32.bin", fp32=True)

    # Step 2: CUTLASS up_proj (BF16 + FP32)
    ok = run_cutlass("up_proj", M, H, F_dim,
                     rms_out_path, f"{DATA_DIR}/up_w.bin",
                     f"{DATA_DIR}/cutlass_up_out.bin")
    if not ok: return False
    run_cutlass("up_proj", M, H, F_dim,
                rms_out_path, f"{DATA_DIR}/up_w.bin",
                f"{DATA_DIR}/cutlass_up_out_fp32.bin", fp32=True)

    # Step 3: SiLU(gate) * up on GPU
    print(f"  SiLU(gate)*up...", end=" ", flush=True)
    gate_cut = load_bin(f"{DATA_DIR}/cutlass_gate_out.bin", (M, F_dim))
    up_cut = load_bin(f"{DATA_DIR}/cutlass_up_out.bin", (M, F_dim))

    gate_bf16 = torch.tensor(gate_cut, device=DEVICE).bfloat16()
    up_bf16 = torch.tensor(up_cut, device=DEVICE).bfloat16()

    with torch.no_grad():
        # SiLU kernel returns BF16 (upcasts to FP32 internally, casts output to BF16).
        # The BF16 cast here is LOAD-BEARING — same lesson as RMSNorm cast ordering.
        silu_gate = F.silu(gate_bf16.float()).bfloat16()
        intermediate = (silu_gate.float() * up_bf16.float()).bfloat16()

    save_bin(f"{DATA_DIR}/cutlass_silu_gate.bin", silu_gate.cpu())
    save_bin(f"{DATA_DIR}/cutlass_down_input.bin", intermediate.cpu())
    print("OK")

    # Step 4: CUTLASS down_proj with chained intermediate (BF16 + FP32)
    ok = run_cutlass("down_proj", M, F_dim, H,
                     f"{DATA_DIR}/cutlass_down_input.bin", f"{DATA_DIR}/down_w.bin",
                     f"{DATA_DIR}/cutlass_down_out.bin")
    if not ok: return False
    run_cutlass("down_proj", M, F_dim, H,
                f"{DATA_DIR}/cutlass_down_input.bin", f"{DATA_DIR}/down_w.bin",
                f"{DATA_DIR}/cutlass_down_out_fp32.bin", fp32=True)

    # Step 5: Residual add
    print(f"  residual add...", end=" ", flush=True)
    residual = load_bin(f"{DATA_DIR}/model_ffn_residual.bin", (M, H))
    down_out = load_bin(f"{DATA_DIR}/cutlass_down_out.bin", (M, H))
    ffn_out = to_bf16_f32(
        torch.tensor(residual).bfloat16().float().numpy() +
        torch.tensor(down_out).bfloat16().float().numpy()
    )
    save_bin(f"{DATA_DIR}/cutlass_ffn_block_out.bin", ffn_out)
    print("OK")

    print()
    print("Phase 2 complete. CUTLASS chain intermediates saved.")
    print(f"Next: python3 {sys.argv[0]} compare")
    return True


# ============================================================
# PHASE 3: CPU emulator + three-way comparison
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
    F_dim = meta["ffn_dim"]
    eps = meta["eps"]

    print(f"Layer {meta['layer']}, seq_len={M}, hidden={H}, ffn={F_dim}")
    print(f"Activation: {meta.get('hidden_act','?')}, eps: {eps}")
    # Rough estimate: ~0.5s per 1M multiply-accumulate ops on one core
    gate_ops = M * H * F_dim      # gate_proj
    up_ops = M * H * F_dim        # up_proj
    down_ops = M * F_dim * H      # down_proj
    total_ops = gate_ops + up_ops + down_ops
    est_secs = total_ops * 0.5e-6
    print(f"Estimated CPU emulator time: ~{est_secs:.0f}s for 3 matmuls ({total_ops/1e6:.0f}M MAC ops)")
    print()

    gpu = detect_gpu()
    profile = get_profile(gpu, INPUT_FMT, OUTPUT_FMT)
    tc_emu = TensorCoreEmulator(profile)
    mufu = MUFUEmulator(gpu_name=gpu)
    print(f"GPU detected: {gpu}")
    print(f"Emulator: {profile.describe()}")

    # Load inputs
    residual = load_bin(f"{DATA_DIR}/model_ffn_residual.bin", (M, H))
    ln_w = load_bin(f"{DATA_DIR}/ln_weight.bin", (H,))
    gate_w = load_bin(f"{DATA_DIR}/gate_w.bin", (H, F_dim))
    up_w = load_bin(f"{DATA_DIR}/up_w.bin", (H, F_dim))
    down_w = load_bin(f"{DATA_DIR}/down_w.bin", (F_dim, H))

    residual_bf16 = to_bf16_f32(residual)
    ln_w_bf16 = to_bf16_f32(ln_w)
    gate_w_bf16 = to_bf16_f32(gate_w)
    up_w_bf16 = to_bf16_f32(up_w)
    down_w_bf16 = to_bf16_f32(down_w)

    # ============================================================
    # Emulator chain with full intermediate logging
    # ============================================================
    emu = {}       # BF16 intermediates (for BF16 comparison)
    emu_fp32 = {}  # raw FP32 matmul accumulators (for FP32 comparison)
    timings = {}

    print()
    print("EMULATOR CHAIN")
    print("-" * 60)

    # 1. RMSNorm: normalize FP32 → cast BF16 → weight multiply BF16
    # SOFTWARE STACK ASSUMPTION: HuggingFace transformers Qwen3RMSNorm
    # implements this as:
    #   hidden_states = hidden_states.to(torch.float32)
    #   variance = hidden_states.pow(2).mean(-1, keepdim=True)
    #   hidden_states = hidden_states * torch.rsqrt(variance + eps)
    #   return weight * hidden_states.to(input_dtype)
    # This means .pow(2) and .mean(-1) are TWO SEPARATE kernel launches
    # (not fused), and the cast ordering is: normalize in FP32, cast to
    # BF16, then weight multiply in BF16.
    # A different framework (vLLM, TensorRT) may fuse these differently.
    print(f"  [1/7] RMSNorm...", end=" ", flush=True)
    t0 = time.time()
    x_f32 = residual_bf16.astype(np.float32)

    # SOFTWARE STACK ASSUMPTION: The sum-of-squares uses PyTorch's generic
    # reduce_kernel (Reduce.cuh).  The tree topology (block shape, vectorize,
    # shuffle direction) is derived from Reduce.cuh heuristics.  See
    # emulate_pytorch_reduce.py for the full derivation and all sub-assumptions.
    #
    # SOFTWARE STACK ASSUMPTION: warp_shfl_decreasing=True requires PyTorch
    # >= ~2.7 (Oct 2025 change).  For PyTorch <= 2.6, use False.
    emu_sumsq = emulate_sum_reduce(residual_bf16, warp_shfl_decreasing=True)

    # SOFTWARE STACK ASSUMPTION: PyTorch's MeanOps::project() computes
    # acc / factor, but the CUDA compiler (nvcc) optimizes float division
    # by a compile-time-known constant into multiply-by-reciprocal:
    #   acc / 2560.0f  →  acc * (1.0f / 2560.0f)
    # These are NOT bit-identical in FP32.  1.0f/2560 = 0x3A800000...
    # which has its own rounding, so (x * recip) != (x / N) on ~20%
    # of values.  Empirically confirmed: multiply-by-reciprocal gives
    # 256/256 FP32 match, true division gives only 201/256.
    variance = (emu_sumsq * np.float32(1.0 / H)).astype(np.float32)

    # HARDWARE DEPENDENCY: torch.rsqrt on CUDA compiles to bare MUFU.RSQ
    # on SM 8.0 (A100) — a single hardware instruction with no Newton-Raphson
    # refinement.  Its rounding is deterministic but NOT IEEE correctly-rounded
    # (up to ±1 ULP, rarely ±2).  The correction is captured in probed lookup
    # tables, one-time per GPU architecture.  See mufu_emulator.py.
    rsqrt_val = mufu.rsq((variance + np.float32(eps)).astype(np.float32))

    # SOFTWARE STACK ASSUMPTION: Cast ordering (from HuggingFace transformers):
    #   1. x_f32 * rsqrt → FP32
    #   2. Cast to BF16
    #   3. Multiply by weight in BF16
    # Determined empirically in probe_rmsnorm_rsqrt.py (0 diffs with this order).
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

    # 2. gate_proj: [M, H] × [H, F_dim]
    # SOFTWARE STACK ASSUMPTION: Matmul emulator uses CUTLASS tile config
    # (sequential K-iteration, specific tile shape compiled into cutlass_gemm_flex).
    # The model's actual cuBLAS kernel uses a DIFFERENT tile config, so
    # "Emu vs Model" diffs are expected.  "Emu vs CUTLASS" must be 0.
    print(f"  [2/7] gate_proj [{M},{H}]x[{H},{F_dim}]...", end=" ", flush=True)
    t0 = time.time()
    gate_raw_fp32 = tc_emu.matmul(rms_out, gate_w_bf16)
    emu_fp32["gate_out"] = gate_raw_fp32.copy()
    gate_out = to_bf16_f32(gate_raw_fp32)
    emu["gate_out"] = gate_out
    timings["gate"] = time.time() - t0
    print(f"done ({timings['gate']:.1f}s)")

    # 3. up_proj: [M, H] × [H, F_dim]
    print(f"  [3/7] up_proj [{M},{H}]x[{H},{F_dim}]...", end=" ", flush=True)
    t0 = time.time()
    up_raw_fp32 = tc_emu.matmul(rms_out, up_w_bf16)
    emu_fp32["up_out"] = up_raw_fp32.copy()
    up_out = to_bf16_f32(up_raw_fp32)
    emu["up_out"] = up_out
    timings["up"] = time.time() - t0
    print(f"done ({timings['up']:.1f}s)")

    # 4. SiLU(gate)
    # SOFTWARE STACK ASSUMPTION: PyTorch computes SiLU as x * sigmoid(x)
    # using a vectorized_elementwise_kernel in FP32.  Phase 1 verifies
    # this matches between GPU, CPU torch, and CPU numpy at BF16.
    # A fused SwiGLU kernel would combine SiLU + multiply into one op
    # with different intermediate precision.
    # Tied to: HuggingFace transformers (separate SiLU, not fused SwiGLU).
    print(f"  [4/7] SiLU...", end=" ", flush=True)
    t0 = time.time()
    gate_f32 = gate_out.astype(np.float32)
    with np.errstate(over='ignore'):  # exp(-x) overflows for large x, result is 0.0 which is correct
        sigmoid_val = 1.0 / (1.0 + np.exp(-gate_f32))
    silu_f32 = (gate_f32 * sigmoid_val).astype(np.float32)
    silu_gate = to_bf16_f32(silu_f32)
    emu["silu_gate"] = silu_gate
    timings["silu"] = time.time() - t0
    print(f"done ({timings['silu']:.1f}s)")

    # 5. SiLU(gate) * up
    #    SOFTWARE STACK ASSUMPTION: HuggingFace transformers Qwen3MLP does:
    #      self.act_fn(self.gate_proj(x)) * self.up_proj(x)
    #    The SiLU output is cast to BF16 BEFORE the multiply (because it's
    #    stored as a BF16 tensor), then BF16 * BF16 → BF16.
    #    This BF16 cast after SiLU is LOAD-BEARING — same class of bug as
    #    the RMSNorm cast ordering.  Using FP32 SiLU directly in the multiply
    #    gives ~27% diffs.
    #    Tied to: PyTorch's type promotion (BF16 * BF16 upcasts to FP32,
    #    multiplies, downcasts) and HuggingFace's MLP implementation.
    print(f"  [5/7] SiLU*up...", end=" ", flush=True)
    t0 = time.time()
    down_input = to_bf16_f32(silu_gate.astype(np.float32) * up_out.astype(np.float32))
    emu["down_input"] = down_input
    timings["mul"] = time.time() - t0
    print(f"done ({timings['mul']:.1f}s)")

    # 6. down_proj: [M, F_dim] × [F_dim, H]
    print(f"  [6/7] down_proj [{M},{F_dim}]x[{F_dim},{H}]...", end=" ", flush=True)
    t0 = time.time()
    down_raw_fp32 = tc_emu.matmul(down_input, down_w_bf16)
    emu_fp32["down_out"] = down_raw_fp32.copy()
    down_out = to_bf16_f32(down_raw_fp32)
    emu["down_out"] = down_out
    timings["down"] = time.time() - t0
    print(f"done ({timings['down']:.1f}s)")

    # 7. Residual add
    print(f"  [7/7] residual add...", end=" ", flush=True)
    t0 = time.time()
    ffn_block_out = to_bf16_f32(residual_bf16 + down_out)
    emu["ffn_block_out"] = ffn_block_out
    timings["res"] = time.time() - t0
    print(f"done ({timings['res']:.1f}s)")

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
        ("RMSNorm out",    "rms_out",        None,                      "model_rms_out",        (M, H)),
        ("gate_proj out",  "gate_out",       "cutlass_gate_out",         "model_gate_out",       (M, F_dim)),
        ("up_proj out",    "up_out",         "cutlass_up_out",           "model_up_out",         (M, F_dim)),
        ("SiLU(gate)",     "silu_gate",      "cutlass_silu_gate",        "model_silu_gate",      (M, F_dim)),
        ("SiLU*up",        "down_input",     "cutlass_down_input",       "model_down_input",     (M, F_dim)),
        ("down_proj out",  "down_out",       "cutlass_down_out",         "model_down_out",       (M, H)),
        ("FFN block out",  "ffn_block_out",  "cutlass_ffn_block_out",    "model_ffn_block_out",  (M, H)),
    ]

    col_w = 16
    header = f"  {'Stage':<18} {'Emu vs CUT':>{col_w}} {'Emu vs Model':>{col_w}} {'CUT vs Model':>{col_w}}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    all_emu_vs_cut_zero = True
    first_diff_stage = None

    for label, emu_key, cut_file, model_file, shape in stages:
        emu_data = emu[emu_key]

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

        if cut_data is not None:
            d, _ = count_bf16_diffs(emu_data, cut_data)
            if d > 0:
                if all_emu_vs_cut_zero:
                    first_diff_stage = label
                all_emu_vs_cut_zero = False

        print(f"  {label:<18} {evc} {evm} {cvm}")

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

        # FP32 bit-exact
        fp32_diffs = int(np.sum(emu_data.view(np.uint32) != gpu_data.view(np.uint32)))
        # BF16 diffs
        bf16_d, _ = count_bf16_diffs(emu_data, gpu_data)
        # Rows affected
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

        # Show worst-case diffs
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
    # FP32 RAW ACCUMULATOR COMPARISON (matmuls only)
    # ============================================================
    print()
    print("=" * 80)
    print("FP32 RAW ACCUMULATOR DIAGNOSTIC (before BF16 epilogue)")
    print("=" * 80)
    print()

    fp32_stages = [
        ("gate_proj", "gate_out", "cutlass_gate_out_fp32", (M, F_dim)),
        ("up_proj",   "up_out",   "cutlass_up_out_fp32",   (M, F_dim)),
        ("down_proj", "down_out", "cutlass_down_out_fp32", (M, H)),
    ]

    print(f"  {'Stage':<14} {'FP32 bit-exact':>16} {'Hidden by BF16':>16} {'BF16 diffs':>12}")
    print("  " + "-" * 62)

    for label, emu_key, cut_fp32_file, shape in fp32_stages:
        fp32_path = f"{DATA_DIR}/{cut_fp32_file}.bin"
        if not os.path.exists(fp32_path):
            print(f"  {label:<14} {'N/A (run cutlass phase with updated binary)':>48}")
            continue

        if emu_key not in emu_fp32:
            print(f"  {label:<14} {'N/A (no raw FP32 emulator data)':>48}")
            continue

        cut_fp32 = load_bin(fp32_path, shape)
        emu_raw = emu_fp32[emu_key]

        # FP32 bit-exact comparison
        fp32_diffs = int(np.sum(emu_raw.view(np.uint32) != cut_fp32.view(np.uint32)))
        total = emu_raw.size

        # How many of those FP32 diffs survive BF16 rounding?
        bf16_diffs, _ = count_bf16_diffs(emu_raw, cut_fp32)
        hidden = fp32_diffs - bf16_diffs

        fp32_str = f"{fp32_diffs}/{total}".rjust(16)
        hidden_str = f"{hidden}".rjust(16)
        bf16_str = f"{bf16_diffs}".rjust(12)
        print(f"  {label:<14} {fp32_str} {hidden_str} {bf16_str}")

        # If there are FP32 diffs, show worst cases
        if fp32_diffs > 0 and fp32_diffs <= 50:
            diff_mask = (emu_raw.view(np.uint32) != cut_fp32.view(np.uint32))
            indices = np.argwhere(diff_mask)
            for idx in indices[:10]:
                i, j = idx
                ev, cv = emu_raw[i,j], cut_fp32[i,j]
                rel = abs(ev - cv) / max(abs(ev), abs(cv), 1e-30)
                print(f"    [{i},{j}]: emu={ev: .8e} cut={cv: .8e} rel_diff={rel:.2e}")
        elif fp32_diffs > 50:
            # Statistical summary
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
        print(f"The CPU emulator predicts every BF16 bit of a full FFN block")
        print(f"at layer {meta['layer']} of {meta['model']} on {M} real tokens.")
    else:
        print(f"FIRST DIFF at: {first_diff_stage}")
        print()

        # Auto-diagnose: show exact positions and values of diffs
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
                    print(f"  {label}: {d} diffs (too many to list, saved to {DATA_DIR}/emu_{emu_key}.bin)")

        print()
        print("Diagnosis guide:")
        print("  RMSNorm diffs     → cast ordering wrong (normalize FP32, cast BF16, weight BF16)")
        print("  gate/up diffs     → matmul emulator issue (should not happen, proven 0-diff)")
        print("  SiLU diffs        → CPU exp() != GPU exp() at BF16 (check Phase 1 validation)")
        print("  SiLU*up diffs     → multiply precision or cast ordering")
        print("  down_proj diffs   → diffs in its input cascaded, or tile config mismatch")
        print("  FFN block diffs   → residual add issue (should not happen, element-wise)")


# ============================================================
# Main
# ============================================================
def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1].lower()

    # Optional: override MAX_SEQ_LEN from command line
    # e.g. python3 ffn_chain_test.py all 2048
    global MAX_SEQ_LEN
    if len(sys.argv) >= 3 and sys.argv[2].isdigit():
        MAX_SEQ_LEN = int(sys.argv[2])
        print(f"Sequence length override: {MAX_SEQ_LEN}")

    if cmd == "extract":
        phase_extract()
    elif cmd == "cutlass":
        phase_cutlass()
    elif cmd == "compare":
        phase_compare()
    elif cmd == "all":
        phase_extract()
        print("\n" + "=" * 80 + "\n")
        ok = phase_cutlass()
        if ok:
            print("\n" + "=" * 80 + "\n")
            phase_compare()
    else:
        print(f"Unknown command: {cmd}")
        print("Usage: python3 ffn_chain_test.py [extract|cutlass|compare|all] [seq_len]")
        print(f"  Default seq_len: {MAX_SEQ_LEN} (increase after first proof run)")


if __name__ == "__main__":
    main()
