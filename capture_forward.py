#!/usr/bin/env python3
"""
capture_forward.py — GPU-side CUTLASS-consistent forward pass capture.

Runs a full Qwen3-4B forward pass where every matmul (Q/K/V/O, gate/up/down,
LM-head) goes through the cutlass_gemm_flex binary instead of cuBLAS.  All
other ops (RMSNorm, QK-norm, RoPE, FlashAttention-2, residual adds, SiLU,
element-wise multiplies) use the GPU's native kernels — these are the same
ones the CPU emulator in block_emulators.py models bit-exactly.

The result is a ground-truth trace that the emulator can reproduce to 0
BF16 diffs end-to-end.  (cuBLAS is deliberately NOT in the comparison
loop — the emulator was built against CUTLASS tile config, so cuBLAS
intermediates would show expected-but-misleading ~0.2% diffs per matmul.)

Usage:
    python3 capture_forward.py [seq_len] [output_dir]

    seq_len     maximum token count (default 500)
    output_dir  where to write capture data (default ./forward_capture)

Requires:
    ./cutlass_gemm_flex  — built from cutlass_gemm_flex.cu.  If missing,
                            this script prints the nvcc build command and aborts.
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import glob

import numpy as np
import torch
import torch.nn.functional as F

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_SEQ_LEN = 500
DEFAULT_OUT_DIR = "forward_capture"
CUTLASS_BIN = "./cutlass_gemm_flex"


# ----------------------------------------------------------------------------
# CUTLASS subprocess helper
# ----------------------------------------------------------------------------
def _save_tensor_bf16_as_f32(t, path):
    """Write a GPU BF16 tensor to disk as FP32 (values BF16-representable).
    cutlass_gemm_flex reads FP32 files and treats values as BF16 input.
    """
    arr = t.detach().float().cpu().contiguous().numpy()
    np.ascontiguousarray(arr, dtype=np.float32).tofile(path)


def _load_bin(path, shape):
    return np.fromfile(path, dtype=np.float32).reshape(shape)


_cutlass_call_count = [0]
_cutlass_time_total = [0.0]


def cutlass_mm(A, B, tmpdir):
    """Run cutlass_gemm_flex on [M, K] @ [K, N] = [M, N].

    A, B : torch.Tensor on GPU, dtype bfloat16.  A is [M, K], B is [K, N]
           (i.e. the second operand is the transposed weight, NOT the raw
           nn.Linear weight which is [N, K]).
    tmpdir : persistent tempdir path for the intermediate .bin files.

    Returns torch.Tensor on GPU, dtype bfloat16, shape [M, N].
    """
    assert A.dtype == torch.bfloat16 and B.dtype == torch.bfloat16
    M, K = A.shape
    K2, N = B.shape
    assert K == K2, f"K mismatch: A {A.shape} vs B {B.shape}"

    idx = _cutlass_call_count[0]
    _cutlass_call_count[0] += 1

    a_path = os.path.join(tmpdir, f"A_{idx}.bin")
    b_path = os.path.join(tmpdir, f"B_{idx}.bin")
    d_path = os.path.join(tmpdir, f"D_{idx}.bin")

    _save_tensor_bf16_as_f32(A, a_path)
    _save_tensor_bf16_as_f32(B, b_path)

    t0 = time.time()
    result = subprocess.run(
        [CUTLASS_BIN, str(M), str(K), str(N), a_path, b_path, d_path],
        capture_output=True, text=True, timeout=300,
    )
    _cutlass_time_total[0] += time.time() - t0

    if result.returncode != 0:
        raise RuntimeError(
            f"cutlass_gemm_flex failed on [{M},{K}]x[{K},{N}]:\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}"
        )

    D = _load_bin(d_path, (M, N))
    # Clean up the three .bin files immediately to keep tempdir small
    for p in (a_path, b_path, d_path):
        try:
            os.remove(p)
        except OSError:
            pass

    return torch.tensor(D, dtype=torch.bfloat16, device=DEVICE)


# ----------------------------------------------------------------------------
# Output helper: save a BF16 tensor (or any tensor) as FP32 .bin
# ----------------------------------------------------------------------------
def save_bin(path, data):
    if isinstance(data, torch.Tensor):
        data = data.detach().float().cpu().contiguous().numpy()
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)


# ----------------------------------------------------------------------------
# Input text (reuse the literature PDF the chain tests used)
# ----------------------------------------------------------------------------
def extract_pdf_text():
    pdf_paths = []
    for d in ["/workspace", "/mnt/user-data/uploads", "/mnt/project", "/mnt/data"]:
        if os.path.isdir(d):
            pdf_paths.extend(glob.glob(os.path.join(d, "**/*.pdf"), recursive=True))
    for pdf_path in sorted(set(pdf_paths), key=os.path.getsize, reverse=True):
        try:
            r = subprocess.run(
                ["pdftotext", pdf_path, "-"],
                capture_output=True, text=True, timeout=30
            )
            if r.returncode == 0 and len(r.stdout.strip()) > 500:
                print(f"  Using PDF (pdftotext): {pdf_path}")
                return r.stdout
        except Exception:
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


# ----------------------------------------------------------------------------
# Main capture
# ----------------------------------------------------------------------------
def capture(max_seq_len=DEFAULT_SEQ_LEN, out_dir=DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("CUTLASS-CONSISTENT FORWARD PASS CAPTURE")
    print(f"  model:   {MODEL_NAME}")
    print(f"  max_seq: {max_seq_len}")
    print(f"  out_dir: {out_dir}")
    print("=" * 80)
    print()

    if not os.path.exists(CUTLASS_BIN):
        print(f"ERROR: {CUTLASS_BIN} not found.")
        print("Build it with:")
        print("  nvcc -o cutlass_gemm_flex cutlass_gemm_flex.cu \\")
        print("    -I /workspace/cutlass/include \\")
        print("    -I /workspace/cutlass/tools/util/include \\")
        print("    -arch=sm_80 -std=c++17 -O2")
        sys.exit(1)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    cfg = model.config
    n_layers = cfg.num_hidden_layers
    H = cfg.hidden_size
    ffn_dim = cfg.intermediate_size
    num_heads = cfg.num_attention_heads
    num_kv_heads = cfg.num_key_value_heads
    head_dim = cfg.head_dim
    eps = cfg.rms_norm_eps
    gqa_groups = num_heads // num_kv_heads
    rope_config = getattr(cfg, "rope_scaling", None) or {}
    rope_theta = getattr(cfg, "rope_theta", rope_config.get("rope_theta", 10000.0))
    rope_type = rope_config.get("rope_type", "default")

    attn0 = model.model.layers[0].self_attn
    has_qk_norm = getattr(attn0, "q_norm", None) is not None
    qk_eps = None
    if has_qk_norm:
        qk_eps = getattr(
            attn0.q_norm, "variance_epsilon",
            getattr(attn0.q_norm, "eps", eps),
        )

    assert rope_type == "default", f"Only rope_type=default supported, got {rope_type!r}"
    assert cfg.hidden_act == "silu", f"Only silu FFN supported, got {cfg.hidden_act!r}"

    print(f"  {n_layers} layers, hidden={H}, ffn={ffn_dim}")
    print(f"  heads={num_heads}Q/{num_kv_heads}KV, head_dim={head_dim}, GQA={gqa_groups}")
    print(f"  has_qk_norm={has_qk_norm}, rope_theta={rope_theta}")
    print()

    # Tokenize
    print("TOKENIZATION")
    print("-" * 60)
    text = extract_pdf_text()
    if text is None:
        print("  No PDF found, using repeated text")
        text = "The quick brown fox jumps over the lazy dog. " * 200
    tokens = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_seq_len
    )
    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")
    print()

    # ------------------------------------------------------------------------
    # Compute RoPE cos/sin once (GPU, BF16-cast)
    # ------------------------------------------------------------------------
    print("COMPUTING RoPE cos/sin (GPU)...")
    with torch.no_grad():
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        dummy_v = torch.zeros(
            1, seq_len, num_kv_heads, head_dim,
            dtype=torch.bfloat16, device=DEVICE,
        )
        cos_gpu, sin_gpu = model.model.rotary_emb(dummy_v, position_ids)
        # cos_gpu: [1, seq_len, head_dim] (duplicated [cos,cos])
    np.save(os.path.join(out_dir, "rope_cos_gpu.npy"),
            cos_gpu.squeeze(0).float().cpu().numpy())
    np.save(os.path.join(out_dir, "rope_sin_gpu.npy"),
            sin_gpu.squeeze(0).float().cpu().numpy())
    print(f"  rope_cos_gpu.npy / rope_sin_gpu.npy saved")
    print()

    # ------------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------------
    print("EMBEDDING")
    print("-" * 60)
    with torch.no_grad():
        embed_out = model.model.embed_tokens(input_ids).squeeze(0)  # [seq, H]
    save_bin(os.path.join(out_dir, "embed_out.bin"), embed_out)
    print(f"  embed_out.bin  [{seq_len}, {H}]")
    print()

    # ------------------------------------------------------------------------
    # Per-layer forward pass with CUTLASS substitution
    # ------------------------------------------------------------------------
    print("LAYER-BY-LAYER FORWARD PASS")
    print("-" * 60)

    tmpdir = tempfile.mkdtemp(prefix="cutlass_capture_")
    print(f"  CUTLASS tempdir: {tmpdir}")
    print()

    t_capture_start = time.time()
    x = embed_out  # [seq, H] bfloat16

    def _save_rms_sub_intermediates(prefix, residual):
        """Compute GPU RMSNorm FP32 sub-intermediates for diagnostic use."""
        x_f32 = residual.float()
        sumsq = x_f32.pow(2).sum(-1, keepdim=True)          # [seq, 1]
        variance = x_f32.pow(2).mean(-1, keepdim=True)
        rsqrt = torch.rsqrt(variance + eps)
        normed_f32 = x_f32 * rsqrt
        normed_bf16 = normed_f32.bfloat16().float()
        save_bin(os.path.join(out_dir, f"{prefix}_rms_sumsq.bin"), sumsq)
        save_bin(os.path.join(out_dir, f"{prefix}_rms_variance.bin"), variance)
        save_bin(os.path.join(out_dir, f"{prefix}_rms_rsqrt.bin"), rsqrt)
        save_bin(os.path.join(out_dir, f"{prefix}_rms_normed_f32.bin"), normed_f32)
        save_bin(os.path.join(out_dir, f"{prefix}_rms_normed_bf16.bin"), normed_bf16)

    with torch.no_grad():
        for i in range(n_layers):
            t_layer = time.time()
            layer = model.model.layers[i]
            attn = layer.self_attn

            # ============ Attention block ============
            attn_residual = x
            save_bin(os.path.join(out_dir, f"L{i:02d}_attn_residual.bin"), attn_residual)

            # GPU RMSNorm (input_layernorm)
            rms_out = layer.input_layernorm(x)  # [seq, H]
            save_bin(os.path.join(out_dir, f"L{i:02d}_rms_out.bin"), rms_out)
            _save_rms_sub_intermediates(f"L{i:02d}_gpu", attn_residual)

            # CUTLASS Q/K/V projections.  attn.q_proj.weight is [N, K]; we pass
            # its transpose [K, N] to match cutlass_gemm_flex's convention.
            q_w = attn.q_proj.weight.t().contiguous()  # [H, num_heads*head_dim]
            k_w = attn.k_proj.weight.t().contiguous()
            v_w = attn.v_proj.weight.t().contiguous()
            q = cutlass_mm(rms_out, q_w, tmpdir)  # [seq, num_heads*head_dim]
            k = cutlass_mm(rms_out, k_w, tmpdir)  # [seq, num_kv_heads*head_dim]
            v = cutlass_mm(rms_out, v_w, tmpdir)
            save_bin(os.path.join(out_dir, f"L{i:02d}_q_proj_out.bin"), q)
            save_bin(os.path.join(out_dir, f"L{i:02d}_k_proj_out.bin"), k)
            save_bin(os.path.join(out_dir, f"L{i:02d}_v_proj_out.bin"), v)

            # Reshape for per-head ops.  transformers' Qwen3 uses [B, H, S, D]
            # internally (transpose 1,2), but we keep [seq, H, D] since the
            # emulator expects that layout and we're not batched.
            q_heads = q.view(seq_len, num_heads, head_dim)       # [seq, num_heads, D]
            k_heads = k.view(seq_len, num_kv_heads, head_dim)
            v_heads = v.view(seq_len, num_kv_heads, head_dim)

            # GPU QK-norm (per-head RMSNorm with head_dim=128)
            if has_qk_norm:
                q_normed = attn.q_norm(q_heads)
                k_normed = attn.k_norm(k_heads)
                save_bin(os.path.join(out_dir, f"L{i:02d}_q_normed.bin"), q_normed)
                save_bin(os.path.join(out_dir, f"L{i:02d}_k_normed.bin"), k_normed)
            else:
                q_normed = q_heads
                k_normed = k_heads

            # GPU RoPE — transformers' apply_rotary_pos_emb expects [B, H, S, D]
            q_for_rope = q_normed.unsqueeze(0).transpose(1, 2)
            k_for_rope = k_normed.unsqueeze(0).transpose(1, 2)
            q_roped, k_roped = apply_rotary_pos_emb(q_for_rope, k_for_rope, cos_gpu, sin_gpu)
            # Back to [seq, H, D]
            q_roped = q_roped.squeeze(0).transpose(0, 1).contiguous()
            k_roped = k_roped.squeeze(0).transpose(0, 1).contiguous()

            save_bin(os.path.join(out_dir, f"L{i:02d}_fa2_q.bin"), q_roped)
            save_bin(os.path.join(out_dir, f"L{i:02d}_fa2_k.bin"), k_roped)
            save_bin(os.path.join(out_dir, f"L{i:02d}_fa2_v.bin"), v_heads)

            # GPU FlashAttention-2 — expects [B, S, H, D] bfloat16 inputs
            from flash_attn import flash_attn_func
            fa2_out = flash_attn_func(
                q_roped.unsqueeze(0),
                k_roped.unsqueeze(0),
                v_heads.unsqueeze(0),
                causal=True,
            ).squeeze(0)  # [seq, num_heads, head_dim]
            fa2_flat = fa2_out.reshape(seq_len, num_heads * head_dim).contiguous()
            save_bin(os.path.join(out_dir, f"L{i:02d}_o_proj_input.bin"), fa2_flat)

            # CUTLASS O projection
            o_w = attn.o_proj.weight.t().contiguous()
            o = cutlass_mm(fa2_flat, o_w, tmpdir)
            save_bin(os.path.join(out_dir, f"L{i:02d}_o_proj_out.bin"), o)

            # Residual add (promote to FP32 then cast back to BF16 — matches
            # what PyTorch does for BF16 tensor addition).
            attn_block_out = (attn_residual.float() + o.float()).bfloat16()
            save_bin(os.path.join(out_dir, f"L{i:02d}_attn_block_out.bin"), attn_block_out)

            x = attn_block_out

            # ============ FFN block ============
            ffn_residual = x
            save_bin(os.path.join(out_dir, f"L{i:02d}_ffn_residual.bin"), ffn_residual)

            # GPU RMSNorm (post_attention_layernorm)
            ffn_rms_out = layer.post_attention_layernorm(x)
            save_bin(os.path.join(out_dir, f"L{i:02d}_ffn_rms_out.bin"), ffn_rms_out)
            _save_rms_sub_intermediates(f"L{i:02d}_ffn_gpu", ffn_residual)

            # CUTLASS gate/up
            gate_w = layer.mlp.gate_proj.weight.t().contiguous()
            up_w = layer.mlp.up_proj.weight.t().contiguous()
            gate = cutlass_mm(ffn_rms_out, gate_w, tmpdir)  # [seq, ffn_dim]
            up = cutlass_mm(ffn_rms_out, up_w, tmpdir)
            save_bin(os.path.join(out_dir, f"L{i:02d}_gate_out.bin"), gate)
            save_bin(os.path.join(out_dir, f"L{i:02d}_up_out.bin"), up)

            # GPU SiLU(gate) * up (BF16 cast after SiLU — matches HuggingFace MLP)
            silu_gate = F.silu(gate.float()).bfloat16()
            down_input = (silu_gate.float() * up.float()).bfloat16()
            save_bin(os.path.join(out_dir, f"L{i:02d}_down_input.bin"), down_input)

            # CUTLASS down
            down_w = layer.mlp.down_proj.weight.t().contiguous()
            down = cutlass_mm(down_input, down_w, tmpdir)
            save_bin(os.path.join(out_dir, f"L{i:02d}_down_out.bin"), down)

            # Residual add
            ffn_block_out = (ffn_residual.float() + down.float()).bfloat16()
            save_bin(os.path.join(out_dir, f"L{i:02d}_ffn_block_out.bin"), ffn_block_out)

            x = ffn_block_out

            layer_elapsed = time.time() - t_layer
            if i == 0 or (i + 1) % 6 == 0 or i == n_layers - 1:
                print(f"  Layer {i:2d}/{n_layers-1}: done ({layer_elapsed:.1f}s, "
                      f"cumulative CUTLASS calls {_cutlass_call_count[0]})")

    # ------------------------------------------------------------------------
    # Final norm + LM head
    # ------------------------------------------------------------------------
    print()
    print("FINAL NORM + LM HEAD")
    print("-" * 60)

    with torch.no_grad():
        final_norm_out = model.model.norm(x)  # [seq, H]
    save_bin(os.path.join(out_dir, "final_norm_out.bin"), final_norm_out)
    print(f"  final_norm_out.bin   [{seq_len}, {H}]")

    lm_head_w = model.lm_head.weight.t().contiguous()  # [H, vocab]
    logits = cutlass_mm(final_norm_out, lm_head_w, tmpdir)
    save_bin(os.path.join(out_dir, "logits.bin"), logits)
    vocab = lm_head_w.shape[1]
    print(f"  logits.bin           [{seq_len}, {vocab}]")

    # ------------------------------------------------------------------------
    # meta.json
    # ------------------------------------------------------------------------
    from tc_profiles import detect_gpu
    gpu_name = detect_gpu()
    meta = {
        "model": MODEL_NAME,
        "seq_len": seq_len,
        "n_layers": n_layers,
        "H": H,
        "ffn_dim": ffn_dim,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "gqa_groups": gqa_groups,
        "eps": eps,
        "has_qk_norm": has_qk_norm,
        "qk_eps": float(qk_eps) if qk_eps is not None else None,
        "rope_theta": rope_theta,
        "rope_type": rope_type,
        "hidden_act": cfg.hidden_act,
        "gpu": gpu_name,
        "capture_path": "cutlass",   # distinguishes from legacy cuBLAS captures
        "vocab_size": vocab,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  meta.json written")

    # Cleanup
    try:
        import shutil
        shutil.rmtree(tmpdir)
    except Exception:
        pass

    total_elapsed = time.time() - t_capture_start
    total_bytes = sum(
        os.path.getsize(os.path.join(out_dir, fn))
        for fn in os.listdir(out_dir)
    )
    print()
    print(f"Capture complete.")
    print(f"  total time:        {total_elapsed:.0f}s")
    print(f"  cutlass calls:     {_cutlass_call_count[0]}")
    print(f"  cutlass time:      {_cutlass_time_total[0]:.0f}s "
          f"(avg {_cutlass_time_total[0]/max(1,_cutlass_call_count[0]):.2f}s/call)")
    print(f"  total capture:     {total_bytes / 1e9:.2f} GB")
    print()
    print(f"Transfer {out_dir}/ and mufu_cache/ to CPU machine, then run:")
    print(f"  python3 emulate_forward.py {out_dir}")


if __name__ == "__main__":
    max_seq = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else DEFAULT_SEQ_LEN
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    capture(max_seq_len=max_seq, out_dir=out_dir)
