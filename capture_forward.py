#!/usr/bin/env python3
"""
capture_forward.py — GPU-side full forward pass capture.

Run once on a GPU box (RunPod). Captures EVERY intermediate tensor for
EVERY layer, then writes them to a single compressed archive.  The
emulate_forward.py script reads this archive on a CPU-only machine.

Usage:
    python3 capture_forward.py [seq_len] [output_dir]

    seq_len     maximum token count (default 500, max 8000)
    output_dir  where to write capture data (default ./forward_capture)

Output layout in output_dir/
    meta.json                  model config + run metadata
    rope_cos_gpu.npy           [seq_len, head_dim]  GPU-computed cos (pre-BF16-cast)
    rope_sin_gpu.npy           [seq_len, head_dim]  GPU-computed sin (pre-BF16-cast)
    embed_out.bin              [seq_len, H]         token embedding output (FP32)
    L{i}_attn_*.bin            per-layer attn intermediates (see ATTN_KEYS)
    L{i}_gpu_rms_*.bin         per-layer RMSNorm FP32 sub-intermediates (attn)
    L{i}_ffn_*.bin             per-layer FFN intermediates (see FFN_KEYS)
    L{i}_ffn_gpu_rms_*.bin     per-layer RMSNorm FP32 sub-intermediates (ffn)
    final_norm_out.bin         [seq_len, H]
    logits.bin                 [seq_len, vocab_size]
"""

import json
import os
import sys
import glob

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen3-4B"
DEFAULT_SEQ_LEN = 500
DEFAULT_OUT_DIR = "forward_capture"

# Keys captured per layer for each block type.
# These match the keys used in block_emulators.py's captured dicts.
ATTN_KEYS = [
    "attn_residual",   # input to input_layernorm
    "rms_out",         # output of input_layernorm
    "q_proj_out",
    "k_proj_out",
    "v_proj_out",
    "q_normed",        # present only when has_qk_norm
    "k_normed",        # present only when has_qk_norm
    "fa2_q",           # post-RoPE Q entering FA2
    "fa2_k",
    "fa2_v",
    "o_proj_input",    # = FA2 output reshaped, entering O projection
    "o_proj_out",
    "attn_block_out",  # residual + o_proj_out
]
# RMSNorm FP32 sub-intermediates (attn input_layernorm)
ATTN_RMS_KEYS = ["rms_sumsq", "rms_variance", "rms_rsqrt", "rms_normed_f32", "rms_normed_bf16"]

FFN_KEYS = [
    "ffn_residual",    # = attn_block_out
    "rms_out",
    "gate_out",
    "up_out",
    "down_input",
    "down_out",
    "ffn_block_out",
]
FFN_RMS_KEYS = ["rms_sumsq", "rms_variance", "rms_rsqrt", "rms_normed_f32", "rms_normed_bf16"]


def save_bin(path, data):
    if isinstance(data, torch.Tensor):
        data = data.float().cpu().numpy()
    np.ascontiguousarray(data, dtype=np.float32).tofile(path)


def extract_pdf_text():
    pdf_paths = []
    for d in ["/workspace", "/mnt/user-data/uploads", "/mnt/project", "/mnt/data"]:
        if os.path.isdir(d):
            pdf_paths.extend(glob.glob(os.path.join(d, "**/*.pdf"), recursive=True))
    for pdf_path in sorted(set(pdf_paths), key=os.path.getsize, reverse=True):
        try:
            import subprocess
            result = subprocess.run(
                ["pdftotext", pdf_path, "-"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and len(result.stdout.strip()) > 500:
                print(f"  Using PDF (pdftotext): {pdf_path}")
                return result.stdout
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


def capture(max_seq_len=DEFAULT_SEQ_LEN, out_dir=DEFAULT_OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 80)
    print("FULL FORWARD PASS CAPTURE")
    print(f"  model:   {MODEL_NAME}")
    print(f"  max_seq: {max_seq_len}")
    print(f"  out_dir: {out_dir}")
    print("=" * 80)
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.bfloat16, device_map=DEVICE,
        attn_implementation="flash_attention_2"
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

    # Detect QK-norm from first layer
    attn0 = model.model.layers[0].self_attn
    has_qk_norm = getattr(attn0, "q_norm", None) is not None
    qk_eps = None
    if has_qk_norm:
        qk_norm = attn0.q_norm
        qk_eps = getattr(qk_norm, "variance_epsilon", getattr(qk_norm, "eps", eps))

    print(f"  {n_layers} layers, hidden={H}, ffn={ffn_dim}")
    print(f"  heads={num_heads}Q/{num_kv_heads}KV, head_dim={head_dim}, GQA={gqa_groups}")
    print(f"  has_qk_norm={has_qk_norm}, rope_theta={rope_theta}, rope_type={rope_type}")
    print()

    # Tokenize
    print("TOKENIZATION")
    print("-" * 60)
    text = extract_pdf_text()
    if text:
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
    else:
        print("  No PDF found, using repeated text")
        text = "The quick brown fox jumps over the lazy dog. " * 200
        tokens = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_seq_len)
    input_ids = tokens["input_ids"].to(DEVICE)
    seq_len = input_ids.shape[1]
    print(f"  Sequence length: {seq_len} tokens")
    print()

    # -----------------------------------------------------------------------
    # Register hooks on every layer
    # -----------------------------------------------------------------------
    print("REGISTERING HOOKS")
    print("-" * 60)

    layer_captures = [{} for _ in range(n_layers)]
    hooks = []

    for i, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        ln_attn = layer.input_layernorm
        ln_ffn = layer.post_attention_layernorm
        mlp = layer.mlp
        cap = layer_captures[i]

        # -- Attn block --
        def make_ln_pre_hook(cap):
            def h(module, args):
                x = args[0]
                if isinstance(x, tuple): x = x[0]
                cap["attn_residual"] = x.detach().squeeze(0).float().cpu()
            return h

        def make_ln_post_hook(cap):
            def h(module, args, output):
                cap["rms_out"] = output.detach().squeeze(0).float().cpu()
            return h

        def make_proj_hook(cap, key):
            def h(module, args, output):
                cap[key] = output.detach().squeeze(0).float().cpu()
            return h

        def make_o_pre_hook(cap):
            def h(module, args):
                x = args[0]
                if isinstance(x, tuple): x = x[0]
                cap["o_proj_input"] = x.detach().squeeze(0).float().cpu()
            return h

        def make_attn_out_hook(cap):
            def h(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                cap["attn_module_out"] = out.detach().squeeze(0).float().cpu()
            return h

        hooks.append(ln_attn.register_forward_pre_hook(make_ln_pre_hook(cap)))
        hooks.append(ln_attn.register_forward_hook(make_ln_post_hook(cap)))
        hooks.append(attn.q_proj.register_forward_hook(make_proj_hook(cap, "q_proj_out")))
        hooks.append(attn.k_proj.register_forward_hook(make_proj_hook(cap, "k_proj_out")))
        hooks.append(attn.v_proj.register_forward_hook(make_proj_hook(cap, "v_proj_out")))
        if has_qk_norm:
            hooks.append(attn.q_norm.register_forward_hook(make_proj_hook(cap, "q_normed")))
            hooks.append(attn.k_norm.register_forward_hook(make_proj_hook(cap, "k_normed")))
        hooks.append(attn.o_proj.register_forward_pre_hook(make_o_pre_hook(cap)))
        hooks.append(attn.o_proj.register_forward_hook(make_proj_hook(cap, "o_proj_out")))
        hooks.append(attn.register_forward_hook(make_attn_out_hook(cap)))

        # -- FFN block --
        def make_ffn_res_hook(cap):
            def h(module, args):
                x = args[0]
                if isinstance(x, tuple): x = x[0]
                cap["ffn_residual"] = x.detach().squeeze(0).float().cpu()
            return h

        def make_ffn_rms_hook(cap):
            def h(module, args, output):
                cap["ffn_rms_out"] = output.detach().squeeze(0).float().cpu()
            return h

        def make_down_pre_hook(cap):
            def h(module, args):
                x = args[0]
                if isinstance(x, tuple): x = x[0]
                cap["down_input"] = x.detach().squeeze(0).float().cpu()
            return h

        hooks.append(ln_ffn.register_forward_pre_hook(make_ffn_res_hook(cap)))
        hooks.append(ln_ffn.register_forward_hook(make_ffn_rms_hook(cap)))
        hooks.append(mlp.gate_proj.register_forward_hook(make_proj_hook(cap, "gate_out")))
        hooks.append(mlp.up_proj.register_forward_hook(make_proj_hook(cap, "up_out")))
        hooks.append(mlp.down_proj.register_forward_pre_hook(make_down_pre_hook(cap)))
        hooks.append(mlp.down_proj.register_forward_hook(make_proj_hook(cap, "down_out")))

    # Monkey-patch flash_attn to capture post-RoPE Q/K/V for every layer.
    # We use a counter to track which layer's attention is currently executing.
    # Since inference is sequential, a simple counter works.
    #
    # In flash_attn 2.8.x, FlashAttnFunc.forward calls the C++ binding
    # (_C.fwd) directly, bypassing the Python `_flash_attn_forward` wrapper.
    # So we patch flash_attn.flash_attn_func, which is the actual public
    # entry transformers calls into.  We ALSO patch the transformers-side
    # imported reference (modeling_flash_attention_utils.flash_attn_func)
    # in case transformers bound a local name at import time — whichever
    # is hit first increments the counter.
    _attn_call_idx = [0]
    _patched_fa = False

    def _make_capture_fa_func(orig_fn):
        def _capture(q, k, v, *args, **kwargs):
            idx = _attn_call_idx[0]
            if idx < n_layers:
                cap = layer_captures[idx]
                cap["fa2_q"] = q.detach().squeeze(0).float().cpu()
                cap["fa2_k"] = k.detach().squeeze(0).float().cpu()
                cap["fa2_v"] = v.detach().squeeze(0).float().cpu()
            _attn_call_idx[0] += 1
            return orig_fn(q, k, v, *args, **kwargs)
        return _capture

    _patch_restores = []
    try:
        import flash_attn
        _orig_fa_func = flash_attn.flash_attn_func
        flash_attn.flash_attn_func = _make_capture_fa_func(_orig_fa_func)
        _patch_restores.append(("flash_attn", "flash_attn_func", _orig_fa_func))
        _patched_fa = True
        print(f"  flash_attn.flash_attn_func monkey-patched for {n_layers} layers")
    except (ImportError, AttributeError) as e:
        print(f"  WARNING: could not patch flash_attn.flash_attn_func ({e})")

    # Also patch transformers' imported reference (defensive — if it did
    # `from flash_attn import flash_attn_func` at module load, the name in
    # that module points at the original function, not our replacement).
    try:
        import transformers.modeling_flash_attention_utils as _tfa
        if hasattr(_tfa, "flash_attn_func"):
            _orig_tfa_func = _tfa.flash_attn_func
            _tfa.flash_attn_func = _make_capture_fa_func(_orig_tfa_func)
            _patch_restores.append((_tfa.__name__, "flash_attn_func", _orig_tfa_func))
            print(f"  transformers.modeling_flash_attention_utils.flash_attn_func also patched")
    except (ImportError, AttributeError):
        pass

    if not _patched_fa:
        print("  WARNING: Could not monkey-patch flash_attn — fa2_q/k/v not captured")

    print(f"  {len(hooks)} hooks registered across {n_layers} layers")
    print()

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    print("RUNNING FORWARD PASS...")
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=True)
    print("done.")
    print()

    # Remove hooks and restore original flash_attn bindings
    for h in hooks:
        h.remove()
    for mod_name, attr_name, orig in _patch_restores:
        try:
            import importlib
            mod = importlib.import_module(mod_name)
            setattr(mod, attr_name, orig)
        except Exception:
            pass

    # Sanity: did the fa2 capture actually fire?
    if _patched_fa:
        print(f"  flash_attn patch fired {_attn_call_idx[0]} times "
              f"(expected ≥{n_layers})")

    # -----------------------------------------------------------------------
    # Compute and save RoPE cos/sin (GPU cosf, pre-BF16-cast)
    # -----------------------------------------------------------------------
    print("COMPUTING RoPE cos/sin (GPU)...")
    with torch.no_grad():
        position_ids = torch.arange(seq_len, device=DEVICE).unsqueeze(0)
        # Use the model's rotary_emb to get the exact GPU cos/sin values.
        # We pass a dummy v just to determine the dtype; the actual values don't matter.
        dummy_v = torch.zeros(1, seq_len, num_kv_heads, head_dim,
                              dtype=torch.bfloat16, device=DEVICE)
        cos_gpu, sin_gpu = model.model.rotary_emb(dummy_v, position_ids)
        # cos_gpu: [1, seq_len, head_dim] (duplicated [cos,cos] for rotate_half)
        cos_gpu = cos_gpu.squeeze(0).float().cpu().numpy()  # [seq_len, head_dim]
        sin_gpu = sin_gpu.squeeze(0).float().cpu().numpy()
    np.save(os.path.join(out_dir, "rope_cos_gpu.npy"), cos_gpu)
    np.save(os.path.join(out_dir, "rope_sin_gpu.npy"), sin_gpu)
    print(f"  rope_cos_gpu.npy  {cos_gpu.shape}")
    print(f"  rope_sin_gpu.npy  {sin_gpu.shape}")
    print()

    # -----------------------------------------------------------------------
    # Token embeddings
    # -----------------------------------------------------------------------
    print("SAVING EMBEDDING OUTPUT...")
    with torch.no_grad():
        embed_out = model.model.embed_tokens(input_ids).squeeze(0).float().cpu().numpy()
    save_bin(os.path.join(out_dir, "embed_out.bin"), embed_out)
    print(f"  embed_out.bin  {embed_out.shape}")
    print()

    # -----------------------------------------------------------------------
    # Final norm + logits
    # -----------------------------------------------------------------------
    print("SAVING FINAL OUTPUTS...")
    # Qwen3Model.forward applies self.norm(hidden_states) BEFORE appending to
    # all_hidden_states, so outputs.hidden_states[-1] is already post-norm.
    # Re-applying the norm here would double-norm and produce a ground truth
    # that nothing can match.  Save it as-is.
    final_norm_out = outputs.hidden_states[-1].squeeze(0).float().cpu().numpy()
    logits = outputs.logits.squeeze(0).float().cpu().numpy()
    save_bin(os.path.join(out_dir, "final_norm_out.bin"), final_norm_out)
    save_bin(os.path.join(out_dir, "logits.bin"), logits)
    print(f"  final_norm_out.bin   {final_norm_out.shape}")
    print(f"  logits.bin           {logits.shape}")
    print()

    # -----------------------------------------------------------------------
    # Per-layer RMSNorm FP32 sub-intermediates and save all tensors
    # -----------------------------------------------------------------------
    print("SAVING PER-LAYER CAPTURES")
    print("-" * 60)

    fa2_missing = 0
    for i in range(n_layers):
        cap = layer_captures[i]

        # Compute attn RMSNorm FP32 sub-intermediates from captured residual
        if "attn_residual" in cap:
            res = cap["attn_residual"].numpy()  # [seq_len, H]
            x_f32 = res.astype(np.float32)
            gpu_sumsq = torch.tensor(res).pow(2).sum(-1, keepdim=True).numpy()
            gpu_variance = torch.tensor(res).pow(2).mean(-1, keepdim=True).numpy()
            gpu_rsqrt = torch.rsqrt(torch.tensor(gpu_variance) + eps).numpy()
            gpu_normed_f32 = (x_f32 * gpu_rsqrt).astype(np.float32)
            gpu_normed_bf16 = torch.tensor(gpu_normed_f32).bfloat16().float().numpy()
            cap["gpu_rms_sumsq"] = torch.tensor(gpu_sumsq)
            cap["gpu_rms_variance"] = torch.tensor(gpu_variance)
            cap["gpu_rms_rsqrt"] = torch.tensor(gpu_rsqrt)
            cap["gpu_rms_normed_f32"] = torch.tensor(gpu_normed_f32)
            cap["gpu_rms_normed_bf16"] = torch.tensor(gpu_normed_bf16)

        # Compute FFN RMSNorm FP32 sub-intermediates from captured ffn_residual
        if "ffn_residual" in cap:
            res = cap["ffn_residual"].numpy()
            x_f32 = res.astype(np.float32)
            gpu_sumsq = torch.tensor(res).pow(2).sum(-1, keepdim=True).numpy()
            gpu_variance = torch.tensor(res).pow(2).mean(-1, keepdim=True).numpy()
            gpu_rsqrt = torch.rsqrt(torch.tensor(gpu_variance) + eps).numpy()
            gpu_normed_f32 = (x_f32 * gpu_rsqrt).astype(np.float32)
            gpu_normed_bf16 = torch.tensor(gpu_normed_f32).bfloat16().float().numpy()
            cap["ffn_gpu_rms_sumsq"] = torch.tensor(gpu_sumsq)
            cap["ffn_gpu_rms_variance"] = torch.tensor(gpu_variance)
            cap["ffn_gpu_rms_rsqrt"] = torch.tensor(gpu_rsqrt)
            cap["ffn_gpu_rms_normed_f32"] = torch.tensor(gpu_normed_f32)
            cap["ffn_gpu_rms_normed_bf16"] = torch.tensor(gpu_normed_bf16)

        # Compute attn_block_out = residual + o_proj_out
        if "attn_residual" in cap and "o_proj_out" in cap:
            res_bf16 = cap["attn_residual"].bfloat16().float()
            o_bf16 = cap["o_proj_out"].bfloat16().float()
            cap["attn_block_out"] = (res_bf16 + o_bf16).bfloat16().float()

        # Compute ffn_block_out = ffn_residual + down_out
        if "ffn_residual" in cap and "down_out" in cap:
            res_bf16 = cap["ffn_residual"].bfloat16().float()
            d_bf16 = cap["down_out"].bfloat16().float()
            cap["ffn_block_out"] = (res_bf16 + d_bf16).bfloat16().float()

        # Save everything for this layer
        for key, val in cap.items():
            if isinstance(val, torch.Tensor):
                arr = val.float().cpu().numpy()
            else:
                arr = np.asarray(val, dtype=np.float32)
            save_bin(os.path.join(out_dir, f"L{i:02d}_{key}.bin"), arr)

        fa2_present = all(k in cap for k in ["fa2_q", "fa2_k", "fa2_v"])
        if not fa2_present:
            fa2_missing += 1

        if i == 0 or (i + 1) % 8 == 0 or i == n_layers - 1:
            keys_present = len(cap)
            print(f"  Layer {i:2d}: {keys_present} tensors saved"
                  + ("" if fa2_present else "  [!] fa2 Q/K/V missing"))

    if fa2_missing:
        print(f"\n  WARNING: {fa2_missing}/{n_layers} layers missing fa2 Q/K/V captures.")
        print("  Emulator will use its own RoPE output as FA2 input (may produce diffs).")
    print()

    # -----------------------------------------------------------------------
    # meta.json
    # -----------------------------------------------------------------------
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
        "hidden_act": getattr(cfg, "hidden_act", "silu"),
        "fa2_captured": fa2_missing == 0,
        "gpu": gpu_name,  # which tensor-core profile the emulator must use
    }
    meta_path = os.path.join(out_dir, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"meta.json written.")
    print()

    # -----------------------------------------------------------------------
    # Size report
    # -----------------------------------------------------------------------
    total_bytes = sum(
        os.path.getsize(os.path.join(out_dir, fn))
        for fn in os.listdir(out_dir)
    )
    print(f"Total capture size: {total_bytes / 1e9:.2f} GB")
    print()
    print(f"Capture complete. Transfer {out_dir}/ to CPU machine, then run:")
    print(f"  python3 emulate_forward.py {out_dir}")


if __name__ == "__main__":
    max_seq = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else DEFAULT_SEQ_LEN
    out_dir = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT_DIR
    capture(max_seq_len=max_seq, out_dir=out_dir)
