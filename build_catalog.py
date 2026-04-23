#!/usr/bin/env python3
"""
build_catalog.py — sweep tensor shapes, identify the bit-exact recipe+params
for each cuBLAS dispatch, emit a JSON catalog.

Run once per (SKU, cuBLAS version, dtype). The resulting catalog is a static
artifact the emulator consumes at verification time — no GPU required after
the catalog is built.

Usage:
    python3 build_catalog.py                      # FFN shapes only (default)
    python3 build_catalog.py --lanes all          # FFN + attention + LM head
    python3 build_catalog.py --output my.json     # custom output path
    python3 build_catalog.py --max-M 8192         # cap the M sweep range

Runtime: ~30-60 minutes for FFN lanes on an L40.
"""
import torch
import numpy as np
import json
import argparse
from datetime import datetime, timezone

from cublas_recipes import (
    split_k_cutlass_bf16_out,
    split_k_sliced_kernel,
    single_walk,
)

RECIPE_FNS = {
    "split_k_cutlass_bf16_out": split_k_cutlass_bf16_out,
    "split_k_sliced_kernel":    split_k_sliced_kernel,
    "single_walk":               single_walk,
}


# ----------------------------------------------------------------------
# Kernel identification via PyTorch profiler
# ----------------------------------------------------------------------
def get_kernel_name(M, N, K):
    """Run one matmul under the profiler, return the dispatched kernel symbol."""
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
        _ = A @ B
        torch.cuda.synchronize()
    for evt in p.key_averages():
        n = evt.key
        if any(tag in n for tag in ["gemm", "Kernel2", "s16816", "s1688",
                                      "ampere_", "cutlass_", "turing_"]):
            return n
    return None


def identify_recipe_family(kname):
    """Map kernel symbol → (recipe_name, baseline_kwargs)."""
    if kname is None:
        return None, None
    # sliced1x2 family -> sliced recipe (kPartitionsK=2)
    if "sliced1x2" in kname:
        return "split_k_sliced_kernel", {"tb_K": 64, "warp_K": 32}
    # stock CUTLASS 64x64 template -> no sliced-K
    if "_64x64_32x6_" in kname:
        return "split_k_cutlass_bf16_out", {"tb_K": 64}
    # stock CUTLASS 128x64 template -> empirically dispatches with sliced-K
    if "_128x64_32x6_" in kname:
        return "split_k_sliced_kernel", {"tb_K": 64, "warp_K": 32}
    # Plain Split-K without sliced (stages_XxX variants or no-stages plain names)
    if "sliced" not in kname and any(tag in kname for tag in ["ampere_", "cutlass_"]):
        return "single_walk", {}
    return None, None


# ----------------------------------------------------------------------
# Recipe validation: find split_k that makes the recipe bit-exact
# ----------------------------------------------------------------------
def _bits_match(emu_out_np, C_cublas_bf16):
    """Compare emulator FP32-container output (BF16 precision) to cuBLAS BF16 tensor."""
    emu_bf16 = torch.from_numpy(emu_out_np).bfloat16()
    return int((emu_bf16 != C_cublas_bf16).sum().item()) == 0


def probe_split_k(A_np, B_np, C_cublas_bf16, recipe_name, kwargs,
                    candidates=(1, 2, 3, 4, 6, 8)):
    fn = RECIPE_FNS[recipe_name]
    if recipe_name == "single_walk":
        if _bits_match(fn(A_np, B_np), C_cublas_bf16):
            return 1
        return None
    for sk in candidates:
        try:
            if _bits_match(fn(A_np, B_np, split_k=sk, **kwargs), C_cublas_bf16):
                return sk
        except Exception:
            continue
    return None


# ----------------------------------------------------------------------
# Per-lane sweep
# ----------------------------------------------------------------------
def sweep_lane(N, K, M_grid, label):
    results = []
    for i, M in enumerate(M_grid):
        print(f"  [{i+1:>2d}/{len(M_grid):>2d}] M={M:>5d} N={N:>5d} K={K:>5d} ", end="", flush=True)

        torch.manual_seed(0)
        A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
        B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
        C_cublas_bf16 = torch.matmul(A, B).cpu()

        kname = get_kernel_name(M, N, K)
        recipe_name, kwargs = identify_recipe_family(kname)
        if recipe_name is None:
            short = (kname or "<none>")[:70]
            print(f"  unknown family: {short}")
            results.append({"M": M, "N": N, "K": K, "kernel": kname,
                            "error": "unknown family"})
            continue

        A_np = A.float().cpu().numpy()
        B_np = B.float().cpu().numpy()
        sk = probe_split_k(A_np, B_np, C_cublas_bf16, recipe_name, kwargs)
        verified = sk is not None
        print(f"  recipe={recipe_name:<28} split_k={sk}  {'OK' if verified else '*** NO MATCH ***'}")

        results.append({
            "M": M, "N": N, "K": K,
            "kernel": kname,
            "recipe": recipe_name,
            "recipe_kwargs": kwargs,
            "split_k": sk,
            "verified": verified,
        })
    return results


# ----------------------------------------------------------------------
# Run-length-encode consecutive matching shapes into regions
# ----------------------------------------------------------------------
def compact_regions(results):
    regions = []
    for r in results:
        sig = (r.get("kernel"), r.get("recipe"), r.get("split_k"))
        if regions and regions[-1]["_signature"] == sig:
            regions[-1]["M_max"] = r["M"]
        else:
            regions.append({
                "_signature": sig,
                "N": r["N"], "K": r["K"],
                "M_min": r["M"], "M_max": r["M"],
                "kernel": r.get("kernel"),
                "recipe": r.get("recipe"),
                "recipe_kwargs": r.get("recipe_kwargs", {}),
                "split_k": r.get("split_k"),
                "verified": r.get("verified", False),
            })
    for r in regions:
        del r["_signature"]
    return regions


# ----------------------------------------------------------------------
# Shape grid
# ----------------------------------------------------------------------
def build_m_grid(max_M):
    """Small-but-informative grid. Fine near small values where dispatch flips
    often; moderate in the mid-range; the two shapes the paper measures (256
    and 8000) are explicitly included."""
    grid = set([
        1, 16, 32, 64, 96, 128, 192, 256,
        320, 384, 512, 768, 1024, 1536,
        2048, 3072, 4096, 5120, 6144, 8000, 8192,
    ])
    return sorted(m for m in grid if m <= max_M)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
QWEN3_4B_SHAPES = {
    "ffn": [
        (2560, 9728, "down_proj"),
        (9728, 2560, "gate_up_proj"),
    ],
    "all": [
        (2560, 9728, "down_proj"),
        (9728, 2560, "gate_up_proj"),
        (4096, 2560, "q_proj"),
        (1024, 2560, "kv_proj"),
        (2560, 4096, "o_proj"),
    ],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="cublas_catalog.json")
    parser.add_argument("--max-M", type=int, default=8192)
    parser.add_argument("--lanes", default="ffn", choices=["ffn", "all"])
    args = parser.parse_args()

    M_grid = build_m_grid(max_M=args.max_M)
    print(f"M grid ({len(M_grid)} values): {M_grid}")

    device_name = torch.cuda.get_device_name(0)
    catalog = {
        "meta": {
            "sku": device_name,
            "dtype_in": "bf16",
            "dtype_out": "bf16",
            "dtype_compute": "fp32",
            "generated": datetime.now(timezone.utc).isoformat(),
            "M_grid": M_grid,
        },
        "lanes": [],
    }

    for N, K, label in QWEN3_4B_SHAPES[args.lanes]:
        print(f"\n=== {label} (N={N}, K={K}) ===")
        results = sweep_lane(N, K, M_grid, label)
        regions = compact_regions(results)
        catalog["lanes"].append({
            "label": label, "N": N, "K": K, "regions": regions
        })
        print(f"  {len(regions)} region(s) in this lane")

    with open(args.output, "w") as f:
        json.dump(catalog, f, indent=2)
    n_regions = sum(len(l["regions"]) for l in catalog["lanes"])
    n_verified = sum(1 for l in catalog["lanes"] for r in l["regions"] if r["verified"])
    print(f"\nWrote {args.output}")
    print(f"  {n_regions} regions total, {n_verified} verified bit-exact")


if __name__ == "__main__":
    main()
