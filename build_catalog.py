#!/usr/bin/env python3
"""
build_catalog.py — query cuBLAS dispatch for a grid of shapes and emit a
JSON catalog of (shape -> recipe + params). Uses the cublaslt_inspect binary
(configured for PyTorch's tn layout convention) so the reported split_k
matches what torch.matmul actually launches.

One optional verification run per shape confirms the recipe matches cuBLAS
bit-exactly; any mismatch is flagged but doesn't gate the catalog entry.

Usage:
    python3 build_catalog.py                  # FFN shapes, default grid
    python3 build_catalog.py --lanes all      # FFN + attention + LM head
    python3 build_catalog.py --no-verify      # skip verification (faster)
    python3 build_catalog.py --max-M 8192     # cap the M sweep range
"""
import torch
import numpy as np
import json
import argparse
import subprocess
import os
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

INSPECTOR_BIN = "./cublaslt_inspect"


# ----------------------------------------------------------------------
# Inspector: query cuBLAS's dispatch choice for a shape
# ----------------------------------------------------------------------
def query_inspector(M, N, K):
    """Run cublaslt_inspect and parse the rank-0 (top-1) dispatch row."""
    result = subprocess.run(
        [INSPECTOR_BIN, str(M), str(N), str(K)],
        capture_output=True, text=True, timeout=30, check=True,
    )
    for line in result.stdout.splitlines():
        if line.startswith("0\t"):
            parts = line.split("\t")
            return {
                "algo_id":   int(parts[1]),
                "tile":      parts[2],
                "stages":    int(parts[3]),
                "split_k":   int(parts[4]),
                "reduction": parts[5],
                "swizzle":   int(parts[6]),
                "custom":    int(parts[7]),
            }
    raise RuntimeError(f"Could not parse inspector output for shape ({M},{N},{K})")


# ----------------------------------------------------------------------
# Kernel name via PyTorch profiler (for naming and recipe identification)
# ----------------------------------------------------------------------
def get_kernel_name(M, N, K):
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


# ----------------------------------------------------------------------
# Map dispatch info → recipe
# ----------------------------------------------------------------------
def identify_recipe(dispatch, kernel_name):
    """Given dispatch metadata + kernel symbol, pick the recipe family."""
    sk = dispatch["split_k"]
    reduction = dispatch["reduction"]

    # No Split-K, no reduction → single-walk matches
    if sk == 1 and reduction.startswith("NONE"):
        return "single_walk", {}, 1

    # sliced1x2 in the kernel name → sliced recipe regardless of Split-K factor
    if kernel_name and "sliced1x2" in kernel_name:
        return "split_k_sliced_kernel", {"tb_K": 64, "warp_K": 32}, sk

    # Stock CUTLASS 128x64 template empirically dispatches with kPartitionsK>1
    if kernel_name and "_128x64_32x6_" in kernel_name:
        return "split_k_sliced_kernel", {"tb_K": 64, "warp_K": 32}, sk

    # Stock CUTLASS 64x64 template or other non-sliced Split-K paths
    return "split_k_cutlass_bf16_out", {"tb_K": 64}, sk


# ----------------------------------------------------------------------
# Verification: does the recipe produce bit-exact cuBLAS output?
# ----------------------------------------------------------------------
def _bits_match(emu_out_np, C_cublas_bf16):
    emu_bf16 = torch.from_numpy(emu_out_np).bfloat16()
    return int((emu_bf16 != C_cublas_bf16).sum().item()) == 0


def verify_recipe(M, N, K, recipe_name, kwargs, split_k):
    torch.manual_seed(0)
    A = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
    B = torch.randn(K, N, dtype=torch.bfloat16, device="cuda")
    C_cublas = torch.matmul(A, B).cpu()
    fn = RECIPE_FNS[recipe_name]
    A_np, B_np = A.float().cpu().numpy(), B.float().cpu().numpy()
    if recipe_name == "single_walk":
        out = fn(A_np, B_np)
    else:
        out = fn(A_np, B_np, split_k=split_k, **kwargs)
    return _bits_match(out, C_cublas)


# ----------------------------------------------------------------------
# Per-lane sweep
# ----------------------------------------------------------------------
def sweep_lane(N, K, M_grid, label, verify):
    results = []
    for i, M in enumerate(M_grid):
        print(f"  [{i+1:>2d}/{len(M_grid):>2d}] M={M:>5d} N={N:>5d} K={K:>5d} ",
              end="", flush=True)

        try:
            dispatch = query_inspector(M, N, K)
        except Exception as e:
            print(f"  inspector failed: {e}")
            results.append({"M": M, "N": N, "K": K, "error": f"inspector: {e}"})
            continue

        kname = get_kernel_name(M, N, K)
        recipe_name, kwargs, split_k = identify_recipe(dispatch, kname)

        entry = {
            "M": M, "N": N, "K": K,
            "kernel": kname,
            "dispatch": dispatch,
            "recipe": recipe_name,
            "recipe_kwargs": kwargs,
            "split_k": split_k,
        }

        if verify:
            try:
                ok = verify_recipe(M, N, K, recipe_name, kwargs, split_k)
            except Exception as e:
                ok = False
                entry["verify_error"] = str(e)
            entry["verified"] = ok
            status = "OK" if ok else "*** MISMATCH ***"
        else:
            entry["verified"] = None
            status = "skipped"

        short_kname = (kname or "<none>")
        if "<" in short_kname:
            short_kname = short_kname.split("<", 1)[1].split(">", 1)[0]
        short_kname = short_kname[:55]
        print(f"  recipe={recipe_name:<28} split_k={split_k}  verify={status:<20}  "
              f"kernel={short_kname}")
        results.append(entry)
    return results


# ----------------------------------------------------------------------
# Run-length-encode consecutive matching shapes into regions
# ----------------------------------------------------------------------
def compact_regions(results):
    regions = []
    for r in results:
        if "error" in r:
            regions.append(r)
            continue
        sig = (r.get("kernel"), r.get("recipe"), r.get("split_k"))
        if (regions and regions[-1].get("_signature") == sig
                and regions[-1].get("verified") == r.get("verified")):
            regions[-1]["M_max"] = r["M"]
        else:
            regions.append({
                "_signature": sig,
                "N": r["N"], "K": r["K"],
                "M_min": r["M"], "M_max": r["M"],
                "kernel": r.get("kernel"),
                "dispatch": r.get("dispatch"),
                "recipe": r.get("recipe"),
                "recipe_kwargs": r.get("recipe_kwargs", {}),
                "split_k": r.get("split_k"),
                "verified": r.get("verified"),
            })
    for r in regions:
        r.pop("_signature", None)
    return regions


def build_m_grid(max_M):
    grid = set([
        1, 16, 32, 64, 96, 128, 192, 256,
        320, 384, 512, 768, 1024, 1536,
        2048, 3072, 4096, 5120, 6144, 8000, 8192,
    ])
    return sorted(m for m in grid if m <= max_M)


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
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip per-shape bit-exactness verification (faster).")
    args = parser.parse_args()

    if not os.path.exists(INSPECTOR_BIN):
        parser.error(f"{INSPECTOR_BIN} not found. Compile first:\n"
                     "  nvcc -o cublaslt_inspect cublaslt_inspect.cu "
                     "-lcublasLt -std=c++17 -O2")

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
            "verified": not args.no_verify,
        },
        "lanes": [],
    }

    for N, K, label in QWEN3_4B_SHAPES[args.lanes]:
        print(f"\n=== {label} (N={N}, K={K}) ===")
        results = sweep_lane(N, K, M_grid, label, verify=not args.no_verify)
        regions = compact_regions(results)
        catalog["lanes"].append({
            "label": label, "N": N, "K": K, "regions": regions
        })
        print(f"  {len(regions)} region(s) in this lane")

    with open(args.output, "w") as f:
        json.dump(catalog, f, indent=2)

    n_regions = sum(len(l["regions"]) for l in catalog["lanes"])
    n_verified = sum(1 for l in catalog["lanes"] for r in l["regions"]
                     if r.get("verified"))
    n_mismatch = sum(1 for l in catalog["lanes"] for r in l["regions"]
                     if r.get("verified") is False)
    print(f"\nWrote {args.output}")
    print(f"  {n_regions} regions total, {n_verified} verified bit-exact, "
          f"{n_mismatch} mismatches")


if __name__ == "__main__":
    main()
