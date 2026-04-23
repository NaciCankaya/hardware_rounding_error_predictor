"""
catalog_lookup.py — CPU-only runtime lookup over a pre-built cuBLAS dispatch
catalog. Given a matmul shape, returns (recipe_fn, kwargs) that reproduce
cuBLAS's output bit-exactly for that shape.

Usage:
    catalog = load_catalog("cublas_catalog.json")
    out = catalog_matmul(A_np, B_np, catalog)

The catalog is a static JSON artifact produced by build_catalog.py. No GPU
access is needed at lookup time.
"""
import json

from cublas_recipes import (
    split_k_cutlass_bf16_out,
    split_k_sliced_kernel,
    split_k_workspace_outtype,
    single_walk,
)

RECIPES = {
    "split_k_cutlass_bf16_out":   split_k_cutlass_bf16_out,
    "split_k_sliced_kernel":      split_k_sliced_kernel,
    "split_k_workspace_outtype":  split_k_workspace_outtype,
    "single_walk":                 single_walk,
}


def load_catalog(path="cublas_catalog.json"):
    with open(path) as f:
        return json.load(f)


def lookup_recipe(catalog, M, N, K):
    """Find the catalog region matching this shape. Returns (fn, kwargs).
    Raises KeyError if no region covers this shape or the region is unverified."""
    for lane in catalog["lanes"]:
        if lane["N"] != N or lane["K"] != K:
            continue
        for r in lane["regions"]:
            if r.get("error"):
                continue
            if r["M_min"] <= M <= r["M_max"]:
                if r.get("verified") is False:
                    raise KeyError(
                        f"Shape (M={M}, N={N}, K={K}) matches region but is "
                        f"marked unverified (recipe={r.get('recipe')}, "
                        f"split_k={r.get('split_k')})"
                    )
                recipe_fn = RECIPES[r["recipe"]]
                kwargs = dict(r.get("recipe_kwargs", {}))
                if r.get("split_k") is not None and r["recipe"] != "single_walk":
                    kwargs["split_k"] = r["split_k"]
                return recipe_fn, kwargs
    raise KeyError(f"No catalog region covers shape (M={M}, N={N}, K={K})")


def catalog_matmul(A_np, B_np, catalog):
    """cuBLAS-matching matmul via catalog lookup. A is [M, K], B is [K, N]."""
    M, K = A_np.shape
    _, N = B_np.shape
    fn, kwargs = lookup_recipe(catalog, M, N, K)
    return fn(A_np, B_np, **kwargs)
