This folder holds notebooks that record what results were produced under which exact conditions — the GPU, the model, the seq length, and the resulting diff count. All notebooks run to completion end-to-end; the printed output is the record.

### Single-block validation (Qwen3-4B layer 20)

Per-GPU FFN chain runs at short and long seq lengths:

- `L40_256token_FFN.ipynb`, `L40_8000token_FFN.ipynb`
- `A100_8k_token_FFN_Qwen3.ipynb`
- `H100_256token_FNN.ipynb`, `H100_8000token_FNN.ipynb`

Attention chain (A100 only, FlashAttention 2.8.3):

- `A100_attn_chain_FA2_4000tokens.ipynb`

### Full-model validation (A100)

Full 36-layer forward pass of Qwen3-4B — every attention block, every FFN block, final RMSNorm, and LM head — compared against the GPU model output:

- `fullmodel_64_token.ipynb` (seq=32, 4.86M logit bits, 0 BF16 diffs)
- `fullmodel_250token.ipynb` (seq=250, 37.98M logit bits, 0 BF16 diffs)

### cuBLAS-target validation (dispatch catalog + recipe library)

Same FFN chain test, but with `EMULATOR_TARGET=cublas`: the emulator targets the library dispatch (`torch.matmul` → cuBLAS) instead of CUTLASS. Each notebook builds a dispatch catalog for seq lengths {100, 250, 1000, 4000} and runs Phase 3 against both targets:

- `CUTLASS_and_cuBLAS_A100.ipynb` — 0 diffs at every seq length, both targets
- `CUTLASS_and_cuBLAS_L40S.ipynb` — 0 diffs at every seq length, both targets
- `CUTLASS_and_cuBLAS_H100.ipynb` — 0 diffs everywhere except seq=100 down_proj, which dispatches to an `nvjet_*` kernel the recipe library does not model
- `CUTLASS_and_cuBLAS_H100_confidentialcomputing.ipynb` — identical results on a CC-mode H100 pod (determinism holds under confidential compute)
