/*
 * cublas_gemm_fp32.cu — cuBLASLt BF16-input / FP32-output GEMM.
 *
 * Runs one GEMM with cuBLASLt's default-dispatched algorithm (matches what
 * torch.matmul does) but keeps the output in FP32 instead of BF16, so the
 * raw accumulator bits are exposed to the caller. Used by
 * probe_l40_fp32.py to test whether the INPLACE_ATOMIC Split-K=3 path
 * observed in cublaslt_inspect is deterministic at the FP32 level, or
 * only at the BF16 epilogue level.
 *
 * Usage:
 *   cublas_gemm_fp32 M K N A.bin B.bin D.bin
 * A: [M, K] row-major FP32 (cast to BF16 on host)
 * B: [K, N] row-major FP32 (cast to BF16 on host)
 * D: [M, N] row-major FP32 output
 *
 * Compile:
 *   nvcc -o cublas_gemm_fp32 cublas_gemm_fp32.cu -lcublasLt -std=c++17 -O2
 */

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA %s at %s:%d\n", cudaGetErrorString(e), __FILE__, __LINE__); exit(1); } } while(0)

#define CHECK_BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s M K N A.bin B.bin D.bin\n", argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    const char* a_path = argv[4];
    const char* b_path = argv[5];
    const char* d_path = argv[6];

    // Load FP32 inputs, cast to BF16 on host
    std::vector<float> h_A((size_t)M * K);
    std::vector<float> h_B((size_t)K * N);
    FILE* f;
    f = fopen(a_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", a_path); return 1; }
    fread(h_A.data(), sizeof(float), (size_t)M * K, f);
    fclose(f);
    f = fopen(b_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", b_path); return 1; }
    fread(h_B.data(), sizeof(float), (size_t)K * N, f);
    fclose(f);

    std::vector<__nv_bfloat16> h_A_bf((size_t)M * K);
    std::vector<__nv_bfloat16> h_B_bf((size_t)K * N);
    for (size_t i = 0; i < (size_t)M * K; i++) h_A_bf[i] = __float2bfloat16(h_A[i]);
    for (size_t i = 0; i < (size_t)K * N; i++) h_B_bf[i] = __float2bfloat16(h_B[i]);

    // Device buffers
    __nv_bfloat16 *d_A = nullptr, *d_B = nullptr;
    float *d_D = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, (size_t)M * K * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_B, (size_t)K * N * sizeof(__nv_bfloat16)));
    CHECK_CUDA(cudaMalloc(&d_D, (size_t)M * N * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A_bf.data(), (size_t)M * K * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_bf.data(), (size_t)K * N * sizeof(__nv_bfloat16),
                          cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_D, 0, (size_t)M * N * sizeof(float)));

    // cuBLASLt setup
    cublasLtHandle_t lt;
    CHECK_BLAS(cublasLtCreate(&lt));

    cublasLtMatmulDesc_t desc;
    CHECK_BLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opN = CUBLAS_OP_N;
    CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatrixLayout_t A_l, B_l, D_l;
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&A_l, CUDA_R_16BF, M, K, K));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&B_l, CUDA_R_16BF, K, N, N));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&D_l, CUDA_R_32F, M, N, N));
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(A_l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(B_l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(D_l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));

    // Same workspace ceiling as cublaslt_inspect.cu (1 GiB), so the heuristic
    // returns the same rank-0 algo — on L40 at (8000,2560,9728) that's
    // algo 30, tile 128x64, Split-K=3, INPLACE_ATOMIC.
    cublasLtMatmulPreference_t pref;
    CHECK_BLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws_max = 1ULL << 30;
    CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_max, sizeof(ws_max)));

    cublasLtMatmulHeuristicResult_t heur[1];
    int returned = 0;
    CHECK_BLAS(cublasLtMatmulAlgoGetHeuristic(
        lt, desc, A_l, B_l, D_l, D_l, pref, 1, heur, &returned));
    if (returned == 0) {
        fprintf(stderr, "No algo returned by heuristic\n");
        return 1;
    }

    // Allocate exactly what the chosen algo wants
    void* workspace = nullptr;
    size_t ws_needed = heur[0].workspaceSize;
    if (ws_needed > 0) CHECK_CUDA(cudaMalloc(&workspace, ws_needed));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_BLAS(cublasLtMatmul(lt, desc,
        &alpha, d_A, A_l, d_B, B_l,
        &beta,  d_D, D_l, d_D, D_l,
        &heur[0].algo, workspace, ws_needed, /*stream=*/0));
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy FP32 output back and write
    std::vector<float> h_D((size_t)M * N);
    CHECK_CUDA(cudaMemcpy(h_D.data(), d_D, (size_t)M * N * sizeof(float),
                          cudaMemcpyDeviceToHost));
    f = fopen(d_path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", d_path); return 1; }
    fwrite(h_D.data(), sizeof(float), (size_t)M * N, f);
    fclose(f);

    // Cleanup
    if (workspace) cudaFree(workspace);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_D);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(A_l);
    cublasLtMatrixLayoutDestroy(B_l);
    cublasLtMatrixLayoutDestroy(D_l);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(lt);
    return 0;
}
