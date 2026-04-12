/*
 * cutlass_gemm_flex.cu — BF16 GEMM with FP32 accumulation
 *
 * Tile config: {128,128,64}/{64,64,64}/{16,8,16} on sm_80
 * Matches the emulator's tile configuration exactly.
 *
 * Usage:
 *   cutlass_gemm_flex M K N A.bin B.bin D.bin          # BF16 output
 *   cutlass_gemm_flex M K N A.bin B.bin D.bin fp32     # raw FP32 accumulator
 *
 * A: [M, K] row-major FP32 (will be cast to BF16 internally)
 * B: [K, N] row-major FP32 (will be cast to BF16 internally)
 * D: output, BF16-as-FP32 or raw FP32
 *
 * Compile:
 *   nvcc -O3 -arch=sm_80 -o cutlass_gemm_flex cutlass_gemm_flex.cu \
 *     -I/path/to/cutlass/include -I/path/to/cutlass/tools/util/include
 */

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_bf16.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/numeric_types.h"
#include "cutlass/util/host_tensor.h"

#define CHECK_CUDA(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define CHECK_CUTLASS(status) do { \
    cutlass::Status s = status; \
    if (s != cutlass::Status::kSuccess) { \
        fprintf(stderr, "CUTLASS error at %s:%d: %d\n", __FILE__, __LINE__, (int)s); \
        exit(1); \
    } \
} while(0)

// BF16 GEMM: A(row) x B(row) = D, FP32 accumulation
// Output type is templated: cutlass::bfloat16_t for BF16, float for FP32
template<typename OutputType>
void run_gemm(int M, int K, int N,
              const float* h_A, const float* h_B, float* h_D) {

    using ElementA = cutlass::bfloat16_t;
    using ElementB = cutlass::bfloat16_t;
    using ElementOutput = OutputType;
    using ElementAccumulator = float;

    using LayoutA = cutlass::layout::RowMajor;
    using LayoutB = cutlass::layout::RowMajor;
    using LayoutC = cutlass::layout::RowMajor;

    // Tile configuration: {128,128,64}/{64,64,64}/{16,8,16}
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128, 128, 64>,   // ThreadblockShape
        cutlass::gemm::GemmShape<64, 64, 64>,      // WarpShape
        cutlass::gemm::GemmShape<16, 8, 16>,        // InstructionShape
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        3  // stages
    >;

    // Convert FP32 inputs to BF16 on host
    std::vector<cutlass::bfloat16_t> h_A_bf16(M * K);
    std::vector<cutlass::bfloat16_t> h_B_bf16(K * N);
    for (int i = 0; i < M * K; i++) {
        h_A_bf16[i] = cutlass::bfloat16_t(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_bf16[i] = cutlass::bfloat16_t(h_B[i]);
    }

    // Allocate device memory
    cutlass::bfloat16_t *d_A, *d_B;
    ElementOutput *d_D;
    CHECK_CUDA(cudaMalloc(&d_A, M * K * sizeof(cutlass::bfloat16_t)));
    CHECK_CUDA(cudaMalloc(&d_B, K * N * sizeof(cutlass::bfloat16_t)));
    CHECK_CUDA(cudaMalloc(&d_D, M * N * sizeof(ElementOutput)));

    CHECK_CUDA(cudaMemcpy(d_A, h_A_bf16.data(), M * K * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_B, h_B_bf16.data(), K * N * sizeof(cutlass::bfloat16_t), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_D, 0, M * N * sizeof(ElementOutput)));

    // Configure GEMM: D = 1.0 * A * B + 0.0 * C
    typename Gemm::Arguments args(
        {M, N, K},
        {d_A, K},       // A
        {d_B, N},       // B
        {d_D, N},       // C (unused, but needed)
        {d_D, N},       // D
        {1.0f, 0.0f}    // alpha, beta
    );

    Gemm gemm_op;
    CHECK_CUTLASS(gemm_op.can_implement(args));

    size_t workspace_size = Gemm::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        CHECK_CUDA(cudaMalloc(&workspace, workspace_size));
    }

    CHECK_CUTLASS(gemm_op.initialize(args, workspace));
    CHECK_CUTLASS(gemm_op());
    CHECK_CUDA(cudaDeviceSynchronize());

    // Copy result back
    std::vector<ElementOutput> h_result(M * N);
    CHECK_CUDA(cudaMemcpy(h_result.data(), d_D, M * N * sizeof(ElementOutput), cudaMemcpyDeviceToHost));

    // Convert to FP32 for output
    for (int i = 0; i < M * N; i++) {
        h_D[i] = float(h_result[i]);
    }

    CHECK_CUDA(cudaFree(d_A));
    CHECK_CUDA(cudaFree(d_B));
    CHECK_CUDA(cudaFree(d_D));
    if (workspace) CHECK_CUDA(cudaFree(workspace));
}

int main(int argc, char** argv) {
    if (argc < 7) {
        fprintf(stderr, "Usage: %s M K N A.bin B.bin D.bin [fp32]\n", argv[0]);
        return 1;
    }

    int M = atoi(argv[1]);
    int K = atoi(argv[2]);
    int N = atoi(argv[3]);
    const char* a_path = argv[4];
    const char* b_path = argv[5];
    const char* d_path = argv[6];
    bool fp32_output = (argc > 7 && strcmp(argv[7], "fp32") == 0);

    // Read inputs
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_D(M * N);

    FILE* f;
    f = fopen(a_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", a_path); return 1; }
    fread(h_A.data(), sizeof(float), M * K, f);
    fclose(f);

    f = fopen(b_path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", b_path); return 1; }
    fread(h_B.data(), sizeof(float), K * N, f);
    fclose(f);

    // Run GEMM
    if (fp32_output) {
        // FP32 accumulator output — raw tensor core result, no BF16 epilogue
        run_gemm<float>(M, K, N, h_A.data(), h_B.data(), h_D.data());
    } else {
        // BF16 output — standard inference path
        run_gemm<cutlass::bfloat16_t>(M, K, N, h_A.data(), h_B.data(), h_D.data());
    }

    // Write output
    f = fopen(d_path, "wb");
    if (!f) { fprintf(stderr, "Cannot open %s for writing\n", d_path); return 1; }
    fwrite(h_D.data(), sizeof(float), M * N, f);
    fclose(f);

    return 0;
}
