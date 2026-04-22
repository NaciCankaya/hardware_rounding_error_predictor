/*
 * cublaslt_inspect.cu — gray-box cuBLASLt dispatch inspector.
 *
 * Queries cublasLtMatmulAlgoGetHeuristic for a given (M, N, K) BF16 GEMM
 * and prints the dispatched algorithm configuration. No arithmetic probing:
 * this only observes what the (closed) heuristic function returns for a
 * given shape on the current device.
 *
 * Dtypes are fixed to BF16 inputs, BF16 output, FP32 compute, row-major,
 * no transpose — matching the inference path for Qwen3-style FFN projections.
 *
 * Usage:
 *   cublaslt_inspect M N K [max_algos]
 *
 * Example (Qwen3-4B down_proj at seq=8000):
 *   cublaslt_inspect 8000 2560 9728
 *
 * Output is TSV on stdout. The top row is the default dispatch. Compare
 * across SKUs (L40 vs A100 vs H100) by running the same command on each.
 *
 * For the mangled kernel name of the dispatched algo (useful for grepping
 * against CUTLASS templates or cuobjdump of libcublasLt.so), wrap a
 * companion runner that actually executes the top-1 algo with:
 *   ncu --print-summary ./your_runner M N K
 *
 * Compile:
 *   nvcc -o cublaslt_inspect cublaslt_inspect.cu -lcublasLt -std=c++17 -O2
 */

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cublasLt.h>

#define CHECK_CUDA(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(e)); exit(1); } } while(0)

#define CHECK_BLAS(x) do { cublasStatus_t s = (x); if (s != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %d at %s:%d\n", (int)s, __FILE__, __LINE__); exit(1); } } while(0)

static const char* tile_name(int id) {
    switch (id) {
        case CUBLASLT_MATMUL_TILE_UNDEFINED: return "UNDEF";
        case CUBLASLT_MATMUL_TILE_8x8:       return "8x8";
        case CUBLASLT_MATMUL_TILE_8x16:      return "8x16";
        case CUBLASLT_MATMUL_TILE_16x8:      return "16x8";
        case CUBLASLT_MATMUL_TILE_8x32:      return "8x32";
        case CUBLASLT_MATMUL_TILE_16x16:     return "16x16";
        case CUBLASLT_MATMUL_TILE_32x8:      return "32x8";
        case CUBLASLT_MATMUL_TILE_8x64:      return "8x64";
        case CUBLASLT_MATMUL_TILE_16x32:     return "16x32";
        case CUBLASLT_MATMUL_TILE_32x16:     return "32x16";
        case CUBLASLT_MATMUL_TILE_64x8:      return "64x8";
        case CUBLASLT_MATMUL_TILE_32x32:     return "32x32";
        case CUBLASLT_MATMUL_TILE_32x64:     return "32x64";
        case CUBLASLT_MATMUL_TILE_64x32:     return "64x32";
        case CUBLASLT_MATMUL_TILE_32x128:    return "32x128";
        case CUBLASLT_MATMUL_TILE_64x64:     return "64x64";
        case CUBLASLT_MATMUL_TILE_128x32:    return "128x32";
        case CUBLASLT_MATMUL_TILE_64x128:    return "64x128";
        case CUBLASLT_MATMUL_TILE_128x64:    return "128x64";
        case CUBLASLT_MATMUL_TILE_64x256:    return "64x256";
        case CUBLASLT_MATMUL_TILE_128x128:   return "128x128";
        case CUBLASLT_MATMUL_TILE_256x64:    return "256x64";
        case CUBLASLT_MATMUL_TILE_64x512:    return "64x512";
        case CUBLASLT_MATMUL_TILE_128x256:   return "128x256";
        case CUBLASLT_MATMUL_TILE_256x128:   return "256x128";
        case CUBLASLT_MATMUL_TILE_512x64:    return "512x64";
        default:                             return "?";
    }
}

static const char* reduction_name(int s) {
    switch (s) {
        case CUBLASLT_REDUCTION_SCHEME_NONE:         return "NONE";
        case CUBLASLT_REDUCTION_SCHEME_INPLACE:      return "INPLACE_ATOMIC";
        case CUBLASLT_REDUCTION_SCHEME_COMPUTE_TYPE: return "WORKSPACE_FP32";
        case CUBLASLT_REDUCTION_SCHEME_OUTPUT_TYPE:  return "WORKSPACE_OUTTYPE";
        default:                                     return "?";
    }
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr,
            "Usage: %s M N K [max_algos=32]\n"
            "  BF16 inputs, BF16 output, FP32 compute, row-major, no transpose.\n",
            argv[0]);
        return 1;
    }
    int M = atoi(argv[1]);
    int N = atoi(argv[2]);
    int K = atoi(argv[3]);
    int max_algos = (argc > 4) ? atoi(argv[4]) : 32;

    int dev;
    CHECK_CUDA(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, dev));

    cublasLtHandle_t lt;
    CHECK_BLAS(cublasLtCreate(&lt));
    size_t cublas_ver = cublasLtGetVersion();

    printf("# device=%s sm_%d%d driver_cublaslt=%zu\n",
           prop.name, prop.major, prop.minor, cublas_ver);
    printf("# problem M=%d N=%d K=%d dtypes=BF16/BF16/BF16 compute=FP32 order=row-major\n",
           M, N, K);

    cublasLtMatmulDesc_t desc;
    CHECK_BLAS(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t opN = CUBLAS_OP_N;
    CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CHECK_BLAS(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    // Row-major: leading dimension is the row stride (= number of columns).
    cublasLtMatrixLayout_t A_l, B_l, C_l;
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&A_l, CUDA_R_16BF, M, K, K));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&B_l, CUDA_R_16BF, K, N, N));
    CHECK_BLAS(cublasLtMatrixLayoutCreate(&C_l, CUDA_R_16BF, M, N, N));
    cublasLtOrder_t row = CUBLASLT_ORDER_ROW;
    CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(A_l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(B_l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));
    CHECK_BLAS(cublasLtMatrixLayoutSetAttribute(C_l, CUBLASLT_MATRIX_LAYOUT_ORDER, &row, sizeof(row)));

    // PyTorch's default workspace is 32 MiB for BF16 on Ampere/Ada,
    // 1 GiB on Hopper. Use the larger bound so we see every algo either
    // product family could pick.
    cublasLtMatmulPreference_t pref;
    CHECK_BLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t ws = 1ULL << 30;
    CHECK_BLAS(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws)));

    cublasLtMatmulHeuristicResult_t* results =
        new cublasLtMatmulHeuristicResult_t[max_algos];
    int returned = 0;
    CHECK_BLAS(cublasLtMatmulAlgoGetHeuristic(
        lt, desc, A_l, B_l, C_l, C_l, pref,
        max_algos, results, &returned));

    printf("# returned=%d  (rank 0 is the default dispatch)\n", returned);
    printf("rank\talgo_id\ttile\tstages\tsplit_k\treduction\tswizzle\tcustom\tinner\tcluster\twaves\tworkspace_bytes\n");

    for (int i = 0; i < returned; i++) {
        cublasLtMatmulAlgo_t& algo = results[i].algo;
        int algo_id = -1, tile = -1, stages = -1, split_k = -1,
            reduction = -1, swizzle = -1, custom = -1,
            inner_shape = -1, cluster_shape = -1;

        // Queries are best-effort; unsupported attributes on a given
        // algo just leave the sentinel -1 in place.
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_ID,
            &algo_id, sizeof(algo_id), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
            &tile, sizeof(tile), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
            &stages, sizeof(stages), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM,
            &split_k, sizeof(split_k), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME,
            &reduction, sizeof(reduction), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING,
            &swizzle, sizeof(swizzle), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CUSTOM_OPTION,
            &custom, sizeof(custom), NULL);

        // INNER_SHAPE_ID and CLUSTER_SHAPE_ID are cuBLAS 12.0+ (the latter
        // is Hopper-specific). Sentinel -1 on older toolchains.
#if defined(CUBLAS_VERSION) && CUBLAS_VERSION >= 120000
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID,
            &inner_shape, sizeof(inner_shape), NULL);
        cublasLtMatmulAlgoConfigGetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            &cluster_shape, sizeof(cluster_shape), NULL);
#endif

        printf("%d\t%d\t%s(%d)\t%d\t%d\t%s(%d)\t%d\t%d\t%d\t%d\t%.2f\t%zu\n",
               i, algo_id, tile_name(tile), tile, stages, split_k,
               reduction_name(reduction), reduction, swizzle, custom,
               inner_shape, cluster_shape,
               results[i].wavesCount, results[i].workspaceSize);
    }

    delete[] results;
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(A_l);
    cublasLtMatrixLayoutDestroy(B_l);
    cublasLtMatrixLayoutDestroy(C_l);
    cublasLtMatmulDescDestroy(desc);
    cublasLtDestroy(lt);
    return 0;
}
