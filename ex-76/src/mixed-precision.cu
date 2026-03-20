#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda::wmma;

__global__ void wmmaGemmKernel(half *A, half *B, float *C,
                               int M, int N, int K) {
    // Each warp will compute a tile of C using wmma
    // NOTE: M, N, K must be multiples of 16 for simplicity

    fragment<matrix_a, 16, 16, 16, half, col_major> aFrag;
    fragment<matrix_b, 16, 16, 16, half, col_major> bFrag;
    fragment<accumulator, 16, 16, 16, float> cFrag;

    fill_fragment(cFrag, 0.0f); // init to zero

    int warpM = (blockIdx.x * blockDim.x + threadIdx.x) / 32; // example

    load_matrix_sync(aFrag, A + warpM*16, K); // if col_major
    load_matrix_sync(bFrag, B + warpM*16, K);

    mma_sync(cFrag, aFrag, bFrag, cFrag);

    store_matrix_sync(C + warpM*16, cFrag, N, mem_col_major);
}

int main() {
    // host data, device memory, etc.
    // A, B in half, C in float
    // call wmmaGemmKernel<<<...>>>(A, B, C, M, N, K);

    return 0;
}
