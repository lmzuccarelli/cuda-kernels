#include <cuda_runtime.h>
#include <stdio.h>

__launch_bounds__(128, 2)
__global__ void matrixMulKernel(const float* A, const float* B, float* C,
                                int N) {
    extern __shared__ float sdataA[]; // Tiled approach
    float* sdataB = &sdataA[blockDim.x * blockDim.y];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * blockDim.y + ty;
    int col = blockIdx.x * blockDim.x + tx;
    float sum = 0.0f;

    int tileSize = blockDim.x;

    for (int tile = 0; tile < N / tileSize; tile++) {
        int A_idx = row * N + (tile * tileSize + tx);
        int B_idx = (tile * tileSize + ty) * N + col;

        sdataA[ty * tileSize + tx] = (row < N && tile * tileSize + tx < N) ?
                                     A[A_idx] : 0.0f;
        sdataB[ty * tileSize + tx] = (col < N && tile * tileSize + ty < N) ?
                                     B[B_idx] : 0.0f;

        __syncthreads();

        for (int k = 0; k < tileSize; k++) {
            sum += sdataA[ty * tileSize + k] * sdataB[k * tileSize + tx];
        }
        __syncthreads();
    }

    if (row < N && col < N) {
        C[row * N + col] = sum;
    }
}

int main() {
    int N = 2048;
    size_t matrixSize = N * N * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, matrixSize);
    cudaMalloc(&d_B, matrixSize);
    cudaMalloc(&d_C, matrixSize);


    dim3 block(16, 16);
    dim3 grid((N + block.x - 1)/ block.x,
              (N + block.y - 1)/ block.y);

    int sharedMemBytes = 2 * block.x * block.y * sizeof(float);

    matrixMulKernel<<<grid, block, sharedMemBytes>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();

    return 0;
}
