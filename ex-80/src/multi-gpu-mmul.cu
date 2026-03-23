#include <cuda_runtime.h>
#include <stdio.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

__global__ void matrixMulKernel(const float* A, const float* B, float* C, 
                                int M, int N, int K, int rowOffset) {
    int row = blockIdx.y * blockDim.y + threadIdx.y + rowOffset;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += A[IDX2C(row, i, M)] * B[IDX2C(i, col, N)];
        }
        C[IDX2C(row, col, M)] = sum;
    }
}

int main() {
    int M = 1024, N = 1024, K = 1024;
    size_t sizeA = M * N * sizeof(float);
    size_t sizeB = N * K * sizeof(float);
    size_t sizeC = M * K * sizeof(float);

    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);


    float *d_A0, *d_C0; // GPU 0 will handle rows 0 to M/2
    float *d_A1, *d_C1; // GPU 1 will handle rows M/2 to M
    float *d_B;        // B is needed in full on both GPUs

    int M_half = M / 2;
    size_t sizeA0 = M_half * N * sizeof(float);
    size_t sizeA1 = (M - M_half) * N * sizeof(float);
    size_t sizeC0 = M_half * K * sizeof(float);
    size_t sizeC1 = (M - M_half) * K * sizeof(float);

    cudaSetDevice(0);
    cudaMalloc(&d_A0, sizeA0);
    cudaMalloc(&d_C0, sizeC0);
    cudaMalloc(&d_B, sizeB); // B allocated on GPU 0

    cudaSetDevice(1);
    cudaMalloc(&d_A1, sizeA1);
    cudaMalloc(&d_C1, sizeC1);
    cudaMalloc(&d_B, sizeB); // or use cudaMemcpyPeer if P2P enabled

    cudaSetDevice(0);
    cudaMemcpy(d_A0, h_A, sizeA0, cudaMemcpyHostToDevice);
    cudaSetDevice(1);
    cudaMemcpy(d_A1, h_A + M_half * N, sizeA1, cudaMemcpyHostToDevice);
    cudaSetDevice(0);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);
    cudaSetDevice(1);
    cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid0((K + block.x - 1) / block.x, (M_half + block.y - 1) / block.y);
    dim3 grid1((K + block.x - 1) / block.x, ((M - M_half) + block.y - 1) / block.y);

    cudaSetDevice(0);
    matrixMulKernel<<<grid0, block>>>(d_A0, d_B, d_C0, M, N, K, 0);
    cudaSetDevice(1);
    matrixMulKernel<<<grid1, block>>>(d_A1, d_B, d_C1, M, N, K, M_half);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    cudaSetDevice(0);
    cudaMemcpy(h_C, d_C0, sizeC0, cudaMemcpyDeviceToHost);
    cudaSetDevice(1);
    cudaMemcpy(h_C + M_half * K, d_C1, sizeC1, cudaMemcpyDeviceToHost);

    cudaSetDevice(0);
    cudaFree(d_A0);
    cudaFree(d_C0);
    cudaFree(d_B);
    cudaSetDevice(1);
    cudaFree(d_A1);
    cudaFree(d_C1);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
