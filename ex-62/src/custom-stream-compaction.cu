#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__device__ int isNonZero(int x) {
    return (x != 0) ? 1 : 0;
}

__global__ void streamCompactionKernel(const int *input, int *output, int *flag, int *indices, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        flag[idx] = isNonZero(input[idx]);
    }
    __syncthreads();

    if (idx < N) {
        int sum = 0;
        for (int i = 0; i < idx; i++) {
            sum += flag[i];
        }
        indices[idx] = sum;
    }
    __syncthreads();

    if (idx < N && flag[idx] == 1) {
        output[indices[idx]] = input[idx];
    }
}

int main() {
    int N = 1000000; // 1 million elements.
    size_t size = N * sizeof(int);

    int *h_input = (int*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_input[i] = (rand() % 10 < 3) ? 0 : (rand() % 1000 + 1);
    }

    int *d_input, *d_output, *d_flag, *d_indices;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    cudaMalloc(&d_flag, size);
    cudaMalloc(&d_indices, size);
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    streamCompactionKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, d_flag, d_indices, N);
    cudaDeviceSynchronize();

    int *h_output = (int*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    int compactSize = 0;
    cudaMemcpy(&compactSize, d_indices + N - 1, sizeof(int), cudaMemcpyDeviceToHost);
    compactSize += h_input[N - 1] != 0 ? 1 : 0;
    printf("Custom Compaction: Original size = %d, Compacted size = %d\n", N, compactSize);

    cudaFree(d_input); cudaFree(d_output); cudaFree(d_flag); cudaFree(d_indices);
    free(h_input); free(h_output);
    
    return 0;
}

