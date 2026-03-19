#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void processKernel(const float *inputPinned, float *umIntermediate, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        umIntermediate[idx] = inputPinned[idx] * 2.5f;
    }
}

int main() {
    int N = 1 << 20; 
    size_t size = N * sizeof(float);

    float *h_pinned;
    cudaMallocHost((void**)&h_pinned, size);

    for (int i = 0; i < N; i++) {
        h_pinned[i] = (float)(rand() % 100);
    }

    float *umIntermediate;
    cudaMallocManaged(&umIntermediate, size);

    cudaMemcpyAsync(umIntermediate, h_pinned, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    processKernel<<<blocksPerGrid, threadsPerBlock>>>(umIntermediate, umIntermediate, N);
    cudaDeviceSynchronize();

    printf("First 10 results after processing:\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", umIntermediate[i]);
    }
    printf("\n");

    cudaFreeHost(h_pinned);
    cudaFree(umIntermediate);

    return 0;
}
