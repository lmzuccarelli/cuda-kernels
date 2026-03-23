#include <cuda_runtime.h>
#include <stdio.h>

__global__ void fusedKernel(float* data, int N, float addVal, float mulVal) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float temp = data[idx] + addVal;
        data[idx] = temp * mulVal;
    }
}

int main() {
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, size);

    cudaMemset(d_data, 1, size);  // Note: This sets bytes to 1, not floats. For proper initialization, a kernel or host copy is preferred.

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float addVal = 2.0f, mulVal = 3.0f;
    fusedKernel<<<blocks, threads>>>(d_data, N, addVal, mulVal);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    return 0;
}


