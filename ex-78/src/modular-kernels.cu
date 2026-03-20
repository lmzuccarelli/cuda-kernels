#include <cuda_runtime.h>
#include <stdio.h>

__global__ void filterKernel(float* data, int N, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (data[idx] < threshold) {
            data[idx] = 0.0f; // mark invalid
        }
    }
}

__global__ void transformKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        if (data[idx] != 0.0f) {
            data[idx] *= 2.0f;
        }
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *d_data;
    cudaMalloc(&d_data, size);

    // Initialize or copy data
    // ...

    dim3 block(256);
    dim3 grid((N + block.x - 1)/block.x);
    filterKernel<<<grid, block>>>(d_data, N, 0.5f);
    cudaDeviceSynchronize();

    transformKernel<<<grid, block>>>(d_data, N);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    return 0;
}
