#include <cuda_runtime.h>
#include <stdio.h>

__global__ void processKernel(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_data[idx] *= 2.0f;
}

int main() {
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);
    float *h_data, *d_data;

    cudaMallocHost((void**)&h_data, size);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }
    
    cudaMalloc(&d_data, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    processKernel<<<blocks, threads, 0, stream>>>(d_data, N);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaStreamDestroy(stream);
    
    return 0;
}
