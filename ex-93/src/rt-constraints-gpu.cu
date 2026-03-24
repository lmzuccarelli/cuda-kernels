#include <cuda_runtime.h>
#include <stdio.h>

__global__ void realTimeKernel(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float generated = idx * 0.001f;
        d_data[idx] = generated * 2.0f;
    }
}

int main() {
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);
    float* h_data;
    
    cudaMallocHost((void**)&h_data, size);
    
    
    float* d_data;
    cudaMalloc(&d_data, size);
    
    cudaStream_t rtStream;
    cudaStreamCreateWithPriority(&rtStream, cudaStreamNonBlocking, -1); // Highest priority
    
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    realTimeKernel<<<blocks, threads, 0, rtStream>>>(d_data, N);
    
    cudaStreamSynchronize(rtStream);
    
    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaStreamDestroy(rtStream);
    
    return 0;
}
