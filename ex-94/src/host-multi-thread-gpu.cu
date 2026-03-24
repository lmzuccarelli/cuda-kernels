#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>
#include <vector>
#include <algorithm>

__global__ void sampleKernel(float* d_data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] *= 2.0f;
    }
}

void threadFunction(int threadId, int N) {
    cudaSetDevice(0);
    
    size_t size = N * sizeof(float);
    float* d_data;
    cudaMalloc(&d_data, size);
    
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    
    sampleKernel<<<blocks, threads, 0, stream>>>(d_data, N);
    
    cudaStreamSynchronize(stream);
    
    cudaFree(d_data);
    cudaStreamDestroy(stream);
    
    printf("Thread %d completed kernel execution.\n", threadId);
}

int main() {
    int N = 1 << 20; 
    const int numThreads = 4; 
    
    std::vector<std::thread> threads;
    for (int i = 0; i < numThreads; ++i) {
        threads.emplace_back(threadFunction, i, N);
    }
    
    for (auto& t : threads) {
        t.join();
    }
    
    return 0;
}
