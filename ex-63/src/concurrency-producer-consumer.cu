// producerConsumerHost.cu
#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

__global__ void producerKernel(float *buffer, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        buffer[idx] = (float)idx;
    }
}

__global__ void consumerKernel(const float *buffer, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = buffer[idx] * 2.0f;
    }
}

int main() {
    int N = 1 << 20; 
    size_t size = N * sizeof(float);

    float *d_buffer, *d_output;
    cudaMalloc(&d_buffer, size);
    cudaMalloc(&d_output, size);

    cudaStream_t streamA, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamB);

    cudaEvent_t event;
    cudaEventCreate(&event);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    producerKernel<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(d_buffer, N);

    cudaEventRecord(event, streamA);

    cudaStreamWaitEvent(streamB, event, 0);

    consumerKernel<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(d_buffer, d_output, N);

    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    float *h_output = (float*)malloc(size);
    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    std::cout << "Sample output (first 10 elements):\n";
    for (int i = 0; i < 10; i++) {
        std::cout << h_output[i] << " ";
    }
    std::cout << std::endl;

    free(h_output);
    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamB);
    cudaEventDestroy(event);

    return 0;
}
