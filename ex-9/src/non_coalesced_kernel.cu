#include <cuda_runtime.h>
#include <stdio.h>

// Kernel with non-coalesced memory access
__global__ void nonCoalescedKernel(const float *input, float *output, int N, int stride) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        int index = (idx * stride) % N; // Wrap around to match workload
        output[index] = input[index] * 2.0f;
    }
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);
    cudaError_t err;

    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    // Initialize host array
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f;
    }

    // Allocate device memory
    float *d_input, *d_output;
    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }
    err = cudaMalloc((void**)&d_output, size);
    if (err != cudaSuccess) { fprintf(stderr, "Error: %s\n", cudaGetErrorString(err)); exit(EXIT_FAILURE); }

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // Configure kernel execution
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int stride = 2;

    // Benchmark non-coalesced kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    nonCoalescedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N, stride);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print result
    printf("Non-Coalesced Kernel Time (stride=%d): %f ms\n", stride, milliseconds);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
