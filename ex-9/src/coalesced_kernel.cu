#include <cuda_runtime.h>
#include <stdio.h>

// Kernel with coalesced memory access
__global__ void coalescedKernel(const float *input, float *output, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * 2.0f;
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

    // Benchmark coalesced kernel
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    coalescedKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print result
    printf("Coalesced Kernel Time: %f ms\n", milliseconds);

    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
