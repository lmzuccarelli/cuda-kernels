// kernelTuning.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Kernel: Vector Addition
// Each thread computes one element of the output vector.
// The kernel expects that the total number of threads covers the entire vector.
__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    // Calculate the global index of the thread.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    // Boundary check: ensure the index does not exceed the vector length.
    if (idx < N) {
        // Perform the addition operation.
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    // Vector length.
    int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    // Allocate host memory for input vectors and output vector.
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize host arrays with random values.
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    // Allocate device memory for vectors.
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    // Copy input data from host to device.
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // Define a range of block sizes to test (e.g., multiples of warp size 32).
    int blockSizes[] = {32, 64, 128, 256, 512, 1024};
    int numBlockSizes = sizeof(blockSizes) / sizeof(blockSizes[0]);
    float timeTaken;

    // Create CUDA events for timing.
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Loop through the different block sizes.
    for (int i = 0; i < numBlockSizes; i++) {
        int threadsPerBlock = blockSizes[i];
        int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

        // Record the start event.
        cudaEventRecord(start);

        // Launch the kernel with the current configuration.
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

        // Record the stop event.
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        // Calculate the elapsed time.
        cudaEventElapsedTime(&timeTaken, start, stop);
        printf("Block Size: %d, Blocks Per Grid: %d, Time Taken: %f ms\n", threadsPerBlock, blocksPerGrid, timeTaken);
    }

    // Optionally, copy the result back to host for correctness verification.
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory.
    free(h_A);
    free(h_B);
    free(h_C);

    // Destroy CUDA events.
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}


