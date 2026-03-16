// streamSyncDependencies.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel A: Perform simple vector addition.
__global__ void kernelA(const int *A, const int *B, int *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

// Kernel B: Multiply each element by 2 (depends on the output of Kernel A).
__global__ void kernelB(const int *C, int *D, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        D[idx] = C[idx] * 2;
    }
}

// Macro for error checking.
#define CUDA_CHECK(call) {                                      \
    cudaError_t err = call;                                     \
    if (err != cudaSuccess) {                                   \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err));                        \
        exit(EXIT_FAILURE);                                     \
    }                                                           \
}

int main() {
    const int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(int);

    // Allocate host memory.
    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);
    int *h_D = (int*)malloc(size);
    if (!h_A || !h_B || !h_D) {
        printf("Failed to allocate host memory\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays.
    for (int i = 0; i < N; i++) {
        h_A[i] = rand() % 100;
        h_B[i] = rand() % 100;
    }

    // Allocate device memory.
    int *d_A, *d_B, *d_C, *d_D;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    CUDA_CHECK(cudaMalloc(&d_D, size));

    // Copy input data from host to device.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Create two CUDA streams.
    cudaStream_t streamA, streamB;
    CUDA_CHECK(cudaStreamCreate(&streamA));
    CUDA_CHECK(cudaStreamCreate(&streamB));

    // Create a CUDA event to signal the completion of Kernel A.
    cudaEvent_t eventA;
    CUDA_CHECK(cudaEventCreate(&eventA));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel A in streamA.
    kernelA<<<blocksPerGrid, threadsPerBlock, 0, streamA>>>(d_A, d_B, d_C, N);

    // Record eventA in streamA after Kernel A finishes.
    CUDA_CHECK(cudaEventRecord(eventA, streamA));

    // In streamB, wait for eventA to complete before launching Kernel B.
    CUDA_CHECK(cudaStreamWaitEvent(streamB, eventA, 0));

    // Launch Kernel B in streamB.
    kernelB<<<blocksPerGrid, threadsPerBlock, 0, streamB>>>(d_C, d_D, N);

    // Synchronize both streams.
    CUDA_CHECK(cudaStreamSynchronize(streamA));
    CUDA_CHECK(cudaStreamSynchronize(streamB));

    // Copy final results from device to host.
    CUDA_CHECK(cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost));

    // Verify results (for example, print first 10 elements).
    printf("First 10 elements of final result (Kernel B output):\n");
    for (int i = 0; i < 10; i++) {
        printf("%d ", h_D[i]);
    }
    printf("\n");

    // Cleanup: Free device and host memory, destroy streams and events.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFree(d_D));
    free(h_A);
    free(h_B);
    free(h_D);
    CUDA_CHECK(cudaStreamDestroy(streamA));
    CUDA_CHECK(cudaStreamDestroy(streamB));
    CUDA_CHECK(cudaEventDestroy(eventA));

    return 0;
}
