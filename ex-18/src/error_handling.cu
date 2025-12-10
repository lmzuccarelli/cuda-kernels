// robustErrorHandling.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Macro for error checking
#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// Kernel with boundary check
__global__ void safeKernel(int *d_data, int N) {
    int idx = threadIdx.x;
    if (idx < N) { // Prevents out-of-bounds access
        d_data[idx] = idx * 2;
    }
}

int main() {
    int N = 100;
    int *d_data;

    // Allocate memory with error checking
    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

    // Launch kernel safely
    safeKernel<<<1, 128>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError()); // Checks for kernel launch errors

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());

    // Free memory
    CUDA_CHECK(cudaFree(d_data));

    printf("Execution completed successfully\n");
    return 0;
}
