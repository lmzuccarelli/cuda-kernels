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

__global__ void safeKernel(int *d_data, int N) {
    int idx = threadIdx.x;
    if (idx < N) { 
        d_data[idx] = idx * 2;
    }
}

int main() {
    int N = 100;
    int *d_data;

    CUDA_CHECK(cudaMalloc((void**)&d_data, N * sizeof(int)));

    safeKernel<<<1, 128>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError()); 

    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_data));

    printf("Execution completed successfully\n");
    return 0;
}
