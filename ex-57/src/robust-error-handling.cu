// Error checking macro for CUDA Runtime API calls
// example 

/*
#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Error checking macro for cuBLAS calls
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = call;                                           \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            fprintf(stderr, "cuBLAS Error at %s:%d - code %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

// Error checking macro for cuFFT calls
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult status = call;                                              \
        if (status != CUFFT_SUCCESS) {                                          \
            fprintf(stderr, "cuFFT Error at %s:%d - code %d\n", __FILE__, __LINE__, status); \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)
  */

#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = call;                                                 \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                   \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

__global__ void sampleKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    int N = 1 << 20; 
    size_t size = N * sizeof(float);
    float *h_data = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_data[i] = 1.0f;
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    sampleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost));
    printf("Sample output: h_data[0] = %f\n", h_data[0]);

    CUDA_CHECK(cudaFree(d_data));
    free(h_data);

    return 0;
}
