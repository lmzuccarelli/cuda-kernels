#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

#define CUDA_CHECK(call) { \
    cudaError_t err = call; \
    if(err != cudaSuccess) { \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

int main() {
    int N = 1 << 20; // 1M elements
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C1 = (float*)malloc(size); // Output for stream 1
    float *h_C2 = (float*)malloc(size); // Output for stream 2

    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)(rand() % 100) / 10.0f;
        h_B[i] = (float)(rand() % 100) / 10.0f;
    }

    float *d_A, *d_B, *d_C1, *d_C2;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C1, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C2, size));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start, stream1));
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A, d_B, d_C1, N);
    CUDA_CHECK(cudaEventRecord(stop, stream1));

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A, d_B, d_C2, N);

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    float milliseconds = 0;
    CUDA_CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    printf("Stream 1 Kernel Execution Time: %f ms\n", milliseconds);

    CUDA_CHECK(cudaMemcpy(h_C1, d_C1, size, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_C2, d_C2, size, cudaMemcpyDeviceToHost));

    printf("First 5 elements from stream1 output:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_C1[i]);
    }
    printf("\n");

    printf("First 5 elements from stream2 output:\n");
    for (int i = 0; i < 5; i++) {
        printf("%f ", h_C2[i]);
    }
    printf("\n");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C1));
    CUDA_CHECK(cudaFree(d_C2));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    free(h_A);
    free(h_B);
    free(h_C1);
    free(h_C2);

    return 0;
}


