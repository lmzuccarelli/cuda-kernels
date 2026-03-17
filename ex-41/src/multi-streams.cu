#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        C[idx] = A[idx] + B[idx];
    }
}

#define CUDA_CHECK(call) {                                            \
    cudaError_t err = call;                                           \
    if(err != cudaSuccess) {                                          \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,      \
               cudaGetErrorString(err));                              \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
}

int main(){
    int N = 1 << 20; // 1M
    size_t size = N * sizeof(float);

    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);
    float *h_D = (float*)malloc(size);

    srand(time(NULL));
    for(int i=0; i<N; i++){
        h_A[i] = (float)(rand() % 100);
        h_B[i] = (float)(rand() % 100);
    }

    // Allocate device memory
    float *d_A1, *d_B1, *d_C1;
    float *d_A2, *d_B2, *d_C2;
    CUDA_CHECK(cudaMalloc(&d_A1, size));
    CUDA_CHECK(cudaMalloc(&d_B1, size));
    CUDA_CHECK(cudaMalloc(&d_C1, size));
    CUDA_CHECK(cudaMalloc(&d_A2, size));
    CUDA_CHECK(cudaMalloc(&d_B2, size));
    CUDA_CHECK(cudaMalloc(&d_C2, size));

    cudaStream_t stream1, stream2;
    CUDA_CHECK(cudaStreamCreate(&stream1));
    CUDA_CHECK(cudaStreamCreate(&stream2));

    CUDA_CHECK(cudaMemcpyAsync(d_A1, h_A, size, cudaMemcpyHostToDevice, stream1));
    CUDA_CHECK(cudaMemcpyAsync(d_B1, h_B, size, cudaMemcpyHostToDevice, stream1));

    CUDA_CHECK(cudaMemcpyAsync(d_A2, h_A, size, cudaMemcpyHostToDevice, stream2));
    CUDA_CHECK(cudaMemcpyAsync(d_B2, h_B, size, cudaMemcpyHostToDevice, stream2));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock -1)/threadsPerBlock;

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream1>>>(d_A1, d_B1, d_C1, N);
    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream2>>>(d_A2, d_B2, d_C2, N);

    CUDA_CHECK(cudaMemcpyAsync(h_C, d_C1, size, cudaMemcpyDeviceToHost, stream1));
    CUDA_CHECK(cudaMemcpyAsync(h_D, d_C2, size, cudaMemcpyDeviceToHost, stream2));

    CUDA_CHECK(cudaStreamSynchronize(stream1));
    CUDA_CHECK(cudaStreamSynchronize(stream2));

    printf("h_C[0] = %f, h_D[0] = %f\n", h_C[0], h_D[0]);

    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaStreamDestroy(stream2));
    cudaFree(d_A1); cudaFree(d_B1); cudaFree(d_C1);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    free(h_A); free(h_B); free(h_C); free(h_D);

    return 0;
}
