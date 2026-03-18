#include <cuda_runtime.h>
#include <stdio.h>
#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

__global__ void dummyKernel2(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main(){
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    dummyKernel2<<<(N+255)/256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
    printf("Process 2 done.\n");
    return 0;
}
