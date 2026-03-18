// multi-process-service
// set envars
// export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
// export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps
//
// launch mps
// sudo nvidia-cuda-mps-control -d

#include <cuda_runtime.h>
#include <stdio.h>
#define CUDA_CHECK(call) do {                                         \
    cudaError_t err = call;                                           \
    if (err != cudaSuccess) {                                         \
        fprintf(stderr, "CUDA Error: %s\n", cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                                           \
    }                                                                 \
} while (0)

__global__ void dummyKernel1(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

int main(){
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    dummyKernel1<<<(N+255)/256, 256>>>(d_data, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_data));
    printf("Process A done.\n");
    return 0;
}
