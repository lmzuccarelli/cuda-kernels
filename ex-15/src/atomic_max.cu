#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

__device__ float atomicMaxFloat(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void atomicMaxKernel(const float *input, float *result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        atomicMaxFloat(result, input[idx]);
    }
}

int main() {
    int N = 1 << 20;  
    size_t size = N * sizeof(float);
    float *h_input = (float*)malloc(size);
    float h_result = -FLT_MAX; 

    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 101);
    }

    float *d_input, *d_result;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_result, sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    atomicMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_result, N);
    cudaDeviceSynchronize();

    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Maximum value found = %f\n", h_result);

    cudaFree(d_input);
    cudaFree(d_result);
    free(h_input);
    return 0;
}
