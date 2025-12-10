#include <cuda_runtime.h>
#include <stdio.h>

__global__ void atomicExchKernel(float *data, float newValue, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float oldValue = atomicExch(&data[idx], newValue);
        data[idx] = oldValue + newValue;
    }
}

int main() {
    int N = 1024;  // Array size
    size_t size = N * sizeof(float);
    float *h_data = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    float *d_data;
    cudaMalloc((void**)&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    atomicExchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, 100.0f, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    printf("AtomicExch result (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_data[i]);
    }
    printf("\n");

    cudaFree(d_data);
    free(h_data);
    return 0;
}
