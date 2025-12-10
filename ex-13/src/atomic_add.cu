#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


__global__ void atomicSumKernel(const float *input, float *result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    
    if (idx < N) {
        atomicAdd(result, input[idx]);
    }
}

int main() {
    int N = 1 << 20;  // Number of elements (e.g., 1M elements)
    size_t size = N * sizeof(float);
    
    float *h_input = (float*)malloc(size);
    float h_result = 0.0f;
    
    srand(time(NULL));
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 100) / 100.0f;  // Random float between 0 and 1
    }
    
    float *d_input, *d_result;
    cudaError_t err;
    err = cudaMalloc((void**)&d_input, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for input (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    err = cudaMalloc((void**)&d_result, sizeof(float));
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to allocate device memory for result (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_input);
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy input array from host to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_result);
        exit(EXIT_FAILURE);
    }
    
    err = cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result initialization to device (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_result);
        exit(EXIT_FAILURE);
    }
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    atomicSumKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_result, N);
    
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to launch atomicSumKernel (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_result);
        exit(EXIT_FAILURE);
    }
    
    cudaDeviceSynchronize();
    
    err = cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        fprintf(stderr, "Failed to copy result from device to host (error code %s)!\n", cudaGetErrorString(err));
        cudaFree(d_input);
        cudaFree(d_result);
        exit(EXIT_FAILURE);
    }
    
    printf("Parallel sum using atomicAdd: %f\n", h_result);
    
    cudaFree(d_input);
    cudaFree(d_result);
    free(h_input);
    
    return 0;
}
