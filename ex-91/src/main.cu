// main.cu
#include <stdio.h>
#include <cuda_runtime.h>
#include "kernel-functions.h"

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *h_in, *h_out;
    float *d_in, *d_out;

    h_in = (float*)malloc(size);
    h_out = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }

    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_in, size, cudaMemcpyHostToDevice);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    addKernel<<<blocks, threads>>>(d_out, d_in, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_out, size, cudaMemcpyDeviceToHost);
    printf("Result: h_out[0] = %f\n", h_out[0]);

    free(h_in);
    free(h_out);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
