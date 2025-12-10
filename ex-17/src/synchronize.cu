#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAddKernel(const float *A, const float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *h_A, *h_B, *h_C, *d_A, *d_B, *d_C;

    // Allocate host memory
    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    // Allocate device memory
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    // Copy data to device
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    vectorAddKernel<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("Kernel Execution Time: %f ms\n", milliseconds);

    // Launch kernel
    // vectorAddKernel<<<(N + 255) / 256, 256>>>(d_A, d_B, d_C, N);

    // Synchronize to ensure kernel execution is complete
    // cudaDeviceSynchronize();

    // Now safely copy data back to host
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("Result: %f\n", h_C[0]); // Correct result

    // Cleanup
    free(h_A);
    free(h_B);
    free(h_C);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}
