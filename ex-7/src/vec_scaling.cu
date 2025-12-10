#include <cuda_runtime.h>
#include <stdio.h>

__global__ void scaleVector(const float *input, float *output, float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        output[idx] = input[idx] * scale;
    }
}

int main() {
    int N = 1024;                     
    size_t size = N * sizeof(float);  

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < N; i++) {
        h_input[i] = i * 1.0f;
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    scaleVector<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, 2.0f, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    for (int i = 0; i < 10; i++) {
        printf("h_output[%d] = %f\n", i, h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);

    free(h_input);
    free(h_output);

    return 0;
}
