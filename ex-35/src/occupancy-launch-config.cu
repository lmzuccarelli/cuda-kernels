#include <cuda_runtime.h>
#include <stdio.h>

__global__ void partialReductionKernel(const float *input, float *output, int N) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int globalIndex = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (globalIndex < N) ? input[globalIndex] : 0.0f;
    __syncthreads();

    for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

int main() {
    int N = 1 << 20; // 1M
    size_t size = N * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc( ( (N + 255) / 256 ) * sizeof(float));
    for(int i = 0; i < N; i++) {
        h_input[i] = 1.0f; // or some random data
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, ((N+255)/256)*sizeof(float));

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int blockSize = 256; // We'll test e.g. 128, 256, 512, etc
    int gridSize = (N + blockSize - 1) / blockSize;

    partialReductionKernel<<<gridSize, blockSize, blockSize*sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_output, d_output, gridSize*sizeof(float), cudaMemcpyDeviceToHost);

    float finalSum = 0.0f;
    for(int i = 0; i < gridSize; i++){
        finalSum += h_output[i];
    }
    printf("Final partial reduction sum = %f\n", finalSum);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);

    return 0;
}
