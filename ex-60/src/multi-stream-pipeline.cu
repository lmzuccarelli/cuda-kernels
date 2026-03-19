#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) do {                                 \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        fprintf(stderr, "CUDA Error: %s (line %d)\n",         \
                cudaGetErrorString(err), __LINE__);           \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
} while (0)

__global__ void transformKernel(const float *in, float *out, int chunkSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < chunkSize) {
        out[idx] = in[idx] * 2.0f + 1.0f;
    }
}

int main(){
    int totalFeedSize = 1 << 22; 
    int chunkSize = 1 << 20;     
    int numChunks = (totalFeedSize + chunkSize - 1) / chunkSize;

    float *h_in, *h_out;
    size_t chunkBytes = chunkSize * sizeof(float);
    CUDA_CHECK(cudaMallocHost((void**)&h_in, chunkBytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, chunkBytes));

    float *d_in[2], *d_out[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaMalloc(&d_in[i], chunkBytes));
        CUDA_CHECK(cudaMalloc(&d_out[i], chunkBytes));
    }

    cudaStream_t stream[2];
    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamCreate(&stream[i]));
    }

    int threadsPerBlock = 256;
    int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

    srand(time(NULL));
    for (int chunkIdx = 0; chunkIdx < numChunks; chunkIdx++) {
        int currentChunkSize = ((chunkIdx + 1) * chunkSize > totalFeedSize)
                                ? (totalFeedSize - chunkIdx * chunkSize)
                                : chunkSize;
        for (int j = 0; j < currentChunkSize; j++) {
            h_in[j] = (float)(rand() % 100);
        }

        int bufIndex = chunkIdx % 2;

        CUDA_CHECK(cudaMemcpyAsync(d_in[bufIndex], h_in,
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream[bufIndex]));

        transformKernel<<<(currentChunkSize + threadsPerBlock - 1)/threadsPerBlock, 
                          threadsPerBlock, 0, stream[bufIndex]>>>(
                          d_in[bufIndex], d_out[bufIndex], currentChunkSize);

        CUDA_CHECK(cudaMemcpyAsync(h_out, d_out[bufIndex],
                                   currentChunkSize * sizeof(float),
                                   cudaMemcpyDeviceToHost,
                                   stream[bufIndex]));

    }

    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaStreamSynchronize(stream[i]));
    }

    printf("Last chunk processed: h_out[0] = %f\n", h_out[0]);

    for (int i = 0; i < 2; i++) {
        CUDA_CHECK(cudaFree(d_in[i]));
        CUDA_CHECK(cudaFree(d_out[i]));
        CUDA_CHECK(cudaStreamDestroy(stream[i]));
    }
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));

}
