#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void processChunk(const float *d_input, float *d_output, int start, int chunkSize, int totalSize) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int globalIdx = start + idx;

    if (idx < chunkSize && globalIdx < totalSize) {
        d_output[globalIdx] = d_input[globalIdx] * 2.0f;
    }
}

int main() {
    // 16 million elements
    int totalSize = 1 << 24; 
    size_t size = totalSize * sizeof(float);

    float *h_input = (float*)malloc(size);
    float *h_output = (float*)malloc(size);

    for (int i = 0; i < totalSize; i++) {
        h_input[i] = (float)(i % 100); // Values between 0 and 99
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // 1 million elementrs per chunk
    int chunkSize = 1 << 20; 
    int numChunks = (totalSize + chunkSize - 1) / chunkSize;

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threadsPerBlock = 256;
    int blocksPerGrid;

    for (int chunk = 0; chunk < numChunks; chunk++) {
        int start = chunk * chunkSize;
        int currentChunkSize = ((start + chunkSize) > totalSize) ? (totalSize - start) : chunkSize;
        
        blocksPerGrid = (currentChunkSize + threadsPerBlock - 1) / threadsPerBlock;
        
        processChunk<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(d_input, d_output, start, currentChunkSize, totalSize);
    }
    
    cudaStreamSynchronize(stream);

    cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost);

    printf("Sample Output: h_output[0] = %f\n", h_output[0]);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    cudaStreamDestroy(stream);

    return 0;
}
