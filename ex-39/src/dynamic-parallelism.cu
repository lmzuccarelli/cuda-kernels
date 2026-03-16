#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void sumSegmentKernel(const float* data, float* segmentResult, int start, int length) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int globalIdx = start + blockIdx.x * blockDim.x + tid;
    if (tid < length) {
        sdata[tid] = data[globalIdx];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for(int stride = blockDim.x/2; stride>0; stride >>=1) {
        if(tid < stride) {
            sdata[tid]+= sdata[tid+stride];
        }
        __syncthreads();
    }
    if(tid == 0) {
        segmentResult[blockIdx.x] = sdata[0];
    }
}

__global__ void parentKernel(const float* data, float* results, int totalLen) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= totalLen) return;

    int segLength = 128; 
    int segStart = idx * segLength;
    if (segStart + segLength > totalLen) {
        segLength = totalLen - segStart;
    }

    if(segLength > 0) {
        dim3 block(128,1,1);
        dim3 grid(1,1,1);
        sumSegmentKernel<<<grid, block, block.x*sizeof(float)>>>(data, results+idx, segStart, segLength);
        cudaDeviceSynchronize(); 
    }
}

#define CUDA_CHECK(call) {                                    \
    cudaError_t err = call;                                   \
    if(err != cudaSuccess) {                                  \
        printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err));                      \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
}

int main(){
    int N = 1 << 20; // 1M
    size_t size = N * sizeof(float);
    float *h_data = (float*)malloc(size);
    for(int i=0; i<N; i++){
        h_data[i] = 1.0f; // trivial
    }

    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));

    float *d_results;
    CUDA_CHECK(cudaMalloc(&d_results, N*sizeof(float))); // each thread can store a partial sum

    dim3 parentBlock(256);
    dim3 parentGrid((N+parentBlock.x-1)/parentBlock.x);

    parentKernel<<<parentGrid, parentBlock>>>(d_data, d_results, N);
    CUDA_CHECK(cudaDeviceSynchronize());

    float *h_results = (float*)malloc(N*sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_results, d_results, N*sizeof(float), cudaMemcpyDeviceToHost));

    double finalSum=0.0;
    for(int i=0;i<N;i++){
        finalSum+= h_results[i];
    }
    printf("Final sum from dynamic sub-kernels= %f\n", finalSum);

    free(h_data);
    free(h_results);
    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_results));
    return 0;
}
