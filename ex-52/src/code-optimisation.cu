#include <cuda_runtime.h>
#include <stdio.h>

__global__ void baselineSumKernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        float sum = 0.0f;
        for(int i=0; i<8; i++){
            sum += input[idx*8 + i];
        }
        output[idx] = sum;
    }
}

__global__ void unrolledSumKernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        float sum = 0.0f;
        sum += input[idx*8 + 0];
        sum += input[idx*8 + 1];
        sum += input[idx*8 + 2];
        sum += input[idx*8 + 3];
        sum += input[idx*8 + 4];
        sum += input[idx*8 + 5];
        sum += input[idx*8 + 6];
        sum += input[idx*8 + 7];

        output[idx] = sum;
    }
}
