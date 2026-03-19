#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>

namespace cg = cooperative_groups;

__global__ void cooperativeReductionKernel(const float *input, float *output, int N) {
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    float val = (idx < N) ? input[idx] : 0.0f;
    sdata[tid] = val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }

    grid.sync();

    if (blockIdx.x == 0) {
        if (tid < gridDim.x) {
            sdata[tid] = (tid < gridDim.x) ? output[tid] : 0.0f;
        }
        __syncthreads();
        
        for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
            if (tid < stride && tid + stride < gridDim.x) {
                sdata[tid] += sdata[tid + stride];
            }
            __syncthreads();
        }

        if (tid == 0) {
            output[0] = sdata[0];
        }
    }
}

int main(){
    int N = 1 << 20; 
    size_t size = N * sizeof(float);

    float *h_input = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_input[i] = 1.0f; 
    }

    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, sizeof(float) * 1024); // Enough to hold partial sums from each block

    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    cooperativeReductionKernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    float result;
    cudaMemcpy(&result, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Final reduction result: %f\n", result);

    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    return 0;
}
