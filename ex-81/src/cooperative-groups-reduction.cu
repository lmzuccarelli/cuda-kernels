#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <stdio.h>
namespace cg = cooperative_groups;

__launch_bounds__(256)
__global__ void cooperativeReductionKernel(const float* input, float* output, int N) {
    cg::grid_group grid = cg::this_grid();

    extern __shared__ float sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
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
            sdata[tid] = output[tid];
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

int main() {
    int N = 1 << 20; 
    size_t size = N * sizeof(float);
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, (N / 256 + 1) * sizeof(float)); 

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    cooperativeReductionKernel<<<grid, block, block.x * sizeof(float)>>>(d_input, d_output, N);
    cudaDeviceSynchronize();

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}
