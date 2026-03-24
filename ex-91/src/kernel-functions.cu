#include "kernel-functions.h"

__global__ void addKernel(float* d_out, const float* d_in, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_out[idx] = d_in[idx] + 1.0f;
}
