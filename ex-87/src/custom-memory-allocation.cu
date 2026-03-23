#include <cuda_runtime.h>
#include <stdio.h>

struct DevicePool {
    char* base;      // Base pointer of the pool
    size_t poolSize; // Total size of the pool in bytes
    size_t* offset;  // Pointer to a global offset counter (device memory)
};

__device__ void* streamOrderedAllocate(DevicePool* dp, size_t size) {
    size_t oldOffset = atomicAdd(dp->offset, size);
    if (oldOffset + size > dp->poolSize)
        return nullptr;
    return (void*)(dp->base + oldOffset);
}

__global__ void allocationKernel(DevicePool* dp, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    void* ptr = streamOrderedAllocate(dp, sizeof(float));
    if (ptr != nullptr && idx < N) {
        ((float*)ptr)[0] = 3.14f; // Write a sample value
    }
    if (idx < N)
        output[idx] = 1.0f;
}

int main() {
    int N = 1024;
    size_t poolSize = 1024 * sizeof(float);

    char* d_poolBase;
    size_t* d_offset;
    cudaMalloc(&d_poolBase, poolSize);
    cudaMalloc(&d_offset, sizeof(size_t));
    cudaMemset(d_offset, 0, sizeof(size_t));

    DevicePool h_dp;
    h_dp.base = d_poolBase;
    h_dp.poolSize = poolSize;
    h_dp.offset = d_offset;

    DevicePool* d_dp;
    cudaMalloc(&d_dp, sizeof(DevicePool));
    cudaMemcpy(d_dp, &h_dp, sizeof(DevicePool), cudaMemcpyHostToDevice);

    float* d_output;
    cudaMalloc(&d_output, N * sizeof(float));
    cudaMemset(d_output, 0, N * sizeof(float));

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    allocationKernel<<<blocks, threads>>>(d_dp, d_output, N);
    cudaDeviceSynchronize();

    cudaFree(d_poolBase);
    cudaFree(d_offset);
    cudaFree(d_dp);
    cudaFree(d_output);
    return 0;
}
