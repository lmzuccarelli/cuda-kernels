#include <cuda_runtime.h>
#include <stdio.h>
#include <vector>

struct DeviceMemoryPool {
    char *poolStart;
    size_t poolSize;
    size_t offset;

    DeviceMemoryPool() : poolStart(nullptr), poolSize(0), offset(0) {}

    void init(size_t size) {
        cudaMalloc(&poolStart, size);
        poolSize = size;
        offset = 0;
    }

    void* allocate(size_t size) {
        size_t newOffset = offset + size;
        if (newOffset > poolSize) {
            return nullptr;
        }
        void* ptr = (void*)(poolStart + offset);
        offset = newOffset;
        return ptr;
    }

    void freeAll() {
        offset = 0; 
    }

    void destroy() {
        cudaFree(poolStart);
        poolStart = nullptr;
    }
};

__global__ void sampleKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] + 1.0f;
    }
}

int main() {
    DeviceMemoryPool pool;
    size_t poolBytes = 32 * 1024 * 1024;
    pool.init(poolBytes);

    size_t N = 10000000;
    size_t allocSize = N * sizeof(float);
    float* d_data = (float*) pool.allocate(allocSize);
    if (!d_data) {
        printf("Pool allocation failed.\n");
        return 0;
    }

    cudaMemset(d_data, 0, allocSize);

    int threadsPerBlock = 256;
    int blocksPerGrid = (int)((N + threadsPerBlock - 1) / threadsPerBlock);
    sampleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, N);
    cudaDeviceSynchronize();

    std::vector<float> h_data(N);
    cudaMemcpy(h_data.data(), d_data, allocSize, cudaMemcpyDeviceToHost);
    printf("Sample output[0] after kernel = %f\n", h_data[0]);

    pool.destroy();
    return 0;
}
