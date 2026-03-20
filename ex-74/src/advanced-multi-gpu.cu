// File: multi_gpu_split_example.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void doubleKernel(float *data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] *= 2.0f;
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("Need at least 2 GPUs.\n");
        return 0;
    }

    int N = 2000000;
    size_t halfN = N / 2;
    size_t halfSize = halfN * sizeof(float);

    float *d_gpu0, *d_gpu1;
    cudaSetDevice(0);
    cudaMalloc(&d_gpu0, halfSize);
    cudaMemset(d_gpu0, 1, halfSize); // pretend data ~1

    cudaSetDevice(1);
    cudaMalloc(&d_gpu1, halfSize);
    cudaMemset(d_gpu1, 1, halfSize);

    cudaSetDevice(0);
    doubleKernel<<<(halfN + 255)/256, 256>>>(d_gpu0, halfN);

    cudaSetDevice(1);
    doubleKernel<<<(halfN + 255)/256, 256>>>(d_gpu1, halfN);

    int canAccess01, canAccess10;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);
    if (canAccess01 && canAccess10) {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1,0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0,0);

        float* d_combined;
        cudaSetDevice(0);
        cudaMalloc(&d_combined, N * sizeof(float));
        cudaMemcpyPeer(d_combined + halfN, 0, d_gpu1, 1, halfSize);
        cudaMemcpy(d_combined, d_gpu0, halfSize, cudaMemcpyDeviceToDevice);
        cudaFree(d_combined);
    } else {
        // fallback: copy d_gpu1 to host, then to GPU0, etc.
    }

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaFree(d_gpu0);

    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaFree(d_gpu1);

    return 0;
}
