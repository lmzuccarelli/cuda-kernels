#include <cuda_runtime.h>
#include <stdio.h>

__global__ void doubleKernel(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount < 2) {
        printf("Requires at least two GPUs.\n");
        return 0;
    }

    int canAccessPeer01, canAccessPeer10;
    cudaDeviceCanAccessPeer(&canAccessPeer01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccessPeer10, 1, 0);

    if (canAccessPeer01 && canAccessPeer10) {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);
        printf("Peer-to-Peer enabled between GPU 0 and GPU 1.\n");
    } else {
        printf("Peer-to-Peer NOT available between GPU 0 and 1.\n");
    }

    int N = 2000000;
    int halfN = N / 2;

    float *d_gpu0, *d_gpu1;
    size_t sizeHalf = halfN * sizeof(float);

    cudaSetDevice(0);
    cudaMalloc(&d_gpu0, sizeHalf);

    cudaSetDevice(1);
    cudaMalloc(&d_gpu1, sizeHalf);

    cudaSetDevice(0);
    cudaMemset(d_gpu0, 1, sizeHalf); // pretend these are floats set to ~1
    cudaSetDevice(1);
    cudaMemset(d_gpu1, 1, sizeHalf);

    int threadsPerBlock = 256;
    int blocksPerGrid = (halfN + threadsPerBlock - 1) / threadsPerBlock;

    cudaSetDevice(0);
    doubleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_gpu0, halfN);

    cudaSetDevice(1);
    doubleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_gpu1, halfN);

    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaDeviceSynchronize(); // or do them separately

    if (canAccessPeer01 && canAccessPeer10) {
        float *d_combined;
        cudaSetDevice(0);
        cudaMalloc(&d_combined, N * sizeof(float));

        cudaMemcpyPeer(d_combined + halfN, 0, d_gpu1, 1, sizeHalf);
        
        cudaMemcpy(d_combined, d_gpu0, sizeHalf, cudaMemcpyDeviceToDevice);

        cudaFree(d_combined);
    }

    cudaSetDevice(0);
    cudaFree(d_gpu0);
    cudaSetDevice(1);
    cudaFree(d_gpu1);

    printf("Multi-GPU P2P example completed.\n");
    return 0;
}
