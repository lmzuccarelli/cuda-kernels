#include <cuda_runtime.h>
#include <stdio.h>

__global__ void processData(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f; // read and write
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* umPtr;
    cudaMallocManaged(&umPtr, size);

    for(int i=0; i<N; i++){
        umPtr[i] = (float)i;
    }

    int deviceId = 0;
    cudaMemPrefetchAsync(umPtr, size, deviceId, 0);

    cudaMemAdvise(umPtr, size, cudaMemAdviseSetReadMostly, deviceId);

    cudaSetDevice(deviceId);
    processData<<<(N+255)/256, 256>>>(umPtr, N);
    cudaDeviceSynchronize();

    cudaMemPrefetchAsync(umPtr, size, cudaCpuDeviceId, 0);
    cudaDeviceSynchronize();

    printf("umPtr[0]=%f\n", umPtr[0]);

    cudaFree(umPtr);
    return 0;
}
