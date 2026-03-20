#include <cuda_runtime.h>
#include <stdio.h>
#include <thread>
#include <chrono>

__global__ void kernelA(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

__global__ void kernelB(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

void cpuSideTask() {
    printf("CPU task started...\n");
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    printf("CPU task completed.\n");
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float* h_data;
    cudaMallocHost((void**)&h_data, size);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    float* d_data;
    cudaMalloc(&d_data, size);

    cudaStream_t streamA, streamCopy, streamB;
    cudaStreamCreate(&streamA);
    cudaStreamCreate(&streamCopy);
    cudaStreamCreate(&streamB);

    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, streamCopy);

    std::thread cpuThread(cpuSideTask);

    cudaEvent_t copyDone;
    cudaEventCreate(&copyDone);
    cudaEventRecord(copyDone, streamCopy);
    cudaStreamWaitEvent(streamA, copyDone, 0);
    kernelA<<<(N+255)/256, 256, 0, streamA>>>(d_data, N);

    cudaEvent_t kernelA_done;
    cudaEventCreate(&kernelA_done);
    cudaEventRecord(kernelA_done, streamA);

    cudaStreamWaitEvent(streamB, kernelA_done, 0);
    kernelB<<<(N+255)/256, 256, 0, streamB>>>(d_data, N);

    cudaStreamSynchronize(streamCopy);
    cudaStreamSynchronize(streamA);
    cudaStreamSynchronize(streamB);

    cpuThread.join();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("Sample result: h_data[0] = %f\n", h_data[0]);

    cudaFree(d_data);
    cudaFreeHost(h_data);
    cudaStreamDestroy(streamA);
    cudaStreamDestroy(streamCopy);
    cudaStreamDestroy(streamB);
    cudaEventDestroy(copyDone);
    cudaEventDestroy(kernelA_done);
    return 0;
}
