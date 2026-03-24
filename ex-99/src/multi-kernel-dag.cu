#include <cuda_runtime.h>
#include <stdio.h>

__global__ void incrementKernel(float* d_data, int N, float offset) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] += offset;
    }
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    float initOffset = 1.0f;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    incrementKernel<<<blocks, threads, 0, stream>>>(d_data, N, initOffset);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    float newOffset = 2.0f;
    cudaKernelNodeParams newKernelParams = {0};
    void* newArgs[3] = { (void*)&d_data, (void*)&N, (void*)&newOffset };
    newKernelParams.func = (void*)incrementKernel;
    newKernelParams.gridDim = dim3(blocks);
    newKernelParams.blockDim = dim3(threads);
    newKernelParams.sharedMemBytes = 0;
    newKernelParams.kernelParams = newArgs;

    size_t numUpdatedNodes;
    cudaGraphExecUpdate(graphExec, graph, NULL, NULL, &numUpdatedNodes);
    printf("Graph updated: %zu node(s) modified.\n", numUpdatedNodes);

    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    return 0;
}
