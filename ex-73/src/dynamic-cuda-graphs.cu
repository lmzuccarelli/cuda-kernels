#include <cuda_runtime.h>
#include <stdio.h>

__global__ void mainKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 1.0f;
    }
}

__global__ void extraKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] *= 2.0f;
    }
}

int main() {
    int N = 1 << 20; // 1 million
    size_t size = N * sizeof(float);

    bool runExtra = true; // dynamic condition in real code

    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0, size);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t nodeMain;
    cudaKernelNodeParams mainParams = {0};
    dim3 block(256), grid((N+255)/256);
    void* mainArgs[2] = {(void*)&d_data, (void*)&N};

    mainParams.func = (void*)mainKernel;
    mainParams.gridDim = grid;
    mainParams.blockDim = block;
    mainParams.sharedMemBytes = 0;
    mainParams.kernelParams = mainArgs;
    cudaGraphAddKernelNode(&nodeMain, graph, nullptr, 0, &mainParams);

    cudaGraphNode_t nodeExtra;
    if(runExtra) {
        cudaKernelNodeParams extraParams = mainParams; // copy base config
        extraParams.func = (void*)extraKernel;
        cudaGraphAddKernelNode(&nodeExtra, graph, &nodeMain, 1, &extraParams);
    }

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFree(d_data);

    return 0;
}
