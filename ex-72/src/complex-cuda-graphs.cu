#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernel1(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

__global__ void kernel2(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        data[idx] += 100.0f;
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);

    float* h_data;
    cudaMallocHost((void**)&h_data, size);
    float* d_data;
    cudaMalloc(&d_data, size);

    for(int i=0; i<N; i++) {
        h_data[i] = (float)i;
    }

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaMemcpy3DParms copyParams = {0};
    copyParams.srcPtr = make_cudaPitchedPtr((void*)h_data, size, N, 1);
    copyParams.dstPtr = make_cudaPitchedPtr((void*)d_data, size, N, 1);
    copyParams.extent = make_cudaExtent(size, 1, 1);
    copyParams.kind   = cudaMemcpyHostToDevice;

    cudaGraphNode_t h2dNode;
    cudaGraphAddMemcpyNode(&h2dNode, graph, nullptr, 0, &copyParams);

    cudaGraphNode_t k1Node;
    cudaKernelNodeParams k1Params = {0};
    dim3 block(256), grid((N+255)/256);
    void* k1Args[2] = {(void*)&d_data, (void*)&N};
    k1Params.func = (void*)kernel1;
    k1Params.gridDim = grid;
    k1Params.blockDim = block;
    k1Params.sharedMemBytes = 0;
    k1Params.kernelParams = k1Args;
    cudaGraphAddKernelNode(&k1Node, graph, &h2dNode, 1, &k1Params);

    cudaGraphNode_t k2Node;
    cudaKernelNodeParams k2Params = k1Params; // copy base config
    k2Params.func = (void*)kernel2;  // different kernel
    cudaGraphAddKernelNode(&k2Node, graph, &k1Node, 1, &k2Params);

    cudaGraphNode_t d2hNode;
    cudaMemcpy3DParms copyParamsBack = {0};
    copyParamsBack.srcPtr = make_cudaPitchedPtr((void*)d_data, size, N, 1);
    copyParamsBack.dstPtr = make_cudaPitchedPtr((void*)h_data, size, N, 1);
    copyParamsBack.extent = make_cudaExtent(size, 1, 1);
    copyParamsBack.kind   = cudaMemcpyDeviceToHost;
    cudaGraphAddMemcpyNode(&d2hNode, graph, &k2Node, 1, &copyParamsBack);

    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    cudaGraphLaunch(graphExec, 0);
    cudaDeviceSynchronize();

    printf("Sample h_data[0] after graph run = %f\n", h_data[0]);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaFreeHost(h_data);
    cudaFree(d_data);
    return 0;
}
