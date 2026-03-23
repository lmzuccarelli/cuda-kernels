#include <cuda_runtime.h>
#include <stdio.h>

__global__ void kernelA(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] += 1.0f;
}

__global__ void kernelB(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] *= 2.0f;
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_data;
    cudaMalloc(&d_data, size);
    cudaMemset(d_data, 0, size);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Timing kernelA
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    cudaEventRecord(start, stream);
    kernelA<<<blocks, threads, 0, stream>>>(d_data, N);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    
    bool fuseKernels = (elapsed < 1.0f);

    cudaGraph_t graph;
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    
    kernelA<<<blocks, threads, 0, stream>>>(d_data, N);
    if (!fuseKernels) {
        kernelB<<<blocks, threads, 0, stream>>>(d_data, N);
    } else {
        kernelB<<<blocks, threads, 0, stream>>>(d_data, N);
    }
    
    cudaStreamEndCapture(stream, &graph);
    
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    return 0;
}
