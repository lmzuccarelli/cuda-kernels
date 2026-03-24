#include <cuda_runtime.h>
#include <stdio.h>

__global__ void addKernel(float* d_data, int N, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] += value;
    }
}

__global__ void multiplyKernel(float* d_data, int N, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_data[idx] *= factor;
    }
}

int main() {
    // Problem size
    int N = 1 << 20; // 1 million elements
    size_t size = N * sizeof(float);

    // Allocate host memory
    float* h_data = (float*)malloc(size);
    // Initialize host data with some values
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    // Allocate device memory
    float* d_data;
    cudaMalloc(&d_data, size);

    // Create a CUDA stream for graph capture
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Begin graph capture
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

    // Asynchronously copy data from host to device
    cudaMemcpyAsync(d_data, h_data, size, cudaMemcpyHostToDevice, stream);

    // Define kernel launch parameters
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // Launch first kernel: addKernel with a value of 1.0
    float addValue = 1.0f;
    addKernel<<<blocks, threads, 0, stream>>>(d_data, N, addValue);

    // Launch second kernel: multiplyKernel with a factor of 2.0
    float multFactor = 2.0f;
    multiplyKernel<<<blocks, threads, 0, stream>>>(d_data, N, multFactor);

    // Asynchronously copy data from device back to host
    cudaMemcpyAsync(h_data, d_data, size, cudaMemcpyDeviceToHost, stream);

    // End graph capture and obtain the graph
    cudaGraph_t graph;
    cudaStreamEndCapture(stream, &graph);

    // Instantiate the captured graph into an executable graph object
    cudaGraphExec_t graphExec;
    cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0);

    // Launch the graph and synchronize the stream
    cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    // (Optional) Verify results - e.g., print the first element
    printf("Result h_data[0]: %f\n", h_data[0]);

    // Cleanup: Free device and host memory, destroy graph and stream
    cudaGraphExecDestroy(graphExec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    free(h_data);

    return 0;
}
