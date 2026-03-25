#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void simpleKernel(float* d_data, int N, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        d_data[idx] += value;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int N = 1 << 20; // 1 million elements per MPI process
    size_t size = N * sizeof(float);

    float* h_data = (float*)malloc(size);
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)world_rank; // Initialize with the rank for differentiation
    }
    float* d_data;
    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int num_threads = 4; // Adjust as needed
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        cudaSetDevice(0);
        cudaStream_t stream;
        cudaStreamCreate(&stream);

        int threads = 256;
        int blocks = (N + threads - 1) / threads;
        float value = 1.0f; // Each thread adds 1.0f
        simpleKernel<<<blocks, threads, 0, stream>>>(d_data, N, value);

        cudaStreamSynchronize(stream);
        cudaStreamDestroy(stream);
    }

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("MPI Rank %d: h_data[0] = %f\n", world_rank, h_data[0]);

    free(h_data);
    cudaFree(d_data);
    MPI_Finalize();
    return 0;
}
