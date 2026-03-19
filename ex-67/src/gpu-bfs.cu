#include <cuda_runtime.h>
#include <stdio.h>

__global__ void bfsKernel(const int* __restrict__ d_rowOffsets,
                          const int* __restrict__ d_colIndices,
                          const int  frontierSize,
                          const int* __restrict__ frontier,
                          int* __restrict__ nextFrontier,
                          int* __restrict__ visited,
                          int* __restrict__ nextCount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= frontierSize) return;

    int node = frontier[idx];
    
    int rowStart = d_rowOffsets[node];
    int rowEnd   = d_rowOffsets[node + 1];

    for (int offset = rowStart; offset < rowEnd; offset++) {
        int neighbor = d_colIndices[offset];
        if (atomicCAS(&visited[neighbor], 0, 1) == 0) {
            int pos = atomicAdd(nextCount, 1); // get position for writing
            nextFrontier[pos] = neighbor;
        }
    }
}

int main() {
    int *d_frontier, *d_nextFrontier;
    int *d_visited, *d_nextCount;
    int threadsPerBlock = 256;
    int blocksPerGrid = (frontierSize + threadsPerBlock - 1) / threadsPerBlock;
    bfsKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_rowOffsets, d_colIndices, frontierSize,
        d_frontier, d_nextFrontier, d_visited, d_nextCount
    );
    cudaDeviceSynchronize();


    return 0;
}
