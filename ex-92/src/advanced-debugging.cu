// faulty kernel
__global__ void faultyKernel(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Divergent branch: Only threads with even index execute some work and sync.
    if (idx % 2 == 0) {
        // Perform some computation
        data[idx] += 10;
        __syncthreads();  // Only executed by threads where idx % 2 == 0
    } else {
        // Threads with odd index skip __syncthreads()
        data[idx] -= 10;
    }
}


__global__ void fixedKernel(int* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Compute condition, but ensure all threads reach __syncthreads()
    bool condition = (idx % 2 == 0);
    if (condition) {
        data[idx] += 10;
    } else {
        data[idx] -= 10;
    }
    // All threads synchronize regardless of the branch taken
    __syncthreads();
}
