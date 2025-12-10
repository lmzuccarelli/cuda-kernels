// A warp is a group of 32 threads within a block that execute instructions in lockstep on a Streaming Multiprocessor (SM). 
// All threads in a warp execute the same instruction simultaneously, following the Single Instruction, Multiple Threads (SIMT) model.

__global__ void warpExampleKernel(float *A, float *B, float *C, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    // Each warp executes the following instructions
    if(globalIdx < N) {
        C[globalIdx] = A[globalIdx] + B[globalIdx];
    }
}


// if some threads in a warp take the if branch while others take the else branch, 
// the warp will execute both branches serially, leading to performance degradation.
__global__ void divergenceKernel(int *A, int *B, int *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        if(A[idx] > 0) {
            C[idx] = A[idx] + B[idx];
        } else {
            C[idx] = A[idx] - B[idx];
        }
    }
}

// Mitigation:
//
// Minimize Branching: Design kernels to minimize conditional statements within warps.
// Data Reorganization: Structure data to reduce the likelihood of divergence.
// Predication: Use predicated instructions to handle conditional operations without causing divergence.
__global__ void noDivergenceKernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float a = A[idx];
        float b = B[idx];
        // Use mathematical operations to avoid branching
        C[idx] = a * (a > 0 ? 1.0f : -1.0f) + b;
    }
}

// Shared Memory: Allocates shared memory for storing intermediate data.
// Synchronization: __syncthreads() ensures all threads have loaded their data into shared memory before performing the addition.
// Warp-Specific Operations: Operates within a warp by using warpIdx.
__global__ void warpSyncKernel(float *A, float *B, float *C, int N) {
    __shared__ float sharedA[32];
    __shared__ float sharedB[32];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x % 32;
    
    if(idx < N) {
        sharedA[warpIdx] = A[idx];
        sharedB[warpIdx] = B[idx];
        __syncthreads(); // Synchronize within the block
        
        C[idx] = sharedA[warpIdx] + sharedB[warpIdx];
    }
}
