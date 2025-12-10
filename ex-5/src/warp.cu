__global__ void warpExampleKernel(float *A, float *B, float *C, int N) {
    int globalIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if(globalIdx < N) {
        C[globalIdx] = A[globalIdx] + B[globalIdx];
    }
}

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

__global__ void noDivergenceKernel(float *A, float *B, float *C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N) {
        float a = A[idx];
        float b = B[idx];
        C[idx] = a * (a > 0 ? 1.0f : -1.0f) + b;
    }
}

__global__ void warpSyncKernel(float *A, float *B, float *C, int N) {
    __shared__ float sharedA[32];
    __shared__ float sharedB[32];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warpIdx = threadIdx.x % 32;
    
    if(idx < N) {
        sharedA[warpIdx] = A[idx];
        sharedB[warpIdx] = B[idx];
        __syncthreads();
        
        C[idx] = sharedA[warpIdx] + sharedB[warpIdx];
    }
}
