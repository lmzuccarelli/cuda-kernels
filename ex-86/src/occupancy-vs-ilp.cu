
// standard version

__global__ void standardKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += data[i] * 0.5f;
    }
    data[idx] = sum;
}

// unrolled version
__global__ void unrolledKernel(float* data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float sum = 0.0f;
    // Unroll loop by factor of 4
    for (int i = 0; i < N; i += 4) {
        sum += data[i] * 0.5f;
        sum += data[i+1] * 0.5f;
        sum += data[i+2] * 0.5f;
        sum += data[i+3] * 0.5f;
    }
    data[idx] = sum;
}
