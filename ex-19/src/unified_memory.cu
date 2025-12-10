// unifiedMemoryVectorAdd.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void vectorAdd(float *A, float *B, float *C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1 << 20;  // 1M elements
    size_t size = N * sizeof(float);

    float *A, *B, *C;
    cudaMallocManaged(&A, size);
    cudaMallocManaged(&B, size);
    cudaMallocManaged(&C, size);

    for (int i = 0; i < N; i++) {
        A[i] = i * 1.0f;
        B[i] = i * 2.0f;
    }

    vectorAdd<<<(N + 255) / 256, 256>>>(A, B, C, N);
    cudaDeviceSynchronize();

    printf("C[0] = %f, C[N-1] = %f\n", C[0], C[N - 1]);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}
