#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <stdio.h>

#define IDX2C(i,j,ld) (((j)*(ld))+(i))

#define CUBLAS_CHECK(call)                                   \
    do {                                                     \
        cublasStatus_t err = call;                           \
        if (err != CUBLAS_STATUS_SUCCESS) {                  \
            printf("cuBLAS error %d at line %d\n", err, __LINE__); \
            exit(EXIT_FAILURE);                              \
        }                                                    \
    } while(0)

int main() {
    int M = 64, K = 64, N = 64;
    int batchSize = 1000; // Number of small GEMMs in the batch

    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);

    float* h_A = (float*)malloc(sizeA * batchSize);
    float* h_B = (float*)malloc(sizeB * batchSize);
    float* h_C = (float*)malloc(sizeC * batchSize);

    for (int i = 0; i < batchSize * M * K; i++) h_A[i] = 1.0f;
    for (int i = 0; i < batchSize * K * N; i++) h_B[i] = 2.0f;
    for (int i = 0; i < batchSize * M * N; i++) h_C[i] = 0.0f;

    float **d_Aarray, **d_Barray, **d_Carray;
    cudaMalloc((void**)&d_Aarray, batchSize * sizeof(float*));
    cudaMalloc((void**)&d_Barray, batchSize * sizeof(float*));
    cudaMalloc((void**)&d_Carray, batchSize * sizeof(float*));

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, sizeA * batchSize);
    cudaMalloc((void**)&d_B, sizeB * batchSize);
    cudaMalloc((void**)&d_C, sizeC * batchSize);

    cudaMemcpy(d_A, h_A, sizeA * batchSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, sizeB * batchSize, cudaMemcpyHostToDevice);

    float **h_Aptrs = (float**)malloc(batchSize * sizeof(float*));
    float **h_Bptrs = (float**)malloc(batchSize * sizeof(float*));
    float **h_Cptrs = (float**)malloc(batchSize * sizeof(float*));
    for (int i = 0; i < batchSize; i++) {
        h_Aptrs[i] = d_A + i * M * K;
        h_Bptrs[i] = d_B + i * K * N;
        h_Cptrs[i] = d_C + i * M * N;
    }
    cudaMemcpy(d_Aarray, h_Aptrs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Barray, h_Bptrs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);
    cudaMemcpy(d_Carray, h_Cptrs, batchSize * sizeof(float*), cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    float alpha = 1.0f, beta = 0.0f;
    
    CUBLAS_CHECK(cublasSgemmBatched(handle,
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    M, N, K,
                                    &alpha,
                                    (const float**)d_Aarray, M,
                                    (const float**)d_Barray, K,
                                    &beta,
                                    d_Carray, M,
                                    batchSize));

    cudaMemcpy(h_C, d_C, sizeC * batchSize, cudaMemcpyDeviceToHost);

    printf("Sample h_C[0] = %f\n", h_C[0]);

    free(h_A); free(h_B); free(h_C);
    free(h_Aptrs); free(h_Bptrs); free(h_Cptrs);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_Aarray); cudaFree(d_Barray); cudaFree(d_Carray);
    CUBLAS_CHECK(cublasDestroy(handle));
    return 0;
}
