#include <stdio.h>
#include <stdlib.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

int main(){
    int N=512;  // NxN
    float alpha=1.0f, beta=0.0f;

    size_t size= N*N*sizeof(float);
    float *h_A=(float*)malloc(size);
    float *h_B=(float*)malloc(size);
    float *h_C=(float*)malloc(size);

    for(int i=0; i<N*N; i++){
        h_A[i]= (float)(rand()%5);
        h_B[i]= (float)(rand()%5);
        h_C[i]= 0.0f;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, size);
    cudaMalloc((void**)&d_B, size);
    cudaMalloc((void**)&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, h_C, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    int lda=N, ldb=N, ldc=N; // leading dimensions
    cublasOperation_t transA= CUBLAS_OP_N;  // no transpose
    cublasOperation_t transB= CUBLAS_OP_N;

    cublasSgemm(handle,
                transA, transB,
                N,   // M
                N,   // N
                N,   // K
                &alpha,
                d_A, lda,
                d_B, ldb,
                &beta,
                d_C, ldc);

    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    printf("C[0]=%f, C[end]=%f\n", h_C[0], h_C[N*N-1]);

    cublasDestroy(handle);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
