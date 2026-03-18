#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>

int main(){
    int N=1<<20;
    size_t size= N*sizeof(float);
    float alpha= 2.5f;

    float *h_x=(float*)malloc(size);
    float *h_y=(float*)malloc(size);
    for(int i=0; i<N; i++){
        h_x[i]= (float)(rand()%100);
        h_y[i]= (float)(rand()%100);
    }

    float *d_x, *d_y;
    cudaMalloc(&d_x, size);
    cudaMalloc(&d_y, size);
    cudaMemcpy(d_x, h_x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, h_y, size, cudaMemcpyHostToDevice);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasSaxpy(handle, N, &alpha, d_x, 1, d_y, 1);

    cudaMemcpy(h_y, d_y, size, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    printf("After saxpy, h_y[0]=%f\n", h_y[0]);

    cublasDestroy(handle);
    cudaFree(d_x); cudaFree(d_y);
    free(h_x); free(h_y);
    return 0;
}
