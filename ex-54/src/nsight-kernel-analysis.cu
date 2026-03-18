#include <cuda_runtime.h>
#include <stdio.h>

// use nsights
// ncu --target-processes all ./nsight-kernel-analysis

__global__ void sumArrays(const float *A, const float *B, float *C, int N){
    int idx= blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N){
        float valA = A[idx];
        float valB = B[idx];
        #pragma unroll
        for(int i=0;i<4;i++){
            valA *= 1.0001f; 
            valB += 0.5f;
        }
        C[idx] = valA + valB;
    }
}

int main(){
    int N=1<<20;
    size_t size= N*sizeof(float);
    float *h_A=(float*)malloc(size);
    float *h_B=(float*)malloc(size);
    float *h_C=(float*)malloc(size);

    for(int i=0;i<N;i++){
        h_A[i]= (float)(rand()%100);
        h_B[i]= (float)(rand()%100);
    }

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);

    cudaMemcpy(d_A, h_A, size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size,cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid( (N+block.x-1)/block.x );
    sumArrays<<<grid, block>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
    printf("h_C[0]=%f\n", h_C[0]);

    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    free(h_A); free(h_B); free(h_C);
    return 0;
}
