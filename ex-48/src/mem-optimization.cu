#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 16 

__global__ void matMulKernel(const float *A, const float *B, float *C, int N) {
    __shared__ float sA[TILE_DIM][TILE_DIM];  // reduce tile size => reduce shared memory
    __shared__ float sB[TILE_DIM][TILE_DIM];

    int row= blockIdx.y*TILE_DIM + threadIdx.y;
    int col= blockIdx.x*TILE_DIM + threadIdx.x;
    float val=0.0f;

    for(int k=0; k<N/TILE_DIM; k++){
        int aRow= row, aCol= k*TILE_DIM + threadIdx.x;
        if(aRow<N && aCol<N)
            sA[threadIdx.y][threadIdx.x] = A[aRow*N + aCol];
        else
            sA[threadIdx.y][threadIdx.x]= 0.0f;

        int bRow= k*TILE_DIM + threadIdx.y, bCol= col;
        if(bRow<N && bCol<N)
            sB[threadIdx.y][threadIdx.x] = B[bRow*N + bCol];
        else
            sB[threadIdx.y][threadIdx.x]=0.0f;
        __syncthreads();

        for(int n=0; n<TILE_DIM; n++){
            val += sA[threadIdx.y][n]* sB[n][threadIdx.x];
        }
        __syncthreads();
    }

    if(row<N && col<N){
        C[row*N + col]= val;
    }
}

int main(){
    int N=1024; 
    size_t size= N*N*sizeof(float);
    float *h_A=(float*)malloc(size);
    float *h_B=(float*)malloc(size);
    float *h_C=(float*)malloc(size);

    for(int i=0;i<N*N;i++){
        h_A[i]= (float)(rand()%5);
        h_B[i]= (float)(rand()%5);
    }

    float *d_A,*d_B,*d_C;
    cudaMalloc(&d_A,size);
    cudaMalloc(&d_B,size);
    cudaMalloc(&d_C,size);
    cudaMemcpy(d_A,h_A,size,cudaMemcpyHostToDevice);
    cudaMemcpy(d_B,h_B,size,cudaMemcpyHostToDevice);

    dim3 block(TILE_DIM,TILE_DIM);
    dim3 grid((N+TILE_DIM-1)/TILE_DIM, (N+TILE_DIM-1)/TILE_DIM);

    matMulKernel<<<grid, block>>>(d_A,d_B,d_C,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_C,d_C,size,cudaMemcpyDeviceToHost);
    printf("C[0]=%f, C[end]=%f\n",h_C[0],h_C[N*N-1]);

    free(h_A);free(h_B);free(h_C);
    cudaFree(d_A); cudaFree(d_B);cudaFree(d_C);
    return 0;
}
