#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// NOTE add debugging flags to nvcc
// cd build
// execute cuda-gdb ./intermediate-debugging

__global__ void incrementKernel(int *data, int N) {
    __shared__ int sTemp[256]; 

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int localId = threadIdx.x; 

    if(idx < N){
        sTemp[localId] = data[idx];
        __syncthreads(); 

        sTemp[localId] += 100; 
        __syncthreads();

        data[idx] = sTemp[localId]; 
    }
}

int main(){
    int N = 1024;
    size_t size = N*sizeof(int);
    int *h_data = (int*)malloc(size);

    for(int i=0; i<N; i++){
        h_data[i] = i;
    }

    int *d_data;
    cudaMalloc(&d_data, size);

    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N+block.x-1)/block.x);
    incrementKernel<<<grid, block>>>(d_data, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);

    printf("h_data[0]= %d, h_data[N-1]=%d\n", h_data[0], h_data[N-1]);

    free(h_data);
    cudaFree(d_data);
    return 0;
}
