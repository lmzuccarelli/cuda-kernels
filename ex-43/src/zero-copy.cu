#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void zeroCopyKernel(const float *arr, float *out, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        out[idx] = arr[idx] * 2.0f;
    }
}

int main() {
    int N=1<<20;
    size_t size = N*sizeof(float);

    float *h_in=NULL, *h_out=NULL;
    cudaHostAlloc((void**)&h_in, size, cudaHostAllocMapped);
    cudaHostAlloc((void**)&h_out, size, cudaHostAllocMapped);

    for(int i=0; i<N; i++){
        h_in[i] = (float)(rand()%100);
        h_out[i] = 0.0f;
    }

    float *d_in, *d_out;
    cudaHostGetDevicePointer((void**)&d_in, (void*)h_in, 0);
    cudaHostGetDevicePointer((void**)&d_out, (void*)h_out, 0);

    int threads=256;
    int blocks=(N+threads-1)/threads;
    zeroCopyKernel<<<blocks, threads>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    printf("h_out[0]= %f\n", h_out[0]);

    cudaFreeHost(h_in);
    cudaFreeHost(h_out);
    return 0;
}
