#include <cuda_runtime.h>
#include <stdio.h>

__inline__ __device__ float warpPrefixSum(float val) {
    unsigned mask = 0xffffffff; // assume full warp active
    for(int offset = 1; offset < 32; offset <<= 1) {
        float n = __shfl_up_sync(mask, val, offset, 32);
        int laneId = threadIdx.x & 31; 
        if(laneId >= offset) {
            val += n;
        }
    }
    return val; 
}

__global__ void warpScanKernel(const float *input, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx < N){
        float val = input[idx];
        float prefix = warpPrefixSum(val);
        output[idx] = prefix;
    }
}

int main(){
    int N=64; 
    size_t size=N*sizeof(float);
    float *h_in=(float*)malloc(size);
    float *h_out=(float*)malloc(size);
    for(int i=0;i<N;i++){
        h_in[i]=1.0f; 
    }

    float *d_in, *d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out,size);
    cudaMemcpy(d_in,h_in,size,cudaMemcpyHostToDevice);

    dim3 block(32);
    dim3 grid((N+31)/32);
    warpScanKernel<<<grid,block>>>(d_in,d_out,N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_out,d_out,size,cudaMemcpyDeviceToHost);
    for(int i=0;i<32;i++){
        printf("h_out[%d]=%f ", i, h_out[i]);
    }
    printf("\n");

    free(h_in);free(h_out);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
