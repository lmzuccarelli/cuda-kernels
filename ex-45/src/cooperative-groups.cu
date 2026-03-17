#include <cooperative_groups.h>
#include <cuda_runtime.h>
#include <stdio.h>
namespace cg = cooperative_groups;

__global__ void tileReduceKernel(const float *input, float *output, int N) {
    cg::thread_block block = cg::this_thread_block();
    auto tile16 = cg::tiled_partition<16>(block);

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx<N) ? input[idx] : 0.0f;

    for(int offset=tile16.size()/2; offset>0; offset>>=1){
        val += tile16.shfl_down(val, offset);
    }

    __shared__ float sdata[256]; 
    int tid = threadIdx.x;
    if(tile16.thread_rank() ==0){
        sdata[tid] = val; 
    }
    block.sync();

    if(tid<16){
        float sum = sdata[tid];
        output[blockIdx.x] = sum; 
    }
}

int main(){
    int N=256;
    size_t size= N*sizeof(float);
    float *h_in=(float*)malloc(size);
    for(int i=0;i<N;i++){
        h_in[i]=(float)(rand()%100);
    }
    float *d_in,*d_out;
    cudaMalloc(&d_in,size);
    cudaMalloc(&d_out, (N/64)*sizeof(float)); 
    cudaMemcpy(d_in,h_in,size,cudaMemcpyHostToDevice);

    dim3 block(64);
    dim3 grid((N+block.x-1)/block.x);

    tileReduceKernel<<<grid, block>>>(d_in,d_out,N);
    cudaDeviceSynchronize();

    free(h_in);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
