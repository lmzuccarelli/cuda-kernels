#include <stdio.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void monteCarloPiKernel(unsigned long long *counts, int totalPoints, unsigned long long seed) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx< totalPoints){
        curandState_t state;
        curand_init(seed, idx, 0, &state); 
        float x = curand_uniform(&state);
        float y = curand_uniform(&state);

        unsigned long long localCount=0ULL;
        if(x*x + y*y <= 1.0f){
            localCount=1ULL;
        }

        counts[idx] = localCount;
    }
}

int main(){
    int nPoints=1<<20;
    dim3 block(256);
    dim3 grid( (nPoints+block.x-1)/block.x );

    size_t size = nPoints*sizeof(unsigned long long);
    unsigned long long *d_counts;
    cudaMalloc(&d_counts, size);

    monteCarloPiKernel<<<grid,block>>>(d_counts,nPoints,1234ULL);
    cudaDeviceSynchronize();

    unsigned long long *h_counts=(unsigned long long*)malloc(size);
    cudaMemcpy(h_counts, d_counts, size, cudaMemcpyDeviceToHost);

    unsigned long long sum=0ULL;
    for(int i=0;i<nPoints;i++){
        sum += h_counts[i];
    }

    double piEst= (4.0*(double)sum)/(double)nPoints;
    printf("Estimated Pi= %f\n", piEst);

    cudaFree(d_counts);
    free(h_counts);
    return 0;
}
