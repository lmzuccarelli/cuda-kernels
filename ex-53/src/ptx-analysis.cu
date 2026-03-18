#include <cuda_runtime.h>
#include <stdio.h>

// ptxas myKernel.ptx -o myKernel.cubin --gpu-name sm_80 --warn-on-spill

__global__ void ptxTestKernel(const float *in, float *out, int N) {
    int idx= blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < N){
        float val= in[idx];
        #pragma unroll 4   
        for(int i=0; i<8; i++){
            val+= 0.5f;
        }
        out[idx]= val;
    }
}

int main(){
    return 0;
}


