#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cufft.h>

int main(){
    int N = 1024;  
    size_t memSize = N * sizeof(cufftComplex);

    cufftComplex *h_in = (cufftComplex*)malloc(memSize);
    cufftComplex *h_out= (cufftComplex*)malloc(memSize);

    for(int i=0; i<N; i++){
        float realVal = (float)(rand()%10);
        float imagVal = 0.0f; 
        h_in[i].x = realVal;
        h_in[i].y = imagVal;
    }

    cufftComplex *d_data;
    cudaMalloc((void**)&d_data, memSize);
    cudaMemcpy(d_data, h_in, memSize, cudaMemcpyHostToDevice);

    cufftHandle plan;
    cufftResult status;
    status = cufftPlan1d(&plan, N, CUFFT_C2C, 1); 
    if(status != CUFFT_SUCCESS){
        fprintf(stderr,"cufftPlan1d failed!\n");
        return -1;
    }

    status = cufftExecC2C(plan, d_data, d_data, CUFFT_FORWARD);
    if(status != CUFFT_SUCCESS){
        fprintf(stderr,"cufftExecC2C forward failed!\n");
        return -1;
    }
    cudaDeviceSynchronize();

    status = cufftExecC2C(plan, d_data, d_data, CUFFT_INVERSE);
    if(status != CUFFT_SUCCESS){
        fprintf(stderr,"cufftExecC2C inverse failed!\n");
        return -1;
    }
    cudaDeviceSynchronize();

    cudaMemcpy(h_out, d_data, memSize, cudaMemcpyDeviceToHost);

    for(int i=0;i<4;i++){
        printf("After inverse, h_out[%d]= (%f, %f)\n", i, h_out[i].x, h_out[i].y);
    }

    cufftDestroy(plan);
    cudaFree(d_data);
    free(h_in); free(h_out);

    return 0;
}
