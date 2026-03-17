#include <cuda_runtime.h>
#include <stdio.h>


// NOTE this fails locally due to the fact I only have 1 gpu

int main(){
    int devCount;
    cudaGetDeviceCount(&devCount);
    if(devCount<2){
        printf("Need at least 2 GPUs for P2P example!\n");
        return 0;
    }

    int canAccess01=0, canAccess10=0;
    cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
    cudaDeviceCanAccessPeer(&canAccess10, 1, 0);

    printf("canAccess01=%d, canAccess10=%d\n", canAccess01, canAccess10);

    if(canAccess01 && canAccess10){
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);

        cudaSetDevice(1);
        cudaDeviceEnablePeerAccess(0, 0);

        cudaSetDevice(0);
        float *d_data0; 
        cudaMalloc(&d_data0, 100*sizeof(float));
        cudaSetDevice(1);
        float *d_data1;
        cudaMalloc(&d_data1, 100*sizeof(float));

        cudaMemcpyPeer(d_data1, 1, d_data0, 0, 100*sizeof(float));

        printf("P2P enabled between dev0 & dev1. \n");
    } else {
        printf("Peer access not available between dev0 & dev1.\n");
    }
    return 0;
}
