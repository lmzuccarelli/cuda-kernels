#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    float *d_data0, *d_data1;
    
    cudaSetDevice(0);
    cudaMalloc(&d_data0, size);
    
    cudaSetDevice(1);
    cudaMalloc(&d_data1, size);
    
    int canAccessPeer;
    cudaDeviceCanAccessPeer(&canAccessPeer, 0, 1);
    if (canAccessPeer) {
        cudaSetDevice(0);
        cudaDeviceEnablePeerAccess(1, 0);
        
        cudaMemcpyPeerAsync(d_data1, 1, d_data0, 0, size);
        cudaDeviceSynchronize();
        printf("P2P transfer completed successfully.\n");
    } else {
        printf("GPUDirect RDMA is not supported between GPU 0 and GPU 1.\n");
    }
    
    cudaSetDevice(0);
    cudaFree(d_data0);
    cudaSetDevice(1);
    cudaFree(d_data1);
    
    return 0;
}
