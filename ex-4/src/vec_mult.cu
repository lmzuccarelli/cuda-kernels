#include <cuda_runtime.h>
#include <stdio.h>
#include <ctime>

#define TILE_SIZE 16

__global__ void matrixMulCUDA(float *C, float *A, float *B, int Width) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float value = 0.0f;
    
    for(int m = 0; m < Width / TILE_SIZE; ++m) {
        tileA[threadIdx.y][threadIdx.x] = A[row * Width + m * TILE_SIZE + threadIdx.x];
        tileB[threadIdx.y][threadIdx.x] = B[(m * TILE_SIZE + threadIdx.y) * Width + col];
        __syncthreads();
        
        for(int k = 0; k < TILE_SIZE; ++k) {
            value += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    C[row * Width + col] = value;
}

int main() {
    int Width = 1024;
    size_t size = Width * Width * sizeof(float);
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    h_A = (float*)malloc(size);
    h_B = (float*)malloc(size);
    h_C = (float*)malloc(size);

    for(int i = 0; i < Width * Width; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 1.0f;
    }

    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);
    cudaMalloc(&d_C, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 blocksPerGrid((Width + TILE_SIZE - 1) / TILE_SIZE, (Width + TILE_SIZE - 1) / TILE_SIZE);
    printf("blocks per grid: %d \n",((Width + TILE_SIZE -1) / TILE_SIZE));

    clock_t start = clock();
    matrixMulCUDA<<<blocksPerGrid, threadsPerBlock>>>(d_C, d_A, d_B, Width);
    clock_t end = clock();
    printf("GPU vector multiply time: %lf seconds\n", ((double)(end - start))/CLOCKS_PER_SEC);


    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for(int i = 0; i < Width * Width; i++) {
        if(h_C[i] != Width) { // Since each element is 1 * Width
            success = false;
            printf("error at index %d: %f != %d\n", i, h_C[i], Width);
            break;
        }
    }

    if(success) {
        printf("matrix multiplication successful!\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
