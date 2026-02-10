#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void matrixMulKernel(const float *A, const float *B, float *C, int M,
                                int N, int K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < M && col < K) {
    float sum = 0.0f;
    for (int i = 0; i < N; i++)
      sum += A[row * N + i] * B[i * K + col];
    C[row * K + col] = sum;
  }
}

#define CUDA_CHECK(call)                                                       \
  {                                                                            \
    cudaError_t err = call;                                                    \
    if (err != cudaSuccess) {                                                  \
      printf("CUDA Error at %s:%d - %s\n", __FILE__, __LINE__,                 \
             cudaGetErrorString(err));                                         \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

int main() {
  int M = 512, N = 512, K = 512;
  size_t sizeA = M * N * sizeof(float), sizeB = N * K * sizeof(float),
         sizeC = M * K * sizeof(float);

  float *h_A, *h_B, *h_C;
  CUDA_CHECK(cudaMallocHost(&h_A, sizeA));
  CUDA_CHECK(cudaMallocHost(&h_B, sizeB));
  CUDA_CHECK(cudaMallocHost(&h_C, sizeC));

  srand(time(NULL));
  for (int i = 0; i < M * N; i++)
    h_A[i] = (float)(rand() % 100) / 10.0f;
  for (int i = 0; i < N * K; i++)
    h_B[i] = (float)(rand() % 100) / 10.0f;

  int chunkRows = 128, numChunks = (M + chunkRows - 1) / chunkRows;
  float *d_A0, *d_A1, *d_B, *d_C0, *d_C1;
  CUDA_CHECK(cudaMalloc(&d_B, sizeB));
  CUDA_CHECK(cudaMalloc(&d_A0, chunkRows * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_A1, chunkRows * N * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C0, chunkRows * K * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_C1, chunkRows * K * sizeof(float)));

  cudaStream_t stream0, stream1;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  CUDA_CHECK(cudaStreamCreate(&stream1));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));
  CUDA_CHECK(cudaEventRecord(start, 0));

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((K + 15) / 16, (chunkRows + 15) / 16);

  for (int chunk = 0; chunk < numChunks; chunk++) {
    int rowOffset = chunk * chunkRows;
    int currentChunkRows =
        (rowOffset + chunkRows <= M) ? chunkRows : (M - rowOffset);
    size_t chunkSizeA = currentChunkRows * N * sizeof(float);
    size_t chunkSizeC = currentChunkRows * K * sizeof(float);

    float *d_A = (chunk % 2 == 0) ? d_A0 : d_A1;
    float *d_C = (chunk % 2 == 0) ? d_C0 : d_C1;
    cudaStream_t stream = (chunk % 2 == 0) ? stream0 : stream1;

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A + rowOffset * N, chunkSizeA,
                               cudaMemcpyHostToDevice, stream));
    matrixMulKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_A, d_B, d_C, currentChunkRows, N, K);
    CUDA_CHECK(cudaMemcpyAsync(h_C + rowOffset * K, d_C, chunkSizeC,
                               cudaMemcpyDeviceToHost, stream));
  }

  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float totalTime;
  CUDA_CHECK(cudaEventElapsedTime(&totalTime, start, stop));
  printf("Execution Time: %f ms\n", totalTime);

  printf("First 10 elements:\n");
  for (int i = 0; i < 10; i++)
    printf("%f ", h_C[i]);
  printf("\n");

  CUDA_CHECK(cudaFree(d_B));
  CUDA_CHECK(cudaFree(d_A0));
  CUDA_CHECK(cudaFree(d_A1));
  CUDA_CHECK(cudaFree(d_C0));
  CUDA_CHECK(cudaFree(d_C1));
  CUDA_CHECK(cudaStreamDestroy(stream0));
  CUDA_CHECK(cudaStreamDestroy(stream1));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  CUDA_CHECK(cudaFreeHost(h_A));
  CUDA_CHECK(cudaFreeHost(h_B));
  CUDA_CHECK(cudaFreeHost(h_C));

  return 0;
}
