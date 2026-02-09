#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void vectorAddKernel(const float *A, const float *B, float *C,
                                int chunkSize) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < chunkSize) {
    C[idx] = A[idx] + B[idx];
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
  int totalElements = 1 << 22;
  int chunkSize = 1 << 20;
  size_t chunkBytes = chunkSize * sizeof(float);
  size_t totalBytes = totalElements * sizeof(float);

  float *h_A, *h_B, *h_C;
  CUDA_CHECK(cudaMallocHost((void **)&h_A, totalBytes));
  CUDA_CHECK(cudaMallocHost((void **)&h_B, totalBytes));
  CUDA_CHECK(cudaMallocHost((void **)&h_C, totalBytes));

  srand(time(NULL));
  for (int i = 0; i < totalElements; i++) {
    h_A[i] = (float)(rand() % 100) / 10.0f;
    h_B[i] = (float)(rand() % 100) / 10.0f;
  }

  float *d_A0, *d_B0, *d_C0;
  float *d_A1, *d_B1, *d_C1;
  CUDA_CHECK(cudaMalloc((void **)&d_A0, chunkBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_B0, chunkBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_C0, chunkBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_A1, chunkBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_B1, chunkBytes));
  CUDA_CHECK(cudaMalloc((void **)&d_C1, chunkBytes));

  cudaStream_t stream0, stream1;
  CUDA_CHECK(cudaStreamCreate(&stream0));
  CUDA_CHECK(cudaStreamCreate(&stream1));

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  int numChunks = totalElements / chunkSize;
  if (totalElements % chunkSize != 0)
    numChunks++;

  int threadsPerBlock = 256;
  int blocksPerGrid = (chunkSize + threadsPerBlock - 1) / threadsPerBlock;

  CUDA_CHECK(cudaEventRecord(start, 0));

  for (int chunk = 0; chunk < numChunks; chunk++) {
    int offset = chunk * chunkSize;
    int currentChunkSize = ((offset + chunkSize) <= totalElements)
                               ? chunkSize
                               : (totalElements - offset);
    size_t currentChunkBytes = currentChunkSize * sizeof(float);

    float *d_A = (chunk % 2 == 0) ? d_A0 : d_A1;
    float *d_B = (chunk % 2 == 0) ? d_B0 : d_B1;
    float *d_C = (chunk % 2 == 0) ? d_C0 : d_C1;
    cudaStream_t stream = (chunk % 2 == 0) ? stream0 : stream1;

    CUDA_CHECK(cudaMemcpyAsync(d_A, h_A + offset, currentChunkBytes,
                               cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_B, h_B + offset, currentChunkBytes,
                               cudaMemcpyHostToDevice, stream));

    vectorAddKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_A, d_B, d_C, currentChunkSize);

    CUDA_CHECK(cudaMemcpyAsync(h_C + offset, d_C, currentChunkBytes,
                               cudaMemcpyDeviceToHost, stream));
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedTime = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Total Pipeline Execution Time: %f ms\n", elapsedTime);

  printf("First 10 elements of result vector:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_C[i]);
  }
  printf("\n");

  CUDA_CHECK(cudaFree(d_A0));
  CUDA_CHECK(cudaFree(d_B0));
  CUDA_CHECK(cudaFree(d_C0));
  CUDA_CHECK(cudaFree(d_A1));
  CUDA_CHECK(cudaFree(d_B1));
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
