#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// good data locality
__global__ void vectorAddSequential(const float *A, const float *B, float *C,
                                    int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    C[idx] = A[idx] + B[idx];
  }
}

// poor data locality
__global__ void vectorAddRandom(const float *A, const float *B, float *C, int N,
                                int stride) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx < N) {
    int randIdx = (idx * stride) % N;
    C[randIdx] = A[randIdx] + B[randIdx];
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
  int N = 1 << 20; // 1M elements.
  size_t size = N * sizeof(float);

  // Allocate unified memory.
  float *A, *B, *C_seq, *C_rand;
  CUDA_CHECK(cudaMallocManaged(&A, size));
  CUDA_CHECK(cudaMallocManaged(&B, size));
  CUDA_CHECK(cudaMallocManaged(&C_seq, size));
  CUDA_CHECK(cudaMallocManaged(&C_rand, size));

  // Initialize unified memory arrays.
  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    A[i] = (float)(rand() % 100) / 10.0f;
    B[i] = (float)(rand() % 100) / 10.0f;
  }

  int device = 0;
  CUDA_CHECK(cudaGetDevice(&device));
  cudaMemLocation deviceLoc;
  deviceLoc.type = cudaMemLocationTypeDevice;
  deviceLoc.id = device;

  CUDA_CHECK(cudaMemPrefetchAsync(A, size, deviceLoc, device, NULL));
  CUDA_CHECK(cudaMemPrefetchAsync(B, size, deviceLoc, device, NULL));
  CUDA_CHECK(cudaMemPrefetchAsync(C_seq, size, deviceLoc, device, NULL));
  CUDA_CHECK(cudaMemPrefetchAsync(C_rand, size, deviceLoc, device, NULL));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t startSeq, stopSeq;
  CUDA_CHECK(cudaEventCreate(&startSeq));
  CUDA_CHECK(cudaEventCreate(&stopSeq));

  CUDA_CHECK(cudaEventRecord(startSeq, 0));
  vectorAddSequential<<<blocksPerGrid, threadsPerBlock>>>(A, B, C_seq, N);
  CUDA_CHECK(cudaEventRecord(stopSeq, 0));
  CUDA_CHECK(cudaEventSynchronize(stopSeq));

  float timeSeq = 0;
  CUDA_CHECK(cudaEventElapsedTime(&timeSeq, startSeq, stopSeq));
  printf("Sequential Kernel Execution Time: %f ms\n", timeSeq);

  cudaEvent_t startRand, stopRand;
  CUDA_CHECK(cudaEventCreate(&startRand));
  CUDA_CHECK(cudaEventCreate(&stopRand));

  int stride = 103;

  CUDA_CHECK(cudaEventRecord(startRand, 0));
  vectorAddRandom<<<blocksPerGrid, threadsPerBlock>>>(A, B, C_rand, N, stride);
  CUDA_CHECK(cudaEventRecord(stopRand, 0));
  CUDA_CHECK(cudaEventSynchronize(stopRand));

  float timeRand = 0;
  CUDA_CHECK(cudaEventElapsedTime(&timeRand, startRand, stopRand));
  printf("Random Kernel Execution Time: %f ms\n", timeRand);

  cudaMemLocation hostLoc;
  deviceLoc.type = cudaMemLocationTypeHost;
  deviceLoc.id = cudaCpuDeviceId;

  CUDA_CHECK(cudaMemPrefetchAsync(C_seq, size, hostLoc, cudaCpuDeviceId, NULL));
  CUDA_CHECK(
      cudaMemPrefetchAsync(C_rand, size, hostLoc, cudaCpuDeviceId, NULL));
  CUDA_CHECK(cudaDeviceSynchronize());

  // (Optional) Verify a few results.
  printf("First 10 results (Sequential):\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", C_seq[i]);
  }
  printf("\nFirst 10 results (Random):\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", C_rand[i]);
  }
  printf("\n");

  // Cleanup: Free unified memory and destroy events.
  CUDA_CHECK(cudaFree(A));
  CUDA_CHECK(cudaFree(B));
  CUDA_CHECK(cudaFree(C_seq));
  CUDA_CHECK(cudaFree(C_rand));
  CUDA_CHECK(cudaEventDestroy(startSeq));
  CUDA_CHECK(cudaEventDestroy(stopSeq));
  CUDA_CHECK(cudaEventDestroy(startRand));
  CUDA_CHECK(cudaEventDestroy(stopRand));

  return 0;
}
