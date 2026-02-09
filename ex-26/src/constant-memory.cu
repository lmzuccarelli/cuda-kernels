#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__constant__ float scaleFactor;

__global__ void vectorScaleKernel(const float *input, float *output, int N) {
  int idx = threadIdx.x + blockIdx.x * blockDim.x;

  // Check bounds to prevent out-of-bounds access.
  if (idx < N) {
    // Multiply each element by the constant scaling factor.
    output[idx] = input[idx] * scaleFactor;
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
  int N = 1 << 20;
  size_t size = N * sizeof(float);

  float *h_input = (float *)malloc(size);
  float *h_output = (float *)malloc(size);
  if (!h_input || !h_output) {
    printf("Host memory allocation failed\n");
    exit(EXIT_FAILURE);
  }

  srand(time(NULL));
  for (int i = 0; i < N; i++) {
    h_input[i] = (float)(rand() % 100) / 10.0f;
  }

  float *d_input, *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_input, size));
  CUDA_CHECK(cudaMalloc((void **)&d_output, size));

  CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));

  float h_scale = 2.5f; // Example scale factor.
  CUDA_CHECK(cudaMemcpyToSymbol(scaleFactor, &h_scale, sizeof(float)));

  int threadsPerBlock = 256;
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  CUDA_CHECK(cudaEventRecord(start, 0));

  vectorScaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, N);

  CUDA_CHECK(cudaEventRecord(stop, 0));
  CUDA_CHECK(cudaEventSynchronize(stop));

  float elapsedTime = 0;
  CUDA_CHECK(cudaEventElapsedTime(&elapsedTime, start, stop));
  printf("Kernel Execution Time: %f ms\n", elapsedTime);

  CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));

  printf("First 10 elements of the scaled vector:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_output[i]);
  }
  printf("\n");

  CUDA_CHECK(cudaFree(d_input));
  CUDA_CHECK(cudaFree(d_output));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
  free(h_input);
  free(h_output);

  return 0;
}
