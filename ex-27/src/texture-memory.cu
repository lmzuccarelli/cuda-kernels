#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void textureVsGlobalKernel(const float *globalData, float *outputTex,
                                      float *outputGlobal, int width,
                                      int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float u = x + 0.5f;
    float v = y + 0.5f;

    float texVal = tex2D(texRef, u, v);

    int idx = y * width + x;
    float globalVal = globalData[idx];

    outputTex[idx] = texVal;
    outputGlobal[idx] = globalVal;
  }
}

int main() {
  int width = 512, height = 512;
  size_t size = width * height * sizeof(float);

  float *h_image = (float *)malloc(size);
  float *h_outputTex = (float *)malloc(size);
  float *h_outputGlobal = (float *)malloc(size);
  if (!h_image || !h_outputTex || !h_outputGlobal) {
    printf("Host memory allocation failed.\n");
    exit(EXIT_FAILURE);
  }

  srand(time(NULL));
  for (int i = 0; i < width * height; i++) {
    h_image[i] = (float)(rand() % 256) / 255.0f;
  }

  // Allocate CUDA array for the 2D texture.
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray_t cuArray;
  cudaMallocArray(&cuArray, &channelDesc, width, height);

  // Copy image data from host to CUDA array.
  cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float),
                      width * sizeof(float), height, cudaMemcpyHostToDevice);

  // Set texture parameters (address mode, filter mode, etc.).
  texRef.addressMode[0] = cudaAddressModeClamp; // Clamp coordinates
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.filterMode = cudaFilterModePoint; // No filtering
  texRef.normalized = false; // Use unnormalized texture coordinates

  // Bind the CUDA array to the texture reference.
  cudaBindTextureToArray(texRef, cuArray, channelDesc);

  // Allocate device memory for global data and output arrays.
  float *d_image, *d_outputTex, *d_outputGlobal;
  cudaMalloc((void **)&d_image, size);
  cudaMalloc((void **)&d_outputTex, size);
  cudaMalloc((void **)&d_outputGlobal, size);

  // Copy the same image data to device global memory.
  cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);

  // Define kernel launch parameters.
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the kernel.
  textureVsGlobalKernel<<<blocksPerGrid, threadsPerBlock>>>(
      d_image, d_outputTex, d_outputGlobal, width, height);
  cudaDeviceSynchronize();

  // Copy results from device to host.
  cudaMemcpy(h_outputTex, d_outputTex, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_outputGlobal, d_outputGlobal, size, cudaMemcpyDeviceToHost);

  // Compare outputs (for demonstration, we print the first 10 elements).
  printf("First 10 values from texture fetch:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_outputTex[i]);
  }
  printf("\n");

  printf("First 10 values from global memory fetch:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_outputGlobal[i]);
  }
  printf("\n");

  // Unbind the texture.
  cudaUnbindTexture(texRef);

  // Free device memory and CUDA array.
  cudaFree(d_image);
  cudaFree(d_outputTex);
  cudaFree(d_outputGlobal);
  cudaFreeArray(cuArray);

  // Free host memory.
  free(h_image);
  free(h_outputTex);
  free(h_outputGlobal);

  return 0;
}
