#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Declare a texture reference for 2D texture sampling.
// We use 'cudaTextureType2D' with 'cudaReadModeElementType' to fetch float
// elements.
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

// Kernel for grayscale conversion using texture memory.
// Each thread reads a pixel from texture memory and writes it to the output
// array.
__global__ void grayscaleKernel(float *output, int width, int height) {
  // Calculate pixel coordinates (x, y).
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  // Ensure coordinates are within the image boundaries.
  if (x < width && y < height) {
    // When using unnormalized coordinates, tex2D expects pixel indices plus an
    // offset of 0.5.
    float pixel = tex2D(texRef, x + 0.5f, y + 0.5f);
    // For a grayscale image, simply copy the pixel value.
    output[y * width + x] = pixel;
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
  // Image dimensions.
  int width = 1024, height = 768;
  size_t imgSize = width * height * sizeof(float);

  // Allocate host memory for the image.
  // For demonstration, we simulate a grayscale image.
  float *h_image = (float *)malloc(imgSize);
  if (!h_image) {
    printf("Failed to allocate host memory for image.\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the image with random grayscale values.
  srand(time(NULL));
  for (int i = 0; i < width * height; i++) {
    h_image[i] = (float)(rand() % 256) / 255.0f;
  }

  // Allocate host memory for output image.
  float *h_output = (float *)malloc(imgSize);
  if (!h_output) {
    printf("Failed to allocate host memory for output.\n");
    free(h_image);
    exit(EXIT_FAILURE);
  }

  // Allocate a CUDA array for the image.
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray_t cuArray;
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));

  // Copy image data from host to the CUDA array.
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float),
                                 width * sizeof(float), height,
                                 cudaMemcpyHostToDevice));

  // Set texture parameters.
  texRef.addressMode[0] = cudaAddressModeClamp; // Clamp x coordinates.
  texRef.addressMode[1] = cudaAddressModeClamp; // Clamp y coordinates.
  texRef.filterMode =
      cudaFilterModePoint;   // Use point sampling (no interpolation).
  texRef.normalized = false; // Use unnormalized coordinates.

  // Bind the CUDA array to the texture reference.
  CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

  // Allocate device memory for the output image.
  float *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_output, imgSize));

  // Define kernel launch parameters.
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

  // Launch the grayscale conversion kernel.
  grayscaleKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy the output image from device to host.
  CUDA_CHECK(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost));

  // Verify output: Print the first 10 pixel values.
  printf("First 10 pixel values of the processed image:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_output[i]);
  }
  printf("\n");

  // Cleanup: Unbind texture and free all resources.
  CUDA_CHECK(cudaUnbindTexture(texRef));
  CUDA_CHECK(cudaFreeArray(cuArray));
  CUDA_CHECK(cudaFree(d_output));
  free(h_image);
  free(h_output);

  return 0;
}
