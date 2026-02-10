#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

__global__ void invertImageKernel(cudaSurfaceObject_t surfObj, int width,
                                  int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;

  if (x < width && y < height) {
    float pixel;
    surf2Dread(&pixel, surfObj, x * sizeof(float), y);

    float inverted = 1.0f - pixel;

    surf2Dwrite(inverted, surfObj, x * sizeof(float), y);
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

  // Allocate host memory for input image.
  float *h_image = (float *)malloc(imgSize);
  float *h_output = (float *)malloc(imgSize);
  if (!h_image || !h_output) {
    printf("Host memory allocation failed.\n");
    exit(EXIT_FAILURE);
  }

  // Initialize the image with random grayscale values (0 to 1).
  srand(time(NULL));
  for (int i = 0; i < width * height; i++) {
    h_image[i] = (float)(rand() % 256) / 255.0f;
  }

  // Allocate a CUDA array for surface memory.
  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray_t cuArray;
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height,
                             cudaArraySurfaceLoadStore));

  // Copy the image data from host to the CUDA array using a 2D copy.
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float),
                                 width * sizeof(float), height,
                                 cudaMemcpyHostToDevice));

  // Create a surface object for output.
  cudaResourceDesc resDesc = {};
  resDesc.resType = cudaResourceTypeArray;
  resDesc.res.array.array = cuArray;
  cudaSurfaceObject_t surfObj = 0;
  CUDA_CHECK(cudaCreateSurfaceObject(&surfObj, &resDesc));

  // Launch the kernel to invert the image.
  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                     (height + threadsPerBlock.y - 1) / threadsPerBlock.y);
  invertImageKernel<<<blocksPerGrid, threadsPerBlock>>>(surfObj, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  // Copy the processed image from the CUDA array back to a device buffer.
  // Allocate device memory for output copy.
  float *d_output;
  CUDA_CHECK(cudaMalloc((void **)&d_output, imgSize));
  CUDA_CHECK(cudaMemcpy2DFromArray(d_output, width * sizeof(float), cuArray, 0,
                                   0, width * sizeof(float), height,
                                   cudaMemcpyDeviceToDevice));

  // Copy the output data from device to host.
  CUDA_CHECK(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost));

  // Print the first 10 pixel values for verification.
  printf("First 10 pixel values of the inverted image:\n");
  for (int i = 0; i < 10; i++) {
    printf("%f ", h_output[i]);
  }
  printf("\n");

  // Cleanup: Unbind texture if applicable (not needed for surfaces), destroy
  // surface object, free device memory and host memory.
  CUDA_CHECK(cudaDestroySurfaceObject(surfObj));
  CUDA_CHECK(cudaFreeArray(cuArray));
  CUDA_CHECK(cudaFree(d_output));
  free(h_image);
  free(h_output);

  return 0;
}
