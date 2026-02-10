
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <time.h>

texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;

__global__ void sobelEdgeKernel(float *output, int width, int height) {
  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x >= 1 && x < width - 1 && y >= 1 && y < height - 1) {
    float tl = tex2D(texRef, x - 1 + 0.5f, y - 1 + 0.5f);
    float t = tex2D(texRef, x + 0.5f, y - 1 + 0.5f);
    float tr = tex2D(texRef, x + 1 + 0.5f, y - 1 + 0.5f);
    float l = tex2D(texRef, x - 1 + 0.5f, y + 0.5f);
    float r = tex2D(texRef, x + 1 + 0.5f, y + 0.5f);
    float bl = tex2D(texRef, x - 1 + 0.5f, y + 1 + 0.5f);
    float b = tex2D(texRef, x + 0.5f, y + 1 + 0.5f);
    float br = tex2D(texRef, x + 1 + 0.5f, y + 1 + 0.5f);

    float Gx = -tl - 2.0f * l - bl + tr + 2.0f * r + br;
    float Gy = -tl - 2.0f * t - tr + bl + 2.0f * b + br;
    output[y * width + x] = sqrtf(Gx * Gx + Gy * Gy);
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
  int width = 512, height = 512;
  size_t imgSize = width * height * sizeof(float);

  float *h_image = (float *)malloc(imgSize);
  float *h_output = (float *)malloc(imgSize);
  srand(time(NULL));
  for (int i = 0; i < width * height; i++)
    h_image[i] = (float)(rand() % 256) / 255.0f;

  cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>();
  cudaArray_t cuArray;
  CUDA_CHECK(cudaMallocArray(&cuArray, &channelDesc, width, height));
  CUDA_CHECK(cudaMemcpy2DToArray(cuArray, 0, 0, h_image, width * sizeof(float),
                                 width * sizeof(float), height,
                                 cudaMemcpyHostToDevice));

  texRef.addressMode[0] = cudaAddressModeClamp;
  texRef.addressMode[1] = cudaAddressModeClamp;
  texRef.filterMode = cudaFilterModePoint;
  texRef.normalized = false;
  CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray, channelDesc));

  float *d_output;
  CUDA_CHECK(cudaMalloc(&d_output, imgSize));

  dim3 threadsPerBlock(16, 16);
  dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);
  sobelEdgeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_output, width, height);
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost));
  printf("First 10 edge values:\n");
  for (int i = 0; i < 10; i++)
    printf("%f ", h_output[i]);
  printf("\n");

  CUDA_CHECK(cudaUnbindTexture(texRef));
  CUDA_CHECK(cudaFreeArray(cuArray));
  CUDA_CHECK(cudaFree(d_output));
  free(h_image);
  free(h_output);

  return 0;
}
