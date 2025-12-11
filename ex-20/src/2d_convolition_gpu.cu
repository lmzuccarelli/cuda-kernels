#include <stdio.h>
#include <cuda_runtime.h>
#include <ctime>

__global__ void cudaConvolve(float *image, float *output, float *kernel, int width, int height, int kernelSize) {
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    int kHalf = kernelSize / 2;

    if (x >= kHalf && y >= kHalf && x < width - kHalf && y < height - kHalf) {
        float sum = 0.0f;
        for (int ky = -kHalf; ky <= kHalf; ky++) {
            for (int kx = -kHalf; kx <= kHalf; kx++) {
                int imgIdx = (y + ky) * width + (x + kx);
                int kernelIdx = (ky + kHalf) * kernelSize + (kx + kHalf);
                sum += image[imgIdx] * kernel[kernelIdx];
            }
        }
        output[y * width + x] = sum;
    }
}

int main() {
    int width = 1024, height = 1024, kernelSize = 3;
    size_t imgSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);

    float *h_image = (float*)malloc(imgSize);
    float *h_output = (float*)malloc(imgSize);
    float *h_kernel = (float*)malloc(kernelSizeBytes);

    float *d_image, *d_output, *d_kernel;
    cudaMalloc(&d_image, imgSize);
    cudaMalloc(&d_output, imgSize);
    cudaMalloc(&d_kernel, kernelSizeBytes);

    cudaMemcpy(d_image, h_image, imgSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_kernel, h_kernel, kernelSizeBytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((width + 15) / 16, (height + 15) / 16);

    clock_t start = clock();
    cudaConvolve<<<blocksPerGrid, threadsPerBlock>>>(d_image, d_output, d_kernel, width, height, kernelSize);
    cudaDeviceSynchronize();
    clock_t end = clock();

    cudaMemcpy(h_output, d_output, imgSize, cudaMemcpyDeviceToHost);

    printf("GPU 2d_convolution tme: %lf seconds\n", ((double)(end - start))/CLOCKS_PER_SEC);

    cudaFree(d_image);
    cudaFree(d_output);
    cudaFree(d_kernel);
    free(h_image);
    free(h_output);
    free(h_kernel);

    return 0;
}
