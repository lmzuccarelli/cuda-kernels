#include <stdio.h>
#include <stdlib.h>
#include <time.h>

void cpuConvolve(float *image, float *output, float *kernel, int width, int height, int kernelSize) {
    int kHalf = kernelSize / 2;
    for (int y = kHalf; y < height - kHalf; y++) {
        for (int x = kHalf; x < width - kHalf; x++) {
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
}

int main() {

    int width = 1024, height = 1024, kernelSize = 3;
    size_t imgSize = width * height * sizeof(float);
    size_t kernelSizeBytes = kernelSize * kernelSize * sizeof(float);
    float *image = (float*)malloc(imgSize * sizeof(float));
    float *output = (float*)malloc(imgSize * sizeof(float));
    float *kernel = (float*)malloc(kernelSizeBytes * sizeof(float));

    clock_t start = clock();
    cpuConvolve(image, output, kernel, width,height,kernelSize);
    clock_t end = clock();

    printf("CPU 2d_convolution time: %lf seconds\n", ((double)(end - start))/CLOCKS_PER_SEC);

    free(image);
    free(output);
    free(kernel);
    return 0;
}

