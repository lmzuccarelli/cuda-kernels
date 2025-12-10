// customAtomicMax.cu
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

// __device__ function that implements atomicMax for floats using atomicCAS.
// Since CUDA does not provide a native atomicMax for floats, we use atomicCAS by
// reinterpreting the float as an int. The bitwise representation is manipulated
// to compare and update the maximum value.
__device__ float atomicMaxFloat(float* address, float val) {
    // Convert the float pointer to an int pointer, since atomicCAS works on integers.
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        // Set 'assumed' to the current value.
        assumed = old;
        // Compute the maximum of the new value and the current value.
        // __float_as_int converts a float to its integer bit representation.
        // fmaxf returns the maximum of two floats.
        // atomicCAS will compare the integer representation of the current value
        // and, if it matches 'assumed', swap it with the new maximum value.
        old = atomicCAS(address_as_i, assumed, __float_as_int(fmaxf(val, __int_as_float(assumed))));
        // Loop until the value at the address does not change.
    } while (assumed != old);
    // Return the maximum value (reinterpreted back to float).
    return __int_as_float(old);
}

// Kernel that applies atomicMaxFloat to find the maximum value in an array.
__global__ void atomicMaxKernel(const float *input, float *result, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Each thread tries to update the global maximum with its element.
        atomicMaxFloat(result, input[idx]);
    }
}

int main() {
    int N = 1 << 20;  // 1 million elements
    size_t size = N * sizeof(float);
    float *h_input = (float*)malloc(size);
    float h_result = -FLT_MAX;  // Initialize result with the lowest possible float value

    // Initialize host input array with random values between 0 and 100.
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)(rand() % 101);
    }

    float *d_input, *d_result;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_result, sizeof(float));

    // Copy input data and initial result to device memory.
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_result, &h_result, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch the atomic maximum kernel.
    atomicMaxKernel<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_result, N);
    cudaDeviceSynchronize();

    // Copy the final result from device to host.
    cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Maximum value found = %f\n", h_result);

    // Free device and host memory.
    cudaFree(d_input);
    cudaFree(d_result);
    free(h_input);
    return 0;
}
