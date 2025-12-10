// atomicExchExample.cu
#include <cuda_runtime.h>
#include <stdio.h>

// Kernel demonstrating the use of atomicExch.
// Each thread swaps its element with a new value (100.0f), retrieves the old value,
// and then adds the new value to the old value, storing the result back.
__global__ void atomicExchKernel(float *data, float newValue, int N) {
    // Calculate the global thread index.
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Atomically exchange the value at data[idx] with newValue.
        // atomicExch returns the old value that was at data[idx].
        float oldValue = atomicExch(&data[idx], newValue);
        // For demonstration, we perform an additional computation:
        // Add the old value and the new value, and write it back to data[idx].
        data[idx] = oldValue + newValue;
    }
}

int main() {
    int N = 1024;  // Array size
    size_t size = N * sizeof(float);
    float *h_data = (float*)malloc(size);

    // Initialize host array with sequential values.
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    float *d_data;
    cudaMalloc((void**)&d_data, size);
    // Copy host data to device.
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    // Launch the kernel with newValue set to 100.0f.
    atomicExchKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, 100.0f, N);
    cudaDeviceSynchronize();

    // Copy the modified array back to host.
    cudaMemcpy(h_data, d_data, size, cudaMemcpyDeviceToHost);
    printf("AtomicExch result (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("%f ", h_data[i]);
    }
    printf("\n");

    // Free device and host memory.
    cudaFree(d_data);
    free(h_data);
    return 0;
}
