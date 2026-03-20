// pinned memeory
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

int main() {
    int N = 1 << 20; 
    size_t size = N * sizeof(float);
    float *h_data;    
    float *d_data;    

    // TODO: Allocate pinned memory for h_data
    // See solution
    
    for (int i = 0; i < N; i++) {
        h_data[i] = (float)i;
    }

    cudaMalloc(&d_data, size);
    cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

    // TODO: Free pinned memory
    cudaFree(d_data);
    return 0;
}

// solution

{
// ...
float *h_data;
// Allocate pinned memory
cudaMallocHost((void**)&h_data, size);

// Initialize pinned host memory

float *d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// (Kernel code if needed) ...

cudaFreeHost(h_data);
cudaFree(d_data);
return 0;
}

// Q2_multi_stream_example.cu
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void producerKernel(float *buffer, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        buffer[idx] = (float)idx;
    }
}

__global__ void consumerKernel(const float *buffer, float *output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = buffer[idx] * 2.0f;
    }
}

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    float *d_buffer, *d_output;
    cudaMalloc(&d_buffer, size);
    cudaMalloc(&d_output, size);

    cudaStream_t streamProd, streamCons;
    cudaStreamCreate(&streamProd);
    cudaStreamCreate(&streamCons);

    producerKernel<<<(N+255)/256, 256, 0, streamProd>>>(d_buffer, N);

    // TODO: Insert event record and make streamCons wait on it

    consumerKernel<<<(N+255)/256, 256, 0, streamCons>>>(d_buffer, d_output, N);

    cudaStreamSynchronize(streamProd);
    cudaStreamSynchronize(streamCons);

    cudaFree(d_buffer);
    cudaFree(d_output);
    cudaStreamDestroy(streamProd);
    cudaStreamDestroy(streamCons);
    return 0;
}

// solution

cudaEvent_t event;
cudaEventCreate(&event);

// Launch producer kernel
producerKernel<<<(N+255)/256, 256, 0, streamProd>>>(d_buffer, N);

// Record event after producer finishes
cudaEventRecord(event, streamProd);

// Make consumer stream wait for the event
cudaStreamWaitEvent(streamCons, event, 0);

consumerKernel<<<(N+255)/256, 256, 0, streamCons>>>(d_buffer, d_output, N);
// ...


// Q3_bfs_atomic_example.cu
__global__ void bfsKernel(const int* d_offsets, const int* d_cols,
                          int* visited, const int* frontier, int frontierSize,
                          int* nextFrontier, int* nextCount, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < frontierSize) {
        int node = frontier[idx];
        int start = d_offsets[node];
        int end = d_offsets[node + 1];
        for (int offset = start; offset < end; offset++) {
            int neighbor = d_cols[offset];
            // TODO: Mark neighbor visited and append to nextFrontier
        }
    }
}

// solution
int oldVal = atomicCAS(&visited[neighbor], 0, 1);
if (oldVal == 0) {
    // neighbor was unvisited
    int pos = atomicAdd(nextCount, 1);
    nextFrontier[pos] = neighbor;
}


// Q4_memory_pool_example.cu
#include <cuda_runtime.h>
#include <stdio.h>

struct DevicePool {
    char* poolStart;
    size_t poolSize;
    size_t offset;
};

void initPool(DevicePool &dp) {
    // TODO: allocate 4 MB device memory
    // set dp.offset = 0
}

void* allocateBlock(DevicePool &dp, size_t size) {
    // TODO: return a pointer from dp.poolStart + dp.offset
    // update dp.offset
}

void destroyPool(DevicePool &dp) {
    // TODO: free dp.poolStart
}

// solution
void initPool(DevicePool &dp) {
    dp.poolSize = 4 * 1024 * 1024; // 4 MB
    cudaMalloc((void**)&dp.poolStart, dp.poolSize);
    dp.offset = 0;
}

void* allocateBlock(DevicePool &dp, size_t size) {
    if (dp.offset + size > dp.poolSize) return nullptr;
    void* ptr = (void*)(dp.poolStart + dp.offset);
    dp.offset += size;
    return ptr;
}

void destroyPool(DevicePool &dp) {
    cudaFree(dp.poolStart);
    dp.poolStart = nullptr;
    dp.offset = 0;
    dp.poolSize = 0;
}

// Q5_occ_bounds_example.cu
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    // Tiled matrix multiply logic omitted
}

// solution
__launch_bounds__(128, 2)
__global__ void matrixMulKernel(const float* A, const float* B, float* C, int N) {
    // Tiled matrix multiply logic ...
}

// Q6_um_bfs_example.cu
#include <cuda_runtime.h>
#include <stdio.h>

int main() {
    int N = 1024;
    // TODO: Use cudaMallocManaged for visited, frontier
    // BFS logic etc.
}

// solution
int *visited, *frontier;
cudaMallocManaged(&visited, N * sizeof(int));
cudaMallocManaged(&frontier, N * sizeof(int));

// Initialize as needed
for (int i = 0; i < N; i++) {
    visited[i] = 0;
    frontier[i] = -1;
}

// BFS kernel usage ...
// Host can read visited, frontier directly

cudaFree(visited);
cudaFree(frontier);


// Q7_multi_gpu_split.cu
int main() {
    int N = 2000000;
    size_t halfSize = (N / 2) * sizeof(float);

    float *d_gpu0, *d_gpu1;

    // TODO: Set device 0, allocate d_gpu0
    // TODO: Set device 1, allocate d_gpu1

    // TODO: Memset or copy data for each half
    // Launch kernels, synchronize
    // Clean up
}

// solution
int main() {
    int N = 2000000;
    size_t halfN = N / 2;
    size_t halfSize = halfN * sizeof(float);

    float *d_gpu0, *d_gpu1;

    cudaSetDevice(0);
    cudaMalloc(&d_gpu0, halfSize);
    cudaMemset(d_gpu0, 0, halfSize);

    cudaSetDevice(1);
    cudaMalloc(&d_gpu1, halfSize);
    cudaMemset(d_gpu1, 0, halfSize);

    // Launch a kernel on each GPU
    cudaSetDevice(0);
    doubleKernel<<<(halfN+255)/256, 256>>>(d_gpu0, halfN);

    cudaSetDevice(1);
    doubleKernel<<<(halfN+255)/256, 256>>>(d_gpu1, halfN);

    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();

    cudaFree(d_gpu0);
    cudaFree(d_gpu1);
    return 0;
}

// Example doubleKernel
__global__ void doubleKernel(float *data, int size) {
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    if(idx < size) {
        data[idx] *= 2.0f;
    }
}


// Q8_minimal_allocator.cu
static void *g_pool = nullptr;
static size_t g_poolSize = 0;

void initPoolSize(size_t poolSize) {
    // ...
}
void* allocateDeviceArray(size_t size) {
    // ...
}

// solution
void initPoolSize(size_t poolSize) {
    cudaMalloc(&g_pool, poolSize);
    g_poolSize = poolSize;
}

void* allocateDeviceArray(size_t size) {
    if (size <= g_poolSize) {
        return g_pool; // naive reuse
    }
    return nullptr;
}


// Q9_p2p_check.cu
int main() {
    int canAccess01, canAccess10;
    // TODO: check if device 0 can access 1, and device 1 can access 0
    // if both true, do cudaDeviceEnablePeerAccess
}

// solution
int canAccess01, canAccess10;
cudaDeviceCanAccessPeer(&canAccess01, 0, 1);
cudaDeviceCanAccessPeer(&canAccess10, 1, 0);

if (canAccess01 && canAccess10) {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);
    printf("P2P enabled between GPU 0 and 1.\n");
} else {
    printf("P2P not supported.\n");
}

// Q10_occ_reduction_example.cu
__global__ void occReductionKernel(const float* input, float* output, int N) {
    // ...
}

// solution
__launch_bounds__(128, 2)
__global__ void occReductionKernel(const float* input, float* output, int N) {
    extern __shared__ float sdata[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // basic partial reduction ...
}

int main() {
    int N = 1 << 20;
    size_t size = N * sizeof(float);
    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);

    dim3 block(128);
    dim3 grid((N + block.x - 1)/block.x);
    occReductionKernel<<<grid, block, block.x * sizeof(float)>>>(d_in, d_out, N);
    cudaDeviceSynchronize();

    // Nsight Compute: Check “Achieved Occupancy” vs. “Theoretical Occupancy”
    // and see if register usage or shared mem are limiting factors.
    return 0;
}
