__inline__ __device__
float warpReduceSum(float val) {
    // Full mask for all 32 threads in warp
    unsigned int mask = 0xffffffff;
    
    // For a 32-thread warp, do log2(32) = 5 steps
    // 1,2,4,8,16
    val += __shfl_down_sync(mask, val, 16, 32);
    val += __shfl_down_sync(mask, val, 8, 32);
    val += __shfl_down_sync(mask, val, 4, 32);
    val += __shfl_down_sync(mask, val, 2, 32);
    val += __shfl_down_sync(mask, val, 1, 32);
    return val;
}

__global__ void warpReductionExample(const float *input, float *output) {
    // We assume each warp only handles 32 elements for simplicity
    int laneId = threadIdx.x & 31;  // mod 32
    int warpId = threadIdx.x >> 5;
    
    float val = input[threadIdx.x];  // each thread reads one element from input
    
    // Now do warp-level reduce sum
    val = warpReduceSum(val);
    
    // Only lane 0 writes out the warp’s result
    if (laneId == 0) {
        output[warpId] = val;
    }
}
