// kernel_functions.h
#ifndef KERNEL_FUNCTIONS_H
#define KERNEL_FUNCTIONS_H

#ifdef __cplusplus
extern "C" {
#endif

__global__ void addKernel(float* d_out, const float* d_in, int N);

#ifdef __cplusplus
}
#endif

#endif // KERNEL_FUNCTIONS_H
