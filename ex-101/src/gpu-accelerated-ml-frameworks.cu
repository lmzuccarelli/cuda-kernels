#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>


__global__ void add_constant_kernel(const float* input, float* output, float constant, int num_elements) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx] + constant;
    }
}


void add_constant_kernel(const float* input, float* output, float constant, int num_elements);

void add_constant_cuda(torch::Tensor input, torch::Tensor output, float constant) {
    const int num_elements = input.numel();
    const int threads = 256;
    const int blocks = (num_elements + threads - 1) / threads;
    
    add_constant_kernel<<<blocks, threads>>>(input.data_ptr<float>(), output.data_ptr<float>(), constant, num_elements);
}

void add_constant(torch::Tensor input, torch::Tensor output, float constant) {
    add_constant_cuda(input, output, constant);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("add_constant", &add_constant, "Add constant to tensor (CUDA)");
}

// Usage in python

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='my_custom_op',
    ext_modules=[
        CUDAExtension(
            name='my_custom_op',
            sources=['my_custom_op.cpp', 'my_custom_kernel.cu'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)


import torch
import my_custom_op

# Create a tensor and an output tensor
input_tensor = torch.randn(1024, device='cuda')
output_tensor = torch.empty_like(input_tensor)

# Call the custom op
my_custom_op.add_constant(input_tensor, output_tensor, 5.0)
print(output_tensor)
