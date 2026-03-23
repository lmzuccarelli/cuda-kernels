#include <cudnn.h>
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDNN_CHECK(status) \
    if (status != CUDNN_STATUS_SUCCESS) { \
        printf("cuDNN error: %s at line %d\n", cudnnGetErrorString(status), __LINE__); \
        exit(EXIT_FAILURE); \
    }

int main() {
    cudnnHandle_t cudnn;
    CUDNN_CHECK(cudnnCreate(&cudnn));

    int N = 1, C = 3, H = 224, W = 224;  // Batch size, Channels, Height, Width
    int K = 64, R = 3, S = 3;            // Number of filters, filter height, filter width

    float *d_input, *d_filter, *d_output;
    size_t inputSize = N * C * H * W * sizeof(float);
    size_t filterSize = K * C * R * S * sizeof(float);
    int outH = H, outW = W;
    size_t outputSize = N * K * outH * outW * sizeof(float);

    cudaMalloc(&d_input, inputSize);
    cudaMalloc(&d_filter, filterSize);
    cudaMalloc(&d_output, outputSize);


    cudnnTensorDescriptor_t inputDesc, outputDesc;
    cudnnFilterDescriptor_t filterDesc;
    cudnnConvolutionDescriptor_t convDesc;

    CUDNN_CHECK(cudnnCreateTensorDescriptor(&inputDesc));
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&filterDesc));
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&outputDesc));
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&convDesc));

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(inputDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           N, C, H, W));
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(filterDesc,
                                           CUDNN_DATA_FLOAT,
                                           CUDNN_TENSOR_NCHW,
                                           K, C, R, S));
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(convDesc,
                                                1, 1, 
                                                1, 1, 
                                                1, 1, 
                                                CUDNN_CROSS_CORRELATION,
                                                CUDNN_DATA_FLOAT));

    int outN, outC, outHComputed, outWComputed;
    CUDNN_CHECK(cudnnGetConvolution2dForwardOutputDim(convDesc,
                                                      inputDesc,
                                                      filterDesc,
                                                      &outN, &outC, &outHComputed, &outWComputed));
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(outputDesc,
                                           CUDNN_TENSOR_NCHW,
                                           CUDNN_DATA_FLOAT,
                                           outN, outC, outHComputed, outWComputed));

    cudnnConvolutionFwdAlgo_t algo;
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                    inputDesc,
                                                    filterDesc,
                                                    convDesc,
                                                    outputDesc,
                                                    CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                    0,
                                                    &algo));

    size_t workspaceSize = 0;
    CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                        inputDesc,
                                                        filterDesc,
                                                        convDesc,
                                                        outputDesc,
                                                        algo,
                                                        &workspaceSize));
    void *d_workspace = nullptr;
    cudaMalloc(&d_workspace, workspaceSize);

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(cudnn,
                                        &alpha,
                                        inputDesc, d_input,
                                        filterDesc, d_filter,
                                        convDesc, algo,
                                        d_workspace, workspaceSize,
                                        &beta,
                                        outputDesc, d_output));

    cudaDeviceSynchronize();

    cudaFree(d_workspace);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudnnDestroyTensorDescriptor(inputDesc);
    cudnnDestroyTensorDescriptor(outputDesc);
    cudnnDestroyFilterDescriptor(filterDesc);
    cudnnDestroyConvolutionDescriptor(convDesc);
    cudnnDestroy(cudnn);

    return 0;
}
