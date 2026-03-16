#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <iostream>

void createCSR(int N, const float *denseMatrix,
               std::vector<int> &rowOffsets,
               std::vector<int> &columns,
               std::vector<float> &values) {
    rowOffsets.resize(N+1, 0);
    int nnz = 0;
    for(int i=0; i<N; i++){
        for(int j=0; j<N; j++){
            float val = denseMatrix[i*N + j];
            if(val != 0.0f){
                columns.push_back(j);
                values.push_back(val);
                nnz++;
            }
        }
        rowOffsets[i+1] = nnz;
    }
}

__global__ void csrSpmvKernel(const int *rowOffsets,
                              const int *columns,
                              const float *values,
                              const float *x,
                              float *y,
                              int numRows) {
    int row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < numRows) {
        int rowStart = rowOffsets[row];
        int rowEnd   = rowOffsets[row+1];
        float sum    = 0.0f;
        for(int jj = rowStart; jj < rowEnd; jj++){
            int col = columns[jj];
            float val = values[jj];
            sum += val * x[col];
        }
        y[row] = sum;
    }
}

int main() {
    int N = 5;  

    std::vector<float> denseMat(N*N, 0.0f);

    std::vector<int> rowOffsets, columns;
    std::vector<float> values;
    createCSR(N, denseMat.data(), rowOffsets, columns, values);
    int nnz = values.size();

    int *d_rowOffsets, *d_columns;
    float *d_values, *d_x, *d_y;
    cudaMalloc(&d_rowOffsets, (N+1)*sizeof(int));
    cudaMalloc(&d_columns, nnz*sizeof(int));
    cudaMalloc(&d_values, nnz*sizeof(float));

    cudaMemcpy(d_rowOffsets, rowOffsets.data(), (N+1)*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_columns, columns.data(), nnz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_values, values.data(), nnz*sizeof(float), cudaMemcpyHostToDevice);

    std::vector<float> h_x(N, 1.0f); 
    std::vector<float> h_y(N, 0.0f);
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));
    cudaMemcpy(d_x, h_x.data(), N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_y, 0, N*sizeof(float));

    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1)/threadsPerBlock;
    csrSpmvKernel<<<blocksPerGrid, threadsPerBlock>>>(d_rowOffsets, d_columns,
                                                      d_values, d_x, d_y, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_y.data(), d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "y[0] = " << h_y[0] << " y[N-1] = " << h_y[N-1] << std::endl;

    cudaFree(d_rowOffsets);
    cudaFree(d_columns);
    cudaFree(d_values);
    cudaFree(d_x);
    cudaFree(d_y);

    return 0;
}


