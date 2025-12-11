CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: 2d_convolition_gpu 

2d_convolution_cpu:
	mkdir -p build
	$(CC) $(CFLAGS) src/2d_convolition_cpu.c -o build/2d_convolition_cpu

2d_convolution_gpu:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/2d_convolition_gpu.cu -o build/2d_convolition_gpu

clean:
	rm -f build/*
