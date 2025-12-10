CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: syncthreads_kernel 

syncthreads_kernel: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/syncthreads_kernel.cu -o build/syncthreads_kernel

clean:
	rm -f build/*
