CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: tiled_kernel 

tiled_kernel: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/tiled_kernel.cu -o build/tiled_kernel

clean:
	rm -f build/*
