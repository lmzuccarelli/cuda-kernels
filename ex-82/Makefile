.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 -lcublas

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/large-scale-batch-gemm.cu -o build/large-scale-batch-gemm

clean:
	rm -f build
