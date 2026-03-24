.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/cuda-grpah-optimized-workload.cu -o build/cuda-grpah-optimized-workload

clean:
	rm -f build
