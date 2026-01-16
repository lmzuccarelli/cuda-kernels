CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build

build:
	mkdir -p build
	$(CC) $(CFLAGS) src/2d_convolition_cpu.c -o build/2d_convolition_cpu
	$(NVCC) $(NVCCFLAGS) src/2d_convolition_gpu.cu -o build/2d_convolition_gpu

clean:
	rm -f build/*
