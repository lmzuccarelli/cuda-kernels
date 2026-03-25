.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	mpicxx -fopenmp -I/usr/cuda/include -L/user/uda/lib64 -lcudart -o build/cuda_openmp src/cuda_openmp.cpp

clean:
	rm -f build
