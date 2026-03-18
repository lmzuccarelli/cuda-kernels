.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 -lcublas

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/saxpy-cublas.cu -o build/saxpy-cublas
	$(NVCC) $(NVCCFLAGS) src/sgemm-cublas.cu -o build/sgemm-cublas

clean:
	rm -f build
