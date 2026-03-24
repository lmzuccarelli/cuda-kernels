.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/cuda-graph-updates.cu -o build/cuda-graph-updates

clean:
	rm -f build
