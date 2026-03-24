.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/pinned-memory-async-tx.cu -o build/pinned-memory-async-tx
	$(NVCC) $(NVCCFLAGS) src/kernel-side-gen.cu -o build/kernel-side-gen

clean:
	rm -f build
