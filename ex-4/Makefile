.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: build

build:
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/vec_add.cu -o build/vec_add
	$(NVCC) $(NVCCFLAGS) src/vec_mult.cu -o build/vec_mult

clean:
	rm -f build
