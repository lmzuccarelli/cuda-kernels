.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 -lcublas -lcufft

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/cufft.cu -o build/cufft

clean:
	rm -f build
