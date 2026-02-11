.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/unified-memory.cu -o build/unified-memory

clean:
	rm -f build
