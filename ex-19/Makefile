.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/unified_memory.cu -o build/unified_memory

clean:
	rm -f build
