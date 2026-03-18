.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/large-data-sets.cu -o build/large-data-sets

clean:
	rm -f build
