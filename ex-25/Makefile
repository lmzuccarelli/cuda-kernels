.PHONY: all build clean 

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/double-buffering.cu -o build/double-buffering

clean:
	rm -f build
