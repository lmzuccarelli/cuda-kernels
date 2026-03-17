.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 -g -G

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/intermediate-debugging.cu -o build/intermediate-debugging

clean:
	rm -f build
