CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/async-memory.cu -o build/async-memory

clean:
	rm -f build/*
