CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: unified_memory 

unified_memory:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/unified_memory.cu -o build/unified_memory

clean:
	rm -f build/*
