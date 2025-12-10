CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: atomic_add 

atomic_add: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/atomic_add.cu -o build/atomic_add

clean:
	rm -f build/*
