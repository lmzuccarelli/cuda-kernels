CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: shm_bank_kernel 

shm_bank_kernel: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/shm_bank_kernel.cu -o build/shm_bank_kernel

clean:
	rm -f build/*
