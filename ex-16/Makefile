CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: kernel_tuning 

kernel_tuning: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/kernel_tuning.cu -o build/kernel_tuning

clean:
	rm -f build/*
