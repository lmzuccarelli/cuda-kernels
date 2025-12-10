CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: error_handling 

error_handling:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/error_handling.cu -o build/error_handling

clean:
	rm -f build/*
