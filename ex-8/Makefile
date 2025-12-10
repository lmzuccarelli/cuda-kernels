CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: vec-add

vec-add:
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/vec_add_error_handling.cu -o build/vec_add

clean:
	rm -f build/*
