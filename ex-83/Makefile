.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 -lcudnn  -I/usr/local/cuda/include -L/usr/local/cuda/lib64 

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/cudann-conv-layer.cu -o build/cudann-conv-layer

clean:
	rm -f build
