.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 -ptx

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/ptx-analysis.cu -o build/ptx-analysis

clean:
	rm -f build
