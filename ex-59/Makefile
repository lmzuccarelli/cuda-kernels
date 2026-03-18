.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/process-a.cu -o build/process-a
	$(NVCC) $(NVCCFLAGS) src/process-a.cu -o build/process-b

clean:
	rm -f build
