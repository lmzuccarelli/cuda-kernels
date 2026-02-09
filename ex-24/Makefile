.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/pageable-memory.cu -o build/pageable-memory
	$(NVCC) $(NVCCFLAGS) src/pinned-memory.cu -o build/pinned-memory
	$(NVCC) $(NVCCFLAGS) src/compare-timing.cu -o build/compare-timing

clean:
	rm -f build
