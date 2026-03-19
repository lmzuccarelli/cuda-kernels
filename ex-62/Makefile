.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/thrust-stream-compaction.cu -o build/thrust-stream-compaction
	$(NVCC) $(NVCCFLAGS) src/thrust-stream-compaction.cu -o build/custom-stream-compaction

clean:
	rm -f build
