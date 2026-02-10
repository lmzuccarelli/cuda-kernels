.PHONY: all clean build

# CC=gcc
NVCC=nvcc
# CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86 

all: clean build 

build:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/double-buffer-mmul.cu -o build/double-buffer-mmul
	$(NVCC) $(NVCCFLAGS) src/sobel-edge-texture-memory.cu -o build/sobel-edge-texture-memory

clean:
	rm -f build
