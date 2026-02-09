.PHONY: all clean build

NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build

build:
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/coalesced_kernel.cu -o build/coalesced_kernel
	$(NVCC) $(NVCCFLAGS) src/non_coalesced_kernel.cu -o build/non_coalesced_kernel

clean:
	rm -f build
