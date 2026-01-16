CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build

build:
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/atomic_max.cu -o build/atomic_max
	$(NVCC) $(NVCCFLAGS) src/atomic_exch.cu -o build/atomic_exch

clean:
	rm -f build/*
