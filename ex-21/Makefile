CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: streams 

streams:
	mkdir -p build 
	$(NVCC) $(NVCCFLAGS) src/streams.cu -o build/streams

clean:
	rm -f build/*
