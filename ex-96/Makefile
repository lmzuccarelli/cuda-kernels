.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	$(CC) $(CCFLAGS) src/standard-summation.c -o build/standard-summation
	$(CC) $(CCFLAGS) src/kahan-summation.c -o build/kahan-summation
	$(CC) $(CCFLAGS) src/pairwise-summation.c -o build/pairwise-sumamtion
	$(CC) $(CCFLAGS) src/neumaier-summation.c -o build/neumaier-sumamtion

clean:
	rm -f build
