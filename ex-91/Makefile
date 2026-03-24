.PHONY: all clean build

CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build 

build:
	mkdir -p build 
	nvcc -rdc=true -arch=sm_86 -c src/kernel-functions.cu -o build/kernel-functions.o
	nvcc -rdc=true -arch=sm_86 -c src/main.cu -o build/main.o
	nvcc -arch=sm_86 build/kernel-functions.o build/main.o -o build/app

clean:
	rm -f build
