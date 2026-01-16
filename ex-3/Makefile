CC = gcc
NVCC = nvcc
CFLAGS = -O2
NVCCFLAGS = -O2 -arch=sm_86

all: clean build

build:
	mkdir -p build
	$(CC) $(CFLAGS) src/main_cpu.c -o build/cpu
	$(NVCC) $(NVCCFLAGS) src/main_gpu.cu -o gpu

clean:
	rm -f cpu gpu
