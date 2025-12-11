CC = gcc
NVCC = nvcc
CFLAGS = -O2
NVCCFLAGS = -O2 -arch=sm_86

all: cpu gpu

cpu: src/main_cpu.c
	mkdir -p build
	$(CC) $(CFLAGS) src/main_cpu.c -o build/cpu

gpu: src/main_gpu.cu
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/main_gpu.cu -o gpu

clean:
	rm -f cpu gpu
