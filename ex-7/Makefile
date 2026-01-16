NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: clean build

build:
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/vec_scaling.cu -o build/vec_scaling

clean:
	rm -f build/*
