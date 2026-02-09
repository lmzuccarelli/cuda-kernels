.PHONY: all clean build

NVCC = nvcc
SRCS = src/main.cu src/kernels.cu
INCLUDES = -I./include
ARCH = -arch=sm_86
FLAGS = -Xcompiler "-Wall -O2"

all: clean build

build:
	mkdir -p build
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o build/$(SRCS)

clean:
	rm -f build
