PROJECT = cuda_app
NVCC = nvcc
SRCS = src/main.cu src/kernels.cu
INCLUDES = -I./include
ARCH = -arch=sm_86
FLAGS = -Xcompiler "-Wall -O2"

all: $(PROJECT)

$(PROJECT): $(SRCS)
	mkdir -p build
	$(NVCC) $(ARCH) $(FLAGS) $(INCLUDES) -o $(PROJECT) $(SRCS)

clean:
	rm -f $(PROJECT) *.o
