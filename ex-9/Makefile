NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: coalesced_kernel

coalesced_kernel: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/coalesced_kernel.cu -o build/coalesced_kernel

non_coalesced_kernel: 
	mkdir -p build
	$(NVCC) $(NVCCFLAGS) src/non_coalesced_kernel.cu -o build/non_coalesced_kernel

clean:
	rm -f build/*
