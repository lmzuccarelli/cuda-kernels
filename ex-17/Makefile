CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: synchronize incorrect-synchronize

synchronize:
	mkdir -p build	
	$(NVCC) $(NVCCFLAGS) src/synchronize.cu -o build/synchronize

incorrect-synchronize: 
	mkdir -p build	
	$(NVCC) $(NVCCFLAGS) src/incorrect_synchronize.cu -o build/incorrect_synchronize

clean:
	rm -f build/*
