#!/bin/bash

mkdir $1
cd $1
mkdir src
mkdir include
mkdir build

cat <<EOF > Makefile
CC=gcc
NVCC=nvcc
CFLAGS=-O2
NVCCFLAGS=-O2 -arch=sm_86

all: $2 

$2:
	mkdir -p build 
	\$(NVCC) \$(NVCCFLAGS) src/$2.cu -o build/$2

clean:
	rm -f build/*
EOF

touch src/$2.cu
