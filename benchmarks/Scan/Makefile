CUDA_PATH := /usr/local/cuda-11.4

HOST_COMPILER ?= g++
ARCH 		  := sm_70
NVCC          := $(CUDA_PATH)/bin/nvcc -ccbin $(HOST_COMPILER)
NVCCFLAGS     := -arch=sm_70 -lpthread -std=c++14 -O3 -cudart=shared


COMMON := ../
INC    := -I${COMMON}/include
INC2   := -I${COMMON}/include2
LIB    := ${COMMON}/lib
SRCS   := ${COMMON}/src
BUILD  := ./build

all: start ${BUILD}/Scan_gpu ${BUILD}/scan_dev_gpu ${BUILD}/Scan_gpm ${BUILD}/Scan_rec

start:
	mkdir -p ${BUILD}

${BUILD}/Scan_gpu: scan.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $^ -o $@ -DNVM_ALLOC_GPU -DCONV_LOG

${BUILD}/Scan_rec: scan.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $^ -o $@ -DNVM_ALLOC_GPU -DCONV_LOG -DRECOVERY

${BUILD}/Scan_gpm: scan.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $^ -o $@ -DNVM_ALLOC_GPU -DCONV_LOG -DGPM_EPOCH_MODEL

${BUILD}/scan_dev_gpu: scan_dev.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $^ -o $@ -DNVM_ALLOC_GPU -DCONV_LOG

clean: 
	rm -rf ${LIB}
	rm -f ${BUILD}/*
