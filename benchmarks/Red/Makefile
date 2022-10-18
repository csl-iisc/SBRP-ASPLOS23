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

all: start ${BUILD}/Red_gpu ${BUILD}/reduction_dev_gpu ${BUILD}/Red_gpm ${BUILD}/Red_rec

start:
	mkdir -p ${BUILD}

${BUILD}/Red_gpu: reduction.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG

${BUILD}/Red_rec: reduction.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG -DRECOVERY

${BUILD}/Red_gpm: reduction.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG -DGPM_EPOCH_MODEL

${BUILD}/reduction_dev_gpu: reduction_gpu.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG

clean: 
	rm -rf ${LIB}
	rm -f ${BUILD}/*
