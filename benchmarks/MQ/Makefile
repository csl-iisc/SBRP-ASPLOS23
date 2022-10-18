# CUDA toolkit installation path
CUDA_DIR = /usr/local/cuda-11.4
 
# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
        CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif
 
SDK_DIR = /usr/local/cuda-11.4/samples/

OPENCL_DIR = /if10/kw5na/Packages/AMD-APP-SDK-v2.8-RC-lnx64
OPENCL_INC = $(OPENCL_DIR)/include/
OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL

CC := $(CUDA_DIR)/bin/nvcc
#CC := nvcc

NVCC_FLAGS := -rdc=true -std=c++11 -lpthread -O3 -arch=sm_70 -Xcompiler -fopenmp --cudart shared

PMEM_LIB := ../../libgpm/src/pmem-helper.cu ../../libgpm/src/libpmem-g.cu ../../libgpm/src/libpmemlog-g. cu ../../libgpm/src/libpmemcp-g.cu

PMEM_INC = ../include2  

INCLUDE := $(CUDA_DIR)/include

SRC = ./src 
BUILD = ./build

#EXE = srad

all: start ${BUILD}/MQ_gpu ${BUILD}/multiqueue_dev_gpu ${BUILD}/MQ_gpm ${BUILD}/MQ_rec

start:
	mkdir -p ${BUILD}

${BUILD}/MQ_gpu: ./src/multi_queue.cu ./src/gpu_queue.h
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU

${BUILD}/MQ_rec: ./src/multi_queue.cu ./src/gpu_queue.h
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DRECOVERY

${BUILD}/MQ_gpm: ./src/multi_queue.cu ./src/gpu_queue.h
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DGPM_EPOCH_MODEL

${BUILD}/multiqueue_dev_gpu: ./src/multi_queue.cu ./src/gpu_queue_gpu.h
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DDEVICE_SCOPE

clean:
	rm -f $(EXE) $(EXE).linkinfo result.txt
	rm -f ${BUILD}/*
