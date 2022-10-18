# CUDA toolkit installation path
CUDA_DIR = /usr/local/cuda-11.4/
 
# CUDA toolkit libraries
CUDA_LIB_DIR := $(CUDA_DIR)/lib64
ifeq ($(shell uname -m), x86_64)
     ifeq ($(shell if test -d $(CUDA_DIR)/lib64; then echo T; else echo F; fi), T)
        CUDA_LIB_DIR := $(CUDA_DIR)/lib64
     endif
endif
 
SDK_DIR = /usr/local/cuda/samples/

OPENCL_DIR = /if10/kw5na/Packages/AMD-APP-SDK-v2.8-RC-lnx64
OPENCL_INC = $(OPENCL_DIR)/include/
OPENCL_LIB = $(OPENCL_DIR)/lib/x86_64/ -lOpenCL

CC := $(CUDA_DIR)/bin/nvcc

NVCC_FLAGS := -rdc=true -std=c++11 -lpthread -O3 -arch=sm_70 -Xcompiler -fopenmp --cudart shared

PMEM_LIB := ../../libgpm/src/pmem-helper.cu ../../libgpm/src/libpmem-g.cu ../../libgpm/src/libpmemlog-g. cu ../../libgpm/src/libpmemcp-g.cu

PMEM_INC = ../include2  

INCLUDE := $(CUDA_DIR)/include

SRC = ./src 
BUILD = ./build

#EXE = srad

all: start ${BUILD}/SRAD_gpu ${BUILD}/SRAD_gpm ${BUILD}/SRAD_rec

start:
	mkdir -p ${BUILD}

${BUILD}/srad_mm_gpu: ./src/srad_mm.cu ./src/srad_kernel_mm.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU

${BUILD}/SRAD_gpu: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU

${BUILD}/SRAD_rec: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DRECOVERY

${BUILD}/SRAD_gpm: ./src/srad.cu ./src/srad_kernel.cu
	$(CC) $(KERNEL_DIM) $(NVCC_FLAGS) $< -o $@ -I$(INCLUDE) -I$(PMEM_INC) -L$(CUDA_LIB_DIR) -DNVM_ALLOC_GPU -DGPM_EPOCH_MODEL

enum: ${SRC}/srad.cu
	$(CC) $(KERNEL_DIM) -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debug: ${SRC}/srad.cu
	$(CC) $(KERNEL_DIM) -g $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

debugenum: ${SRC}/srad.cu
	$(CC) $(KERNEL_DIM) -g -deviceemu $(SRC) -o $(EXE) -I$(INCLUDE) -L$(CUDA_LIB_DIR) 

clean:
	rm -f $(EXE) $(EXE).linkinfo result.txt
	rm -f ${BUILD}/*
