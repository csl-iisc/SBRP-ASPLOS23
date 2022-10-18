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

LIBRARIES := ${LIB}/libgpm.o ${LIB}/libgpmlog.o

all: start ${BUILD}/HM_gpu ${BUILD}/HM_gpm ${BUILD}/HM_rec

start:
	mkdir -p ${BUILD}

nvbit: parallel_hash_map2.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o parallel_hashmap_nvm_nvbit -DNVM_ALLOC_GPU
	
${BUILD}/HM_gpu: parallel_hash_map.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG
	
${BUILD}/HM_rec: parallel_hash_map.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG -DRECOVERY
	
${BUILD}/HM_gpm: parallel_hash_map.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DCONV_LOG -DGPM_EPOCH_MODEL

${BUILD}/hashmap_cpu: parallel_hash_map2.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_CPU -DFAKE_NVM
	
${BUILD}/hashmap_emul_gpu: parallel_hash_map2.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_GPU -DEMULATE_NVM
	
${BUILD}/hashmap_real_cpu: parallel_hash_map2.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_CPU -DEMULATE_NVM
	
${BUILD}/hashmap_real_no-hcl_cpu: parallel_hash_map2.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@ -DNVM_ALLOC_CPU -DEMULATE_NVM -DCONV_LOG
	
${BUILD}/hashmap_fs_gpu: parallel_hash_map_fs.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@

${BUILD}/hashmap_mm_gpu: parallel_hash_map2_mm.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@

${BUILD}/hashmap_mm_tx_gpu: parallel_hash_map_mm_tx.cu
	${NVCC} ${NVCCFLAGS} ${INC2} $< -o $@

#hashmap_gpu: ${LIBRARIES} parallel_hash_map.o ${SRCS}/gpm-helper.cu
#	mkdir -p ${LIB}
#	${NVCC} ${NVCCFLAGS} ${INC} -dc ${SRCS}/gpm-helper.cu -o ${LIB}/gpm-helper.o -DNVM_ALLOC_GPU
#	${NVCC} ${NVCCFLAGS} ${INC} ${LIBRARIES} ${LIB}/gpm-helper.o parallel_hash_map.o -o $@
	

#hashmap_cpu: ${LIBRARIES} parallel_hash_map.o ${SRCS}/gpm-helper.cu
#	mkdir -p ${LIB}
#	${NVCC} ${NVCCFLAGS} ${INC} ${SRCS}/gpm-helper.cu -o ${LIB}/gpm-helper.o -dc
#	${NVCC} ${NVCCFLAGS} ${INC} ${LIBRARIES} ${LIB}/gpm-helper.o parallel_hash_map.o -o $@
	
parallel_hash_map.o: parallel_hash_map.cu
	${NVCC} ${NVCCFLAGS} ${INC} $< -o $@ -dc
	
${LIB}/libgpm.o: ${SRCS}/libgpm.cu
	mkdir -p ${LIB}
	${NVCC} ${NVCCFLAGS} ${INC} $< -o $@ -dc
	
${LIB}/libgpmlog.o: ${SRCS}/libgpmlog.cu
	mkdir -p ${LIB}
	${NVCC} ${NVCCFLAGS} ${INC} $< -o $@ -dc

clean: 
	rm -rf ${LIB}
	rm -f ${BUILD}/*
