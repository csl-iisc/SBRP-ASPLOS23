CC = g++

CFLAGS = -Wall
#-D__USE_GNU -D_GNU_SOURCE 
LIBS = -lrt -lpthread

#OPENSSL_DIR = ../openssl-1.0.1c
#ifneq ($(OPENSSL_DIR), )
#	OPENSSL_LIBDIR = -L$(OPENSSL_DIR)/lib64
#	OPENSSL_INCDIR = -I$(OPENSSL_DIR)/include
#endif

################################################################
# OS Name (Linux or Darwin)
OSUPPER = $(shell uname -s 2>/dev/null | tr [:lower:] [:upper:])
OSLOWER = $(shell uname -s 2>/dev/null | tr [:upper:] [:lower:])

# Flags to detect 32-bit or 64-bit OS platform
OS_SIZE = $(shell uname -m | sed -e "s/i.86/32/" -e "s/x86_64/64/")
OS_ARCH = $(shell uname -m | sed -e "s/i386/i686/")

# These flags will override any settings
ifeq ($(i386),1)
	OS_SIZE = 32
	OS_ARCH = i686
endif

ifeq ($(x86_64),1)
	OS_SIZE = 64
	OS_ARCH = x86_64
endif

# Flags to detect either a Linux system (linux) or Mac OSX (darwin)
DARWIN = $(strip $(findstring DARWIN, $(OSUPPER)))

# Location of the CUDA Toolkit binaries and libraries
CUDA_PATH       ?= /usr/local/cuda-11.4
CUDA_INC_PATH   ?= $(CUDA_PATH)/include
CUDA_BIN_PATH   ?= $(CUDA_PATH)/bin
ifneq ($(DARWIN),)
  CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
else
  ifeq ($(OS_SIZE),32)
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib
  else
    CUDA_LIB_PATH  ?= $(CUDA_PATH)/lib64
  endif
endif

# Common binaries
NVCC            ?= $(CUDA_BIN_PATH)/nvcc

# OS-specific build flags
ifneq ($(DARWIN),) 
      LDFLAGS   := -Xlinker -rpath $(CUDA_LIB_PATH) -L $(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -arch $(OS_ARCH) 
else
  ifeq ($(OS_SIZE),32)
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m32
  else
      LDFLAGS   := -L$(CUDA_LIB_PATH) -lcudart
      CCFLAGS   := -m64
  endif
endif

# OS-architecture specific flags
ifeq ($(OS_SIZE),32)
      NVCCFLAGS := -m32 -arch=sm_75 --cudart shared
else
      NVCCFLAGS := -lpthread -m64 -arch=sm_75 --cudart shared
endif

dbg := 1 

# Debug build flags
ifeq ($(dbg),1)
      CCFLAGS   += -g
      NVCCFLAGS += -g -G
      TARGET    := debug
else
      TARGET    := release
endif


# Common includes and paths for CUDA
NVCCINCLUDES      := -I$(CUDA_INC_PATH) -I../include2 -I .

################################################################

HEADER_DIR = ../
OBJECT_DIR = objs
TARGET_DIR = .

TARGET_FILE = run
TARGET = $(addprefix $(TARGET_DIR)/, $(TARGET_FILE))

SOURCE_FILES = $(wildcard *.c)
HEADER_FILES = $(wildcard *.h) $(wildcard *.h)
OBJECT_FILE = $(SOURCE_FILES:.c=.o)

OBJECT = $(addprefix $(OBJECT_DIR)/, $(OBJECT_FILE))
BUILD  := ./build

all: start ${BUILD}/KVS_gpu ${BUILD}/KVS_gpm ${BUILD}/KVS_rec

start:
	mkdir -p ${BUILD}

${BUILD}/imkv_emul_gpu: insert_test.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(LIBS) -DNVM_ALLOC_GPU -DEMULATE_NVM

${BUILD}/imkv_real_cpu: insert_test.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(LIBS) -DNVM_ALLOC_CPU -DEMULATE_NVM

${BUILD}/KVS_gpu: insert_test.cu gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_GPU -DCONV_LOG

${BUILD}/KVS_rec: insert_test.cu gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_GPU -DCONV_LOG -DRECOVERY

${BUILD}/KVS_gpm: insert_test.cu gpu_hash.h
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@ $(LIBS) -DNVM_ALLOC_GPU -DCONV_LOG -DGPM_EPOCH_MODEL

${BUILD}/imkv_cpu: insert_test.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(LIBS) -DNVM_ALLOC_CPU -DFAKE_NVM

${BUILD}/imkv_fs_gpu: insert_test_fs.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(LIBS)

${BUILD}/imkv_mm_gpu: insert_test_mm.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(LIBS)

${BUILD}/imkv_mm_tx_gpu: insert_test_mm_tx.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) $^ -o $@ $(LIBS)

$(OBJECT_DIR):
	mkdir $(OBJECT_DIR)

$(OBJECT_DIR)/%.o : %.c
	$(CC) $(CFLAGS) -I $(HEADER_DIR) -I $(CUDA_INC_PATH) -c $< -o $@

$(OBJECT_DIR)/%.o : %.cu
	$(NVCC) $(NVCCFLAGS) $(NVCCINCLUDES) -c $< -o $@

.PHONY : clean
clean:
	rm -rf $(TARGET) $(OBJECT) ./objs/gpu_hash.o
	rm -f ${BUILD}/*
