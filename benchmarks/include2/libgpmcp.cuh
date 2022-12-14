#pragma once
#include "libgpm.cuh"
#include <stdio.h>
#include <chrono>

// Non-volatile metadata
struct gpmcp_nv
{
    int elements;        // Maximum number of elements per group
    int partitions;      // Number of groups for the cp
    size_t size;         // Total size of data being cp
};

// Volatile metadata
struct gpmcp
{
    const char *path;    // File path
    
    char *index;         // Set of non-volatile indices
    void *start;         // Pointer to start of non-volatile region
    size_t tot_size;     // Total size, including shadow space and metadata
    int elements;        // Maximum number of elements per group

    gpmcp_nv *cp;        // Pointer to non-volatile metadata

    // Checkpoint entries
    void  **node_addr;   // Set of starting addresses for different elements
    size_t *node_size;   // Set of sizes of each element
    
    // Partition info
    int *part_byte_size; // Set of cp starting addresses for each group
    int *part_bytes;     // Size of contents in partition
    int *part_elem_size; // Set indicating number of elements in each group
};

static __global__ void setup_cp(gpmcp *cp, int size, int elements, int partitions)
{
    cp->cp->elements = elements;
    cp->cp->partitions = partitions;
    cp->cp->size = size;
}

static __global__ void setup_partitions(int *byte_size, int partitions, size_t size)
{
    int ID = threadIdx.x + blockDim.x * blockIdx.x;
    for(int i = ID; i < partitions; i += gridDim.x * blockDim.x) {
        byte_size[i] = i * size / partitions;
    }
}

static __host__ gpmcp *gpmcp_create(const char *path, size_t size, int elements, int partitions)
{
    gpmcp *cp, *copy_cp = new gpmcp;
    cudaMalloc((void **)&cp, sizeof(gpmcp));
    copy_cp->path = path;
    
    // Make all blocks of data equal sizes and 4-byte aligned
    // 4-byte alignment improves checkpoint throughput
    size += partitions - (size % partitions > 0 ? size % partitions : partitions);
    size += (4 - (size / partitions % 4 > 0 ? size / partitions % 4 > 0 : 4 )) * partitions;

    // Header size + location bitmap
    size_t total_size = sizeof(gpmcp) + partitions; 
    // Aligned 2 * Data size (2xsize for crash redundancy)
    total_size += 8 - total_size % 8 + 2 * size;
        
    copy_cp->tot_size = total_size;
    
    // Map file
    char *cp_pointer = (char *)gpm_map_file(path, total_size, 1);
    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);
    
    copy_cp->cp = cp_nv;
    
    
    void **node_addr;
    size_t *node_size;
    cudaMalloc((void **)&node_addr, sizeof(void *) * elements * partitions);
    cudaMalloc((void **)&node_size, sizeof(size_t) * elements * partitions);
    cudaMemset(node_addr, 0, sizeof(void *) * elements * partitions);
    cudaMemset(node_size, 0, sizeof(size_t) * elements * partitions);
    
    copy_cp->node_addr = node_addr;
    copy_cp->node_size = node_size;
    copy_cp->index = (char *)cp_pointer;
    cp_pointer += partitions;
    cp_pointer += 8 - (size_t)cp_pointer % 8;
    copy_cp->start = cp_pointer;
    copy_cp->elements = elements;
    
    cudaMalloc((void **)&copy_cp->part_byte_size, sizeof(int) * partitions);
    cudaMalloc((void **)&copy_cp->part_elem_size, sizeof(int) * partitions);
    cudaMalloc((void **)&copy_cp->part_bytes, sizeof(int) * partitions);
    cudaMemset(copy_cp->part_elem_size, 0, sizeof(int) * partitions);
    cudaMemset(copy_cp->part_bytes, 0, sizeof(int) * partitions);
    setup_partitions <<<(partitions + 1023) / 1024, 1024>>>(copy_cp->part_byte_size, partitions, size);
    
    cudaMemcpy(cp, copy_cp, sizeof(gpmcp), cudaMemcpyHostToDevice);
    delete copy_cp;
    setup_cp<<<1, 1>>>(cp, size, elements, partitions);
    cudaDeviceSynchronize();
    return cp;
}

static __host__ gpmcp *gpmcp_open(const char *path)
{
    gpmcp *cp, *copy_cp = new gpmcp;
    cudaMalloc((void **)&cp, sizeof(gpmcp));
    copy_cp->path = path;
    
    size_t len = 0;
    char *cp_pointer = (char *)gpm_map_file(path, len, false);
    gpmcp_nv *cp_nv = (gpmcp_nv *)cp_pointer;
    cp_pointer += sizeof(gpmcp_nv);
    
    copy_cp->tot_size = len;
    copy_cp->cp = cp_nv;
    
    gpmcp_nv *temp = new gpmcp_nv;
    cudaMemcpy(temp, cp_nv, sizeof(gpmcp_nv), cudaMemcpyDeviceToHost);
    
    void **node_addr;
    size_t *node_size;
    cudaMalloc((void **)&node_addr, sizeof(void *) * temp->elements * temp->partitions);
    cudaMalloc((void **)&node_size, sizeof(size_t) * temp->elements * temp->partitions);
    cudaMemset(node_addr, 0, sizeof(void *) * temp->elements * temp->partitions);
    cudaMemset(node_size, 0, sizeof(size_t) * temp->elements * temp->partitions);
    
    copy_cp->node_addr = node_addr;
    copy_cp->node_size = node_size;
    copy_cp->index = (char *)cp_pointer;
    cp_pointer += temp->partitions;
    cp_pointer += 8 - (size_t)cp_pointer % 8;
    copy_cp->start = cp_pointer;
    copy_cp->elements = temp->elements;
    
    cudaMalloc((void **)&copy_cp->part_byte_size, sizeof(int) * temp->partitions);
    cudaMalloc((void **)&copy_cp->part_elem_size, sizeof(int) * temp->partitions);
    cudaMalloc((void **)&copy_cp->part_bytes, sizeof(int) * temp->partitions);
    cudaMemset(copy_cp->part_elem_size, 0, sizeof(int) * temp->partitions);
    cudaMemset(copy_cp->part_bytes, 0, sizeof(int) * temp->partitions);
    
    int *temp_byte = new int[temp->partitions];
    for(int i = 0; i < temp->partitions; ++i)
        temp_byte[i] = i * temp->size / temp->partitions;
    cudaMemcpy(copy_cp->part_byte_size, temp_byte, sizeof(int) * temp->partitions, cudaMemcpyHostToDevice);
    
    cudaMemcpy(cp, copy_cp, sizeof(gpmcp), cudaMemcpyHostToDevice);
    delete copy_cp;
    return cp;
}

static __host__ void gpmcp_close(gpmcp *cp)
{
    gpmcp temp;
    cudaMemcpy(&temp, cp, sizeof(gpmcp), cudaMemcpyDeviceToHost);
    gpm_unmap(temp.path, temp.cp, temp.tot_size);
    cudaFree(temp.node_addr);
    cudaFree(temp.node_size);
    cudaFree(temp.part_byte_size);
    cudaFree(temp.part_elem_size);
    cudaFree(temp.part_bytes);
    cudaFree(cp);
}

static __device__ __host__ int gpmcp_register(gpmcp *cp, void *addr, size_t size, int partition)
{
    int val = 0;
#if defined(__CUDA_ARCH__)
    int start = cp->part_elem_size[partition];
    if(start >= cp->elements)
        return -1;
    // Device code here
    cp->node_addr[start + cp->elements * partition] = (int *)addr;
    cp->node_size[start + cp->elements * partition] = size;
    cp->part_elem_size[partition]++;
    cp->part_bytes[partition] += size;
    
#else
    gpmcp temp;
    cudaMemcpy(&temp, cp, sizeof(gpmcp), cudaMemcpyDeviceToHost);
    int start;
    cudaMemcpy(&start, &temp.part_elem_size[partition], sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&temp.node_addr[start + temp.elements * partition], &addr, sizeof(int *), cudaMemcpyHostToDevice);
    cudaMemcpy(&temp.node_size[start + temp.elements * partition], &size, sizeof(size_t), cudaMemcpyHostToDevice);
    ++start;
    cudaMemcpy(&temp.part_elem_size[partition], &start, sizeof(int), cudaMemcpyHostToDevice);
    int byte_size;
    cudaMemcpy(&byte_size, &temp.part_bytes[partition], sizeof(int), cudaMemcpyDeviceToHost);
    byte_size += size;
    cudaMemcpy(&temp.part_bytes[partition], &byte_size, sizeof(int), cudaMemcpyHostToDevice);
#endif
    return val;
}

static __global__ void checkpointKernel(gpmcp *cp, int partition)
{
    int offset = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if(offset >= cp->part_bytes[partition])
        return;
    
    size_t start = cp->part_byte_size[partition];
    
    PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
    ind = (ind != 0 ? 0 : 1);
    PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
    start += ind * cp_size;
    
    PMEM_READ_OP( int elems = cp->cp->elements , sizeof(int) )
    
    int curr_offset = 0, curr_elem = 0;
    for(int i = 0; i < elems; ++i) {
        if(curr_offset + cp->node_size[partition * elems + i] > offset)
            break;
        start += cp->node_size[partition * elems + i];
        curr_offset += cp->node_size[partition * elems + i];
        curr_elem++;
    }
        
    void *addr = (int *)cp->node_addr[partition * elems + curr_elem];
    int elem_size = cp->node_size[partition * elems + curr_elem];
    
    if(start >= 2 * cp_size) {
        return;
    }
#if defined(BLK)
    BW_DELAY(CALC(27, 16, min(4, elem_size - (offset - curr_offset))))
#endif
#if defined(LENET)
    BW_DELAY(CALC(33, 22, min(4, elem_size - (offset - curr_offset))))
#endif
#if defined(CFD)
    BW_DELAY(CALC(35, 21, min(4, elem_size - (offset - curr_offset))))
#endif
    gpm_memcpy((char *)cp->start + start + offset - curr_offset, 
        (char *)addr + offset - curr_offset, 
        min(4, elem_size - (offset - curr_offset)), cudaMemcpyDeviceToDevice);
}

static __host__ __device__ int gpmcp_checkpoint(gpmcp *cp, int partition)
{
#if defined(__CUDA_ARCH__)
    // Device code
    size_t start = cp->part_byte_size[partition];
    
    PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
    ind = (ind != 0 ? 0 : 1);
    PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
    start += ind * cp_size;
    
    int elem_size = cp->part_elem_size[partition];
    PMEM_READ_OP( int elems = cp->cp->elements , sizeof(int) )
    for(int i = 0; i < elem_size; ++i)
    {
        if(start >= 2 * cp_size)
            return -1;
        void *addr = (int *)cp->node_addr[partition * elems + i];
        size_t size = cp->node_size[partition * elems + i];
        gpm_memcpy_nodrain((char *)cp->start + start, addr, size, cudaMemcpyDeviceToDevice); 
        start += cp->node_size[partition * elems + i];
    }
    gpm_drain();
    // Update index once complete
    PMEM_READ_WRITE_OP( cp->index[partition] ^= 1; , sizeof(char) )
    gpm_drain();
    return 0;
#else
    gpmcp temp;
    cudaMemcpy(&temp, cp, sizeof(gpmcp), cudaMemcpyDeviceToHost);
    int elem_size;
    cudaMemcpy(&elem_size, &temp.part_bytes[partition], sizeof(int), cudaMemcpyDeviceToHost);
    // Host code
    const int threads = 1024;
    int blocks = (elem_size + threads - 1) / threads;
    // Have each threadblock persist a single element
    // Threads within a threadblock persist at 4-byte offsets
    checkpointKernel<<<blocks, threads>>>(cp, partition);
    cudaDeviceSynchronize();
    // Update index
    char index;
    cudaMemcpy(&index, &temp.index[partition], sizeof(char), cudaMemcpyDeviceToHost);
    gpm_memset(&temp.index[partition], index ^ 1, sizeof(char));
    return 0;
#endif
}

static __device__ int gpmcp_checkpoint_start(gpmcp *cp, int partition, int element, size_t offset, size_t size)
{
    // Device code
    size_t start = cp->part_byte_size[partition];
    
    PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
    ind = (ind != 0 ? 0 : 1);
    PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
    start += ind * cp_size;
    
    PMEM_READ_OP( int elems = cp->cp->elements , sizeof(int) )
    for(int i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];
    
    if(start >= 2 * cp_size)
        return -1;

    void *addr = (char *)cp->node_addr[partition * elems + element] + offset;
    gpm_memcpy((char *)cp->start + start + offset, addr, size, cudaMemcpyDeviceToDevice);
    return 0;
}

static __device__ int gpmcp_checkpoint_value(gpmcp *cp, int partition, int element, size_t offset, size_t size, void *addr)
{
    // Device code
    size_t start = cp->part_byte_size[partition];
    
    PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
    ind = (ind != 0 ? 0 : 1);
    PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
    start += ind * cp_size;
    
    PMEM_READ_OP( int elems = cp->cp->elements , sizeof(int) )
    for(int i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];
    
    if(start >= 2 * cp_size)
        return -1;

    gpm_memcpy((char *)cp->start + start + offset, addr, size, cudaMemcpyDeviceToDevice);
    return 0;
}

static __device__ int gpmcp_checkpoint_finish(gpmcp *cp, int partition)
{
    // Update index once complete
    PMEM_READ_WRITE_OP( cp->index[partition] ^= 1; , sizeof(char) )
    gpm_drain();
    return 0;
}

__global__ void restoreKernel(gpmcp *cp, int partition)
{
    int element = blockIdx.x;
    int offset = threadIdx.x * 4;
    
    size_t start = cp->part_byte_size[partition];
    
    PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
    ind = (ind != 0 ? 1 : 0);
    PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
    start += ind * cp_size;
    
    PMEM_READ_OP( int elems = cp->cp->elements , sizeof(int) )
    
    for(int i = 0; i < element; ++i)
        start += cp->node_size[partition * elems + i];
        
    void *addr = (char *)cp->node_addr[partition * elems + element];
    int elem_size = cp->node_size[partition * elems + element];
    
    if(start >= 2 * cp_size)
        return;
    
    for(int i = offset; i < elem_size; i += blockDim.x * 4) 
        gpm_memcpy_nodrain((char *)addr + i, (char *)cp->start + start + i, 
            min(4, elem_size - i), cudaMemcpyDeviceToDevice);
}

static __host__ __device__ int gpmcp_restore(gpmcp *cp, int partition)
{
#if defined(__CUDA_ARCH__)
    // Device code
    size_t start = cp->part_byte_size[partition];

    PMEM_READ_OP( char ind = cp->index[partition] , sizeof(char) )
    ind = (ind != 0 ? 1 : 0); // Read from last consistent block
    PMEM_READ_OP( size_t cp_size = cp->cp->size , sizeof(size_t) )
    start += ind * cp_size;
    
    int elem_size = cp->part_elem_size[partition];
    for(int i = 0; i < elem_size; ++i)
    {
        if(start >= 2 * cp_size)
            return -1;
        void *addr = cp->node_addr[partition * cp->elements + i];
        size_t size = cp->node_size[partition * cp->elements + i];
        vol_memcpy(addr, (char *)cp->start + start, size); 
        start += cp->node_size[partition * cp->elements + i];
    }
    return 0;
#else
    gpmcp temp;
    cudaMemcpy(&temp, cp, sizeof(gpmcp), cudaMemcpyDeviceToHost);
    int elem_size;
    cudaMemcpy(&elem_size, &temp.part_elem_size[partition], sizeof(int), cudaMemcpyDeviceToHost);
    // Host code
    int blocks = elem_size;
    int threads = 1024;
    // Have each threadblock read a single element
    // Threads within a threadblock read at 4-byte offsets
    restoreKernel<<<blocks, threads>>>(cp, partition);
    //cudaDeviceSynchronize();
    return 0;
#endif
}
