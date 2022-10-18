/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */


#include <assert.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;
//#include <helper_cuda.h>
#include "scan_common.h"
#include "libgpm.cuh"
#include <iostream>

//All three kernels run 512 threads per workgroup
//Must be a power of two
#define THREADBLOCK_SIZE 128
#define COPIES 2

extern "C" void scanExclusiveHost(
    uint *dst,
    uint *src,
    uint batchSize,
    uint arrayLength
)
{
    for (uint i = 0; i < batchSize; i++, src += arrayLength, dst += arrayLength)
    {
        dst[0] = 0;

        for (uint j = 1; j < arrayLength; j++)
            dst[j] = src[j - 1] + dst[j - 1];
    }
}


////////////////////////////////////////////////////////////////////////////////
// Basic scan codelets
////////////////////////////////////////////////////////////////////////////////
//Naive inclusive scan: O(N * log2(N)) operations
//Allocate 2 * 'size' local memory, initialize the first half
//with 'size' zeros avoiding if(pos >= offset) condition evaluation
//and saving instructions
inline __device__ uint scan1Inclusive(uint idata, volatile uint *s_Data, uint size, char *locks)
{
	uint beg = size;
	uint tid = threadIdx.x;
	uint ctr = 0;
	s_Data[tid] = idata;
	release_block(&locks[tid], 0);

    for (ctr = 0; (1 << ctr) < size; ctr++)
    {
    	int offset = (1 << ctr);
    	uint copy = (ctr % COPIES);
		// Read from array
		uint t;
		//if(tid + offset / 2 < THREADBLOCK_SIZE || tid >= offset) {
			t = s_Data[tid + beg * copy];
		    if(tid >= offset) {
		    	while(acquire_block(&locks[tid + size * copy - offset]) != ctr);
				t += s_Data[tid + beg * copy - offset];
				release_block(&locks[tid + size * copy - offset], (1 << 7) | ctr);
			}
	    //}
	    
		//__syncthreads();
		if(tid + offset < THREADBLOCK_SIZE) {
			while(!((acquire_block(&locks[tid + size * copy])) & (1 << 7))); 
		}
		//if(tid + offset / 2 < THREADBLOCK_SIZE || tid >= offset) {
			// Write to copy array
			s_Data[tid + beg * ((copy + 1) % COPIES)] = t;
			// Have thread release
			release_block(&locks[tid + size * ((copy + 1) % COPIES)], ctr + 1);
		//}
    }
	//__syncthreads();

    asm volatile("fence.acq_rel.sys;");
#ifdef GPM_EPOCH_MODEL
	__threadfence(); 
#endif 
    return s_Data[tid + beg * (ctr % COPIES)];
}

// Function to simulate recovery for scan, only simulates UNDO
__global__ void recoveryKernel(volatile uint *gl_Data, uint size, char *gl_locks)
{
	volatile uint *s_Data = &gl_Data[blockIdx.x * (COPIES * THREADBLOCK_SIZE)];
	volatile char *locks = &gl_locks[COPIES * blockDim.x * blockIdx.x];
	uint beg = size;
	uint tid = threadIdx.x;
	// Figure out which iteration we're on
	volatile uint ctr = max(locks[tid] & ((1 << 7) - 1) , locks[tid + size] & ((1 << 7) - 1));
	volatile uint copy = ctr % COPIES;
	// Fetch data for appropriate iteration
	// The increment by 1 is a dummy operation
	s_Data[tid + beg * copy] += 1;
	srp_persist();
	// Begin REDO
	// ...
}

inline __device__ uint scan1Exclusive(uint idata, volatile uint *s_Data, uint size, char *locks)
{
    return scan1Inclusive(idata, s_Data, size, locks) - idata;
}


inline __device__ uint4 scan4Inclusive(uint4 idata4, volatile uint *s_Data, uint size, char *locks)
{
    //Level-0 inclusive scan
    idata4.y += idata4.x;
    idata4.z += idata4.y;
    idata4.w += idata4.z;

    //Level-1 exclusive scan
    uint oval = scan1Exclusive(idata4.w, s_Data, size / 4, locks);

    idata4.x += oval;
    idata4.y += oval;
    idata4.z += oval;
    idata4.w += oval;

    return idata4;
}

//Exclusive vector scan: the array to be scanned is stored
//in local thread memory scope as uint4
inline __device__ uint4 scan4Exclusive(uint4 idata4, volatile uint *s_Data, uint size, char *locks)
{
    uint4 odata4 = scan4Inclusive(idata4, s_Data, size, locks);
    odata4.x -= idata4.x;
    odata4.y -= idata4.y;
    odata4.z -= idata4.z;
    odata4.w -= idata4.w;
    return odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Scan kernels
////////////////////////////////////////////////////////////////////////////////
__global__ void scanExclusiveShared(
    uint4 *d_Dst,
    uint4 *d_Src,
    uint  *s_Data,
    char  *locks,
    uint size
)
{
    //__shared__ uint s_Data[2 * THREADBLOCK_SIZE];

    uint pos = blockIdx.x * blockDim.x + threadIdx.x;

    //Load data
    uint4 idata4 = d_Src[pos];

    //Calculate exclusive scan
    uint4 odata4 = scan4Exclusive(idata4, &s_Data[blockIdx.x * (COPIES * THREADBLOCK_SIZE)], size, &locks[COPIES * blockDim.x * blockIdx.x]);

    //Write back
    d_Dst[pos] = odata4;
}

////////////////////////////////////////////////////////////////////////////////
// Interface function
////////////////////////////////////////////////////////////////////////////////
//Derived as 32768 (max power-of-two gridDim.x) * 4 * THREADBLOCK_SIZE
//Due to scanExclusiveShared<<<>>>() 1D block addressing
extern "C" const uint MAX_BATCH_ELEMENTS = 64 * 1048576;
extern "C" const uint MIN_SHORT_ARRAY_SIZE = 4;
extern "C" const uint MAX_SHORT_ARRAY_SIZE = 4 * THREADBLOCK_SIZE;
extern "C" const uint MIN_LARGE_ARRAY_SIZE = 8 * THREADBLOCK_SIZE;
extern "C" const uint MAX_LARGE_ARRAY_SIZE = 4 * THREADBLOCK_SIZE * THREADBLOCK_SIZE;

//Internal exclusive scan buffer
static uint *d_Buf;

extern "C" void initScan(void)
{
    checkCudaErrors(cudaMalloc((void **)&d_Buf, (MAX_BATCH_ELEMENTS / (4 * THREADBLOCK_SIZE)) * sizeof(uint)));
}

extern "C" void closeScan(void)
{
    checkCudaErrors(cudaFree(d_Buf));
}

static uint factorRadix2(uint &log2L, uint L)
{
    if (!L)
    {
        log2L = 0;
        return 0;
    }
    else
    {
        for (log2L = 0; (L & 1) == 0; L >>= 1, log2L++);

        return L;
    }
}

static uint iDivUp(uint dividend, uint divisor)
{
    return ((dividend % divisor) == 0) ? (dividend / divisor) : (dividend / divisor + 1);
}

extern "C" size_t scanExclusiveShort(
    uint *d_Dst,
    uint *d_Src,
    uint batchSize,
    uint arrayLength
)
{
    //Check power-of-two factorization
    uint log2L;
    uint factorizationRemainder = factorRadix2(log2L, arrayLength);
    assert(factorizationRemainder == 1);

    //Check supported size range
    assert((arrayLength >= MIN_SHORT_ARRAY_SIZE) && (arrayLength <= MAX_SHORT_ARRAY_SIZE));

    //Check total batch size limit
    assert((batchSize * arrayLength) <= MAX_BATCH_ELEMENTS);

    //Check all threadblocks to be fully packed with data
    assert((batchSize * arrayLength) % (4 * THREADBLOCK_SIZE) == 0);
    
    uint *s_Data;
    size_t dummy;
    uint numBlocks = (batchSize * arrayLength) / (4 * THREADBLOCK_SIZE);
    uint numThreads = THREADBLOCK_SIZE;
    char *d_locks;
    checkCudaErrors(cudaMallocPitch((void **) &s_Data, &dummy, 1, sizeof(uint) * numBlocks * (numThreads * COPIES)));
    checkCudaErrors(cudaMemset(s_Data, -1, sizeof(uint) * numBlocks * (numThreads * COPIES)));
    checkCudaErrors(cudaMallocPitch((void **) &d_locks, &dummy, 1, COPIES * sizeof(char) * numBlocks * numThreads));
    checkCudaErrors(cudaMemset(d_locks, 0, COPIES * sizeof(char) * numBlocks * numThreads));

    scanExclusiveShared<<<numBlocks, numThreads>>>(
        (uint4 *)d_Dst,
        (uint4 *)d_Src,
        s_Data,
        d_locks,
        arrayLength
    );
    getLastCudaError("scanExclusiveShared() execution FAILED\n");
#ifdef RECOVERY
    std::cout<<"Recovery begins\n"; 
	recoveryKernel<<<numBlocks, numThreads>>>(s_Data, arrayLength, d_locks);
    cudaDeviceSynchronize();
    std::cout<<"Recovery ends\n"; 
#endif
    return THREADBLOCK_SIZE;
}

int main(int argc, char **argv)
{
    printf("%s Starting...\n\n", argv[0]);

    //Use command-line specified CUDA device, otherwise use device with highest Gflops/s
    //findCudaDevice(argc, (const char **)argv);

    uint *d_Input, *d_Output;
    uint *h_Input, *h_OutputCPU, *h_OutputGPU;
    //StopWatchInterface  *hTimer = NULL;
    const uint N = /*13 */ 1048576 / 32 / 2 / 4 * 30;

    printf("Allocating and initializing host arrays...\n");
    //sdkCreateTimer(&hTimer);
    h_Input     = (uint *)malloc(N * sizeof(uint));
    h_OutputCPU = (uint *)malloc(N * sizeof(uint));
    h_OutputGPU = (uint *)malloc(N * sizeof(uint));
    srand(2009);

    for (uint i = 0; i < N; i++)
    {
        h_Input[i] = 1;//rand();
    }

    printf("Allocating and initializing CUDA arrays...\n");
    checkCudaErrors(cudaMalloc((void **)&d_Input, N * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&d_Output, N * sizeof(uint)));
    checkCudaErrors(cudaMemcpy(d_Input, h_Input, N * sizeof(uint), cudaMemcpyHostToDevice));

    printf("Initializing CUDA-C scan...\n\n");
    initScan();

    int globalFlag = 1;
    size_t szWorkgroup;
    const int iCycles = 1;
    printf("*** Running GPU scan for short arrays (%d identical iterations)...\n\n", iCycles);

    //for (uint arrayLength = MIN_SHORT_ARRAY_SIZE; arrayLength <= MAX_SHORT_ARRAY_SIZE; arrayLength <<= 1)
    uint arrayLength = MAX_SHORT_ARRAY_SIZE;
    printf("Running scan for %u elements (%u arrays)...\n", arrayLength, N / arrayLength);
    checkCudaErrors(cudaDeviceSynchronize());
    
    szWorkgroup = scanExclusiveShort(d_Output, d_Input, N / arrayLength, arrayLength);
    checkCudaErrors(cudaDeviceSynchronize());

    printf("Validating the results...\n");
    printf("...reading back GPU results\n");
    checkCudaErrors(cudaMemcpy(h_OutputGPU, d_Output, N * sizeof(uint), cudaMemcpyDeviceToHost));

    printf(" ...scanExclusiveHost()\n");
    scanExclusiveHost(h_OutputCPU, h_Input, N / arrayLength, arrayLength);
#ifndef RECOVER
    // Compare GPU results with CPU results and accumulate error for this test
    printf(" ...comparing the results\n");
    int localFlag = 1;

    for (uint i = 0; i < N; i++)
    {
        if (h_OutputCPU[i] != h_OutputGPU[i])
        {
            localFlag = 0;
            break;
        }
    }

    // Log message on individual test result, then accumulate to global flag
    printf(" ...Results %s\n\n", (localFlag == 1) ? "Match" : "DON'T Match !!!");
    if(!localFlag)
    	assert(false);
    globalFlag = globalFlag && localFlag;
    printf("Shutting down...\n");
    closeScan();
    checkCudaErrors(cudaFree(d_Output));
    checkCudaErrors(cudaFree(d_Input));

    // pass or fail (cumulative... all tests in the loop)
    exit(globalFlag ? EXIT_SUCCESS : EXIT_FAILURE);
#endif
}
