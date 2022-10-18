 /********************************************************************************************
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * This software contains source code provided by NVIDIA Corporation.
 ********************************************************************************************/
 /******************************************************************************************** 
 * Implementation of Reduction
 *
 * Edited by: 
 * Aditya K Kamath, Indian Institute of Science
 *
 * Each block contains a subarray which it is responsible for reducing using block scope.
 * Original code has been modified to use atomics and threadfences instead of syncthreads.
 * 
 ********************************************************************************************/
/*
    Parallel reduction kernels
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <fstream>
#include <chrono>
using namespace std;

#define DATATYPE int
#define NBLOCKS 60
#define NTHREADS 512

void checkCudaErrors(cudaError_t err)
{
    if (err != cudaSuccess)
    {
        cout << "Error (" << err <<"): " << cudaGetErrorString(err) << "\n";
        exit(1);
    }
}


__device__ unsigned int retirementCount = 0;
/*
    Parallel sum reduction using shared memory
    - takes log(n) steps for n input elements
    - uses n/2 threads

    This version adds multiple elements per thread sequentially.  This reduces the overall
    cost of the algorithm while keeping the work complexity O(n) and the step complexity O(log n).
    (Brent's Theorem optimization)
*/

template <unsigned int blockSize>
__device__ void
reduceBlock(volatile DATATYPE *sdata, DATATYPE mySum, const unsigned int tid, bool recover)
{
	__shared__ volatile char lock[blockSize];
	if(recover && sdata[tid] != 0) {
		lock[tid] = 1;
		return;
	}
    if(blockSize > 1 && tid >= blockSize / 2)
    	sdata[tid] = mySum;
    lock[tid] = 0;
    __threadfence();
    __syncthreads();

    // do reduction in shared mem
    if (blockSize >= 512 && tid < 256)
    {
        if (tid >= 128)
        {
            sdata[tid] = mySum + sdata[tid + 256];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 128)
        {
            mySum = mySum + sdata[tid + 256];
        }
    }

    if (blockSize >= 256 && tid < 128)
    {
    	int val = 0;
        while(blockSize >= 512 && val != 1) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 128]));
	}
        
        if(tid >= 64)
        {
            sdata[tid] = mySum + sdata[tid + 128];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 64)
        {
            mySum = mySum + sdata[tid + 128];
        }
    }

    if (blockSize >= 128 && tid < 64)
    {
    	int val = 0;
        while(blockSize >= 256 && val != 1) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 64]));
	}
    		
        if (tid >= 32)
        {
            sdata[tid] = mySum + sdata[tid +  64];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 32)
        {
            mySum = mySum + sdata[tid + 64];
        }
    }

    if (tid < 32)
    {
    	int val = 0;
        while(blockSize >= 128 && val != 1) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 32]));
	}
        
        if (blockSize >=  64 && tid >= 16 && tid < 32)
        {
            sdata[tid] = mySum + sdata[tid + 32];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 16)
        {
            mySum = mySum + sdata[tid + 32];
        }
		
		val = 0;
        while(blockSize >= 64 && val != 1 && tid < 16) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 16]));
	}
        if (blockSize >= 32 && tid >= 8 && tid < 16)
        {
            sdata[tid] = mySum + sdata[tid + 16];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 8)
        {
            mySum = mySum + sdata[tid + 16];
        }
		
		val = 0;
        while(blockSize >= 32 && val != 1 && tid < 8) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 8]));
	}
        if (blockSize >=  16 && tid >= 4 && tid < 8)
        {
            sdata[tid] = mySum + sdata[tid +  8];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 4)
        {
        	mySum = mySum + sdata[tid +  8];
        }
		
		val = 0;
        while(blockSize >= 16 && val != 1 && tid < 4) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 4]));
	}
        if (blockSize >=   8 && tid >= 2 && tid < 4)
        {
            sdata[tid] = mySum + sdata[tid +  4];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid < 2)
        {
            mySum = mySum + sdata[tid +  4];
        }

		val = 0;
        while(blockSize >= 8 && val != 1 && tid < 2) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 2]));
	}
        if (blockSize >=   4 && tid == 1)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 2];
            asm volatile("st.relaxed.gpu.u8 [%0], 1;":: "l"(&lock[tid]));
        }
        else if(tid == 0)
        {
            mySum = mySum + sdata[tid +  2];        	
        }

		val = 0;
        while(blockSize >= 4 && val != 1 && tid == 0) {
    		asm volatile("ld.relaxed.gpu.u8 %0, [%1];" : "=r"(val) : "l"(&lock[tid + 1]));
	}
        if (blockSize >=   2 && tid == 0)
        {
            sdata[tid] = mySum = mySum + sdata[tid + 1];
        }
        asm volatile("fence.acq_rel.sys;");
    }
}

template <unsigned int blockSize, bool nIsPow2>
__device__ void
reduceBlocks(const DATATYPE *g_idata, volatile DATATYPE *g_odata, volatile DATATYPE *sdata, unsigned int n, bool recover)
{
    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockSize*2) + threadIdx.x;
    unsigned int gridSize = blockSize*2*gridDim.x;
    DATATYPE mySum = 0;

    // we reduce multiple elements per thread.  The number is determined by the
    // number of active thread blocks (via gridDim).  More blocks will result
    // in a larger gridSize and therefore fewer elements per thread
    while (i < n)
    {
        mySum += g_idata[i];

        // ensure we don't read out of bounds -- this is optimized away for powerOf2 sized arrays
        if (nIsPow2 || i + blockSize < n)
            mySum += g_idata[i+blockSize];

        i += gridSize;
    }
    // do reduction in shared mem
    reduceBlock<blockSize>(sdata, mySum, tid, recover);
    // write result for this block to global mem
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}

// This reduction kernel reduces an arbitrary size array in a single kernel invocation
// It does so by keeping track of how many blocks have finished.  After each thread
// block completes the reduction of its own block of data, it "takes a ticket" by
// atomically incrementing a global counter.  If the ticket value is equal to the number
// of thread blocks, then the block holding the ticket knows that it is the last block
// to finish.  This last block is responsible for summing the results of all the other
// blocks.
//
// In order for this to work, we must be sure that before a block takes a ticket, all
// of its memory transactions have completed.  This is what __threadfence() does -- it
// blocks until the results of all outstanding memory transactions within the
// calling thread are visible to all other threads.
template <unsigned int blockSize, bool nIsPow2>
__global__ void reduceSinglePass(const DATATYPE *g_idata, volatile DATATYPE *g_odata, volatile DATATYPE *sdata, unsigned int n, bool recover)
{
    __shared__ bool amLast;
    //
    // PHASE 1: Process all inputs assigned to this block
    //

    reduceBlocks<blockSize, nIsPow2>(g_idata, g_odata, &sdata[blockIdx.x * blockSize], n, recover);

    //
    // PHASE 2: Last block finished will process all partial sums
    //

    if (gridDim.x > 1)
    {
        const unsigned int tid = threadIdx.x;

        // wait until all outstanding memory instructions in this thread are finished
        asm volatile("st.relaxed.gpu.s32 [%0], %1;":: "l"(&g_odata[blockIdx.x]), "r"(g_odata[blockIdx.x]));
		
        // Thread 0 takes a ticket
        if (tid==0)
        {
            unsigned int ticket = atomicInc(&retirementCount, gridDim.x);
            // If the ticket ID is equal to the number of blocks, we are the last block!
            amLast = (ticket == gridDim.x-1);
        }

        __syncthreads();

        // The last block sums the results of all other blocks
        if (amLast)
        {
            int i = tid;
            DATATYPE mySum = 0;

            while (i < gridDim.x)
            {
            	int val;
        		asm volatile("ld.relaxed.gpu.s32 %0, [%1];": "=r"(val) : "l"(&g_odata[i]));
                mySum += val;
                i += blockSize;
            }

            reduceBlock<blockSize>(&sdata[blockDim.x * blockSize], mySum, tid, recover);

            if (tid==0)
            {
                g_odata[gridDim.x] = sdata[blockDim.x * blockSize];
                // reset retirement count so that next run succeeds
                retirementCount = 0;
            }
        }
    }
}

bool isPow2(unsigned int x)
{
    return ((x&(x-1))==0);
}

void reduceSinglePasses(int size, DATATYPE *d_idata, DATATYPE *d_odata, int nThreads, int nBlocks, DATATYPE *smem, bool recover = false)
{
    dim3 dimBlock(nThreads, 1, 1);
    dim3 dimGrid(nBlocks, 1, 1);
      
    // choose which of the optimized versions of reduction to launch
    if (isPow2(size))
    {
        switch (nThreads)
        {
            case 512:
                reduceSinglePass<512, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 256:
                reduceSinglePass<256, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 128:
                reduceSinglePass<128, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 64:
                reduceSinglePass< 64, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 32:
                reduceSinglePass< 32, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 16:
                reduceSinglePass< 16, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  8:
                reduceSinglePass<  8, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  4:
                reduceSinglePass<  4, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  2:
                reduceSinglePass<  2, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  1:
                reduceSinglePass<  1, true><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;
        }
    }
    else
    {
        switch (nThreads)
        {
            case 512:
                reduceSinglePass<512, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 256:
                reduceSinglePass<256, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 128:
                reduceSinglePass<128, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 64:
                reduceSinglePass< 64, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 32:
                reduceSinglePass< 32, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case 16:
                reduceSinglePass< 16, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  8:
                reduceSinglePass<  8, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  4:
                reduceSinglePass<  4, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  2:
                reduceSinglePass<  2, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;

            case  1:
                reduceSinglePass<  1, false><<< dimGrid, dimBlock >>>(d_idata, d_odata, smem, size, recover);
                break;
        }
    }
}

unsigned int nextPow2(unsigned int x)
{
    --x;
    x |= x >> 1;
    x |= x >> 2;
    x |= x >> 4;
    x |= x >> 8;
    x |= x >> 16;
    return ++x;
}


////////////////////////////////////////////////////////////////////////////////
// Compute the number of threads and blocks to use for the reduction
// We set threads / block to the minimum of maxThreads and n/2.
////////////////////////////////////////////////////////////////////////////////
void getNumBlocksAndThreads(int n, int maxBlocks, int maxThreads, int &blocks, int &threads)
{
    if (n == 1)
    {
        threads = 1;
        blocks = 1;
    }
    else
    {
        threads = (n < maxThreads*2) ? nextPow2(n / 2) : maxThreads;
        blocks = max(1, n / (threads * 2));
    }

    blocks = min(maxBlocks, blocks);
}

////////////////////////////////////////////////////////////////////////////////
// The main function which runs the reduction test.
////////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
{
    int size = 4 * 1024 * 1024; // number of elements to reduce
    int iterations = 16;
    int step_size = size / iterations;
    
    DATATYPE *h_idata = (DATATYPE *)malloc(size * sizeof(DATATYPE));

    for(int i = 0; i < size; ++i) {
        h_idata[i] = rand() % (INT_MAX / size);
        //cin >> h_idata[i];
    }

    int nBlocks, nThreads;
    getNumBlocksAndThreads(size, NBLOCKS, NTHREADS, nBlocks, nThreads);

    // allocate mem for the result on host side
    DATATYPE *h_odata = (DATATYPE *) malloc(nBlocks*sizeof(DATATYPE));

    // allocate device memory and data
    int *d_idata = NULL;
    int *d_odata = NULL;

    checkCudaErrors(cudaMalloc((void **) &d_idata, size * sizeof(DATATYPE)));
    size_t dummy;
    checkCudaErrors(cudaMallocPitch((void **) &d_odata, &dummy, 1, (nBlocks + 1)*sizeof(DATATYPE)));
    
    int smemSize = nThreads * (nBlocks + 1);
    DATATYPE *d_smem;
    checkCudaErrors(cudaMallocPitch((void **) &d_smem, &dummy, 1, iterations * smemSize * sizeof(DATATYPE)));
    checkCudaErrors(cudaMemset(d_smem, 0, iterations * smemSize * sizeof(DATATYPE)));

    // copy data directly to device memory
    checkCudaErrors(cudaMemcpy(d_idata, h_idata, size * sizeof(DATATYPE), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_odata, h_idata, nBlocks * sizeof(DATATYPE), cudaMemcpyHostToDevice));
	
    DATATYPE gpu_result = 0;
	// execute the kernel
	for(int i = 0; i < iterations; ++i) {
    	reduceSinglePasses(step_size, &d_idata[step_size * i], d_odata, nThreads, nBlocks, &d_smem[smemSize * i]);
    	DATATYPE result;
    	checkCudaErrors(cudaMemcpy(&result, &d_odata[nBlocks], sizeof(DATATYPE), cudaMemcpyDeviceToHost));
    	gpu_result += result;
   	}
    // copy final sum from device to host
    cout << "GPU result = " << gpu_result << "\n";
    int sum = 0;
    for(int i = 0; i < size; ++i)
		sum += h_idata[i];
	if(sum != gpu_result) {
		printf("Result mismatch. Expected %d, found %d\n", sum, gpu_result);
		return -1;
	} else
		printf("Results match.\n");
    free(h_idata);
    free(h_odata);
    cudaFree(d_idata);
    cudaFree(d_odata);
    return 0;
}
