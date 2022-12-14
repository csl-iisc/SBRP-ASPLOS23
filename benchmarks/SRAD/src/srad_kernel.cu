#include "srad.h"
#include <stdio.h>

__device__ char lock; 

// Kernel for just the "UNDO" portion of recovery. 
// Full recovery (UNDO + REDO) is encapsulated in main kernel
__global__ void 
recovery_kernel(
	float *C_cuda, 
	float *J_cuda_out,
	volatile int *kernel_start,
	long long cols, 
	long long rows) 
{
    //block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //indices
    long index  = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    if (C_cuda[index] < 0) {
		*kernel_start = 0;
		return; 
	}
    else if (J_cuda_out[index] < 0 && *kernel_start > 1) {
    	atomicMin((int*)kernel_start, 1);
	}
}


    __global__ void
srad_cuda_1(
        float *E_C, 
        float *W_C, 
        float *N_C, 
        float *S_C,
        float * J_cuda, 
        float * C_cuda, 
        long long cols, 
        long long rows, 
        float q0sqr
        ) 
{
    int val = 0; 
    asm volatile("ld.relaxed.sys.u8 %0, [%1];" : "=r"(val) : "l"(&lock));
    //block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //indices
    long index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    if (index < cols * rows) {
        long long index_n = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + tx - cols;
        long long index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
        long long index_w = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty - 1;
        long long index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;

        float n, w, e, s, jc, g2, l, num, den, qsqr, c;

        //shared memory allocation
        __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp_result[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float north[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float south[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float  east[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float  west[BLOCK_SIZE][BLOCK_SIZE];

        //load data to shared memory
        if ( by == 0 ){
            north[ty][tx] = J_cuda[BLOCK_SIZE * bx + tx]; 
            south[ty][tx] = J_cuda[index_s];
        }
        else if ( by == gridDim.y - 1 ){
            north[ty][tx] = J_cuda[index_n]; 
            south[ty][tx] = J_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx];
        }
        else {
            north[ty][tx] = J_cuda[index_n]; 
            south[ty][tx] = J_cuda[index_s];
        }
        __syncthreads();

        if ( bx == 0 ){
            west[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + cols * ty];
            east[ty][tx] = J_cuda[index_e]; 
        }
        else if ( bx == gridDim.x - 1 ){
            west[ty][tx] = J_cuda[index_w];
            east[ty][tx] = J_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1];
        }
        else {
            west[ty][tx] = J_cuda[index_w];
            east[ty][tx] = J_cuda[index_e];
        }

        __syncthreads();

        temp[ty][tx] = J_cuda[index];

        __syncthreads();
        
        PMEM_READ_OP( , sizeof(float)) // For the if condition
        // If calculated, skip
        if(C_cuda[index] < 0) { // For recovery. No need to recompute if value exists
            jc = temp[ty][tx];

            if ( ty == 0 && tx == 0 ){ //nw
                n  = north[ty][tx] - jc;
                s  = temp[ty+1][tx] - jc;
                w  = west[ty][tx]  - jc; 
                e  = temp[ty][tx+1] - jc;
            }	    
            else if ( ty == 0 && tx == BLOCK_SIZE-1 ){ //ne
                n  = north[ty][tx] - jc;
                s  = temp[ty+1][tx] - jc;
                w  = temp[ty][tx-1] - jc; 
                e  = east[ty][tx] - jc;
            }
            else if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
                n  = temp[ty-1][tx] - jc;
                s  = south[ty][tx] - jc;
                w  = temp[ty][tx-1] - jc; 
                e  = east[ty][tx]  - jc;
            }
            else if ( ty == BLOCK_SIZE -1 && tx == 0 ){//sw
                n  = temp[ty-1][tx] - jc;
                s  = south[ty][tx] - jc;
                w  = west[ty][tx]  - jc; 
                e  = temp[ty][tx+1] - jc;
            }

            else if ( ty == 0 ){ //n
                n  = north[ty][tx] - jc;
                s  = temp[ty+1][tx] - jc;
                w  = temp[ty][tx-1] - jc; 
                e  = temp[ty][tx+1] - jc;
            }
            else if ( tx == BLOCK_SIZE -1 ){ //e
                n  = temp[ty-1][tx] - jc;
                s  = temp[ty+1][tx] - jc;
                w  = temp[ty][tx-1] - jc; 
                e  = east[ty][tx] - jc;
            }
            else if ( ty == BLOCK_SIZE -1){ //s
                n  = temp[ty-1][tx] - jc;
                s  = south[ty][tx] - jc;
                w  = temp[ty][tx-1] - jc; 
                e  = temp[ty][tx+1] - jc;
            }
            else if ( tx == 0 ){ //w
                n  = temp[ty-1][tx] - jc;
                s  = temp[ty+1][tx] - jc;
                w  = west[ty][tx] - jc; 
                e  = temp[ty][tx+1] - jc;
            }
            else{  //the data elements which are not on the borders 
                n  = temp[ty-1][tx] - jc;
                s  = temp[ty+1][tx] - jc;
                w  = temp[ty][tx-1] - jc; 
                e  = temp[ty][tx+1] - jc;
            }


            g2 = ( n * n + s * s + w * w + e * e ) / (jc * jc);

            l = ( n + s + w + e ) / jc;

            num  = (0.5*g2) - ((1.0/16.0)*(l*l)) ;
            den  = 1 + (.25*l);
            qsqr = num/(den*den);

            // diffusion coefficent (equ 33)
            den = (qsqr-q0sqr) / (q0sqr * (1+q0sqr)) ;
            c = 1.0 / (1.0+den) ;

            // saturate diffusion coefficent
            if (c < 0){temp_result[ty][tx] = 0;}
            else if (c > 1) {temp_result[ty][tx] = 1;}
            else {temp_result[ty][tx] = c;}

            //__syncthreads(); // Seems unneeded?

            gpm_memcpy_nodrain(E_C + index, &e, sizeof(float), cudaMemcpyDeviceToDevice);
            gpm_memcpy_nodrain(W_C + index, &w, sizeof(float), cudaMemcpyDeviceToDevice);
            gpm_memcpy_nodrain(S_C + index, &s, sizeof(float), cudaMemcpyDeviceToDevice);
            gpm_memcpy_nodrain(N_C + index, &n, sizeof(float), cudaMemcpyDeviceToDevice);
            srp_order();
#ifdef GPM_EPOCH_MODEL
	    __threadfence(); 
#endif 
            gpm_memcpy_nodrain(C_cuda + index, &temp_result[ty][tx], sizeof(float), cudaMemcpyDeviceToDevice);
        }
    }
}

    __global__ void
srad_cuda_2(
        float *E_C, 
        float *W_C, 
        float *N_C, 
        float *S_C,	
        float * J_cuda, 
        float * C_cuda, 
        long long cols, 
        long long rows, 
        float lambda,
        float q0sqr,
        float * J_cuda_out 
        ) 
{
    //block id
    int bx = blockIdx.x;
    int by = blockIdx.y;

    //thread id
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //indices
    long long index   = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + tx;
    if (index < cols * rows)
    {
        long long index_s = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * BLOCK_SIZE + tx;
        long long index_e = cols * BLOCK_SIZE * by + BLOCK_SIZE * bx + cols * ty + BLOCK_SIZE;
        float cc, cn, cs, ce, cw, d_sum;

        //shared memory allocation
        __shared__ float south_c[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float  east_c[BLOCK_SIZE][BLOCK_SIZE];

        __shared__ float c_cuda_temp[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float c_cuda_result[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float temp[BLOCK_SIZE][BLOCK_SIZE];

        //load data to shared memory
        temp[ty][tx] = J_cuda[index];

        __syncthreads();

        if ( by == gridDim.y - 1 ){
            PMEM_READ_OP(south_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * (gridDim.y - 1) + BLOCK_SIZE * bx + cols * ( BLOCK_SIZE - 1 ) + tx], sizeof(float))
        }
        else {
            PMEM_READ_OP(south_c[ty][tx] = C_cuda[index_s], sizeof(float))
        }
        __syncthreads();

        if ( bx == gridDim.x - 1 ){
            PMEM_READ_OP(east_c[ty][tx] = C_cuda[cols * BLOCK_SIZE * by + BLOCK_SIZE * ( gridDim.x - 1) + cols * ty + BLOCK_SIZE-1], sizeof(float))
        }
        else {
            PMEM_READ_OP(east_c[ty][tx] = C_cuda[index_e], sizeof(float))
        }

        __syncthreads();

        PMEM_READ_OP(c_cuda_temp[ty][tx] = C_cuda[index], sizeof(float))

        __syncthreads();
        PMEM_READ_OP( , sizeof(float)) // For the if condition
        // Skip if already calculated
        if(J_cuda_out[index] < 0) {
            cc = c_cuda_temp[ty][tx];

            if ( ty == BLOCK_SIZE -1 && tx == BLOCK_SIZE - 1){ //se
                cn  = cc;
                cs  = south_c[ty][tx];
                cw  = cc; 
                ce  = east_c[ty][tx];
            } 
            else if ( tx == BLOCK_SIZE -1 ){ //e
                cn  = cc;
                cs  = c_cuda_temp[ty+1][tx];
                cw  = cc; 
                ce  = east_c[ty][tx];
            }
            else if ( ty == BLOCK_SIZE -1){ //s
                cn  = cc;
                cs  = south_c[ty][tx];
                cw  = cc; 
                ce  = c_cuda_temp[ty][tx+1];
            }
            else{ //the data elements which are not on the borders 
                cn  = cc;
                cs  = c_cuda_temp[ty+1][tx];
                cw  = cc; 
                ce  = c_cuda_temp[ty][tx+1];
            }

            // divergence (equ 58)
            PMEM_READ_OP(d_sum = (cn * N_C[index] + cs * S_C[index] + cw * W_C[index] + ce * E_C[index]), 4* sizeof(float))

            // image update (equ 61)
            c_cuda_result[ty][tx] = temp[ty][tx] + 0.25 * lambda * d_sum;

            //__syncthreads(); // Seems unneeded?

            gpm_memcpy_nodrain(&J_cuda_out[index], &c_cuda_result[ty][tx], sizeof(float), cudaMemcpyDeviceToDevice);
    		srp_persist();
#ifdef GPM_EPOCH_MODEL
	    __threadfence(); 
#endif 
        }
    }
}
