#include "libgpm.cuh" 
#define DATA int 
#define NUMQ 120
#define ELEMS_PER_Q (1024 * 32)
#define THDS_PER_BLK 128

class Queue {
public: 
	__device__ void init(DATA *pArray) {
		head = 0;
		tail = 0;
		array = pArray;
	}
	__device__ void insert(int *elements, int num_elems);
	__device__ void remove(int *elements, int num_elems);
	__device__ int getSize() { 
		if(tail >= head)
			return tail - head;
		return ELEMS_PER_Q - head + tail;
	}
    int head;
    int tail;
private:
    DATA *array;
};

__device__ void Queue::insert(int *elements, int num_elems) {
	// More elements than queue size. Don't bother performing op
	if(num_elems > ELEMS_PER_Q)
		assert(false);

    __shared__ char done_marker[THDS_PER_BLK];
	int tid = threadIdx.x;
	// Mark not done
	done_marker[tid] = false;
	__syncthreads();
	
	for(int i = tid; i < num_elems; i += blockDim.x) {
		// Wraparound overflow
		if(tail > head && tail + i + 1 > ELEMS_PER_Q && (tail + i + 1) % ELEMS_PER_Q > head) {
			assert(false);
			break;
		}
		
		// Overflow
		if(tail < head && tail + i + 1 > head) {
			assert(false);
			break;
		}
		
		// Insert element
		int ind = (tail + i) % ELEMS_PER_Q;
		array[ind] = elements[i];
	}
	
	// Mark thread done
	release_block(&done_marker[tid], 1);
	
	// Thread 0 to update queue
	if(tid == 0) {
		// Wait for inter-thread order within block
		bool done;
		do {
			done = true;
			for(int i = 0; i < blockDim.x; ++i) {
				char val = acquire_block(&done_marker[i]);
				done &= val;
			}
		} while(!done);
		// Set new val of tail
		tail = (tail + num_elems) % ELEMS_PER_Q;
	}
}

__device__ void Queue::remove(int *elements, int num_elems) {
	// More elements than queue size. Don't bother performing op
	if(num_elems > ELEMS_PER_Q)
		assert(false);

    __shared__ char done_marker[THDS_PER_BLK];
	int tid = threadIdx.x;
	// Mark not done
	done_marker[tid] = false;
	__syncthreads();
	
	for(int i = tid; i < num_elems; i += blockDim.x) {
		// Wraparound overflow
		if(tail < head && head + i + 1 > ELEMS_PER_Q && (head + i + 1) % ELEMS_PER_Q > tail)
			assert(false);
		
		// Overflow
		if(tail > head && head + i + 1 > tail)
			assert(false);
		
		// Remove element
		int ind = (head + i) % ELEMS_PER_Q;
		elements[i] = array[ind];
	}
	// Mark thread done
	release_block(&done_marker[tid], 1);
	
	// Thread 0 to update queue
	if(tid == 0) {
		// Wait for inter-thread order within block
		bool done;
		do {
			done = true;
			for(int i = 0; i < blockDim.x; ++i) {
				char val = acquire_block(&done_marker[i]);
				done &= val;
			}
		} while(!done);
		// Set new val of head
		head = (head + num_elems) % ELEMS_PER_Q;
	}
}

