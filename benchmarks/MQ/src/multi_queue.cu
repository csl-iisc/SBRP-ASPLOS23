#include <iostream>
#include <string>
#include <list>
#include "gpm-helper.cuh"
// Include appropriately scoped header
#ifdef DEVICE_SCOPE
#include "gpu_queue_gpu.h"
#else
#include "gpu_queue.h"
#endif

#define NUM_COMMANDS (2000 + NUMQ)
#define NBLKS NUMQ
#define ll long long

enum command_types {
	TYPE_INSERT,
	TYPE_REMOVE,
	TYPE_COMMIT,
};

struct command {
	command_types type;
	int qNo;
	int numElems;
	DATA *buffer;
};

__host__ __device__ ll packData(int a, int b)
{
	ll a1 = a;
	ll b1 = b;
	return (a1 << (ll)32) | (b1 & (((ll)1 << (ll)32) - (ll)1));
}

__host__ __device__ void extractData(ll data, int &a, int &b)
{
	ll a1 = (data >> 32);
	a = a1;
	ll b1 = (data & (((ll)1 << (ll)32) - (ll)1));
	b = b1;
}

class multiQ {
public:
	__device__ void init(ll *transMgr, DATA *array);
	__device__ void performOps(command commands[NUM_COMMANDS]);
	__device__ void recover(command commands[NUM_COMMANDS]);
private:
	Queue queues[NUMQ];
	ll *transMgr;
};

__device__ void multiQ::init(ll *tMgr = NULL, DATA *array = NULL)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	// Fresh new queues
	if(array != NULL && i < NUMQ) {
		queues[i].init(&array[i * ELEMS_PER_Q]);
	}
	srp_persist();
	if(tMgr != NULL)
		transMgr = tMgr;
	// New set of commands, initiate to 0
	for(; i < NUM_COMMANDS; i += gridDim.x * blockDim.x) {
		transMgr[i] = -1;
		transMgr[NUM_COMMANDS + i] = -1;
	}
	srp_persist();
}

__device__ void multiQ::performOps(command commands[NUM_COMMANDS])
{
	// Begin execution of commands
	for(int i = 0; i < NUM_COMMANDS; ++i) {
		int queue = commands[i].qNo;
		if(blockIdx.x == queue) {
			switch(commands[i].type) {
				case TYPE_INSERT:
					// Mark transaction started
					if(threadIdx.x == 0) {
						transMgr[i] = packData(queues[queue].head, queues[queue].tail);
						srp_order();
					}
					queues[queue].insert(commands[i].buffer, commands[i].numElems);
#if !defined(RECOVERY)
					// Mark transaction complete
					if(threadIdx.x == 0)
						transMgr[NUM_COMMANDS + i] = packData(queues[queue].head, queues[queue].tail);
#endif
					break;
				case TYPE_REMOVE:
					// Mark transaction started
					if(threadIdx.x == 0) {
						transMgr[i] = packData(queues[queue].head, queues[queue].tail);
						srp_order();
					}
					queues[queue].remove(commands[i].buffer, commands[i].numElems);
#if !defined(RECOVERY)
					// Mark transaction complete
					if(threadIdx.x == 0)
						transMgr[NUM_COMMANDS + i] = packData(queues[queue].head, queues[queue].tail);
#endif
					break;
				case TYPE_COMMIT:
					srp_persist();
					break;
			}
		}
	}
}


__device__ void multiQ::recover(command commands[NUM_COMMANDS])
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	// Begin execution of commands
	for(int i = tid; i < NUM_COMMANDS; i += gridDim.x * blockDim.x) {
		int queue = commands[i].qNo;

		switch(commands[i].type) {
			case TYPE_INSERT:
				// Check if transaction already done
				if(transMgr[NUM_COMMANDS + i] != -1) {
					extractData(transMgr[NUM_COMMANDS + i], queues[queue].head, queues[queue].tail);
					continue;
				}
				// Operation started, but crashed before completing
				if(transMgr[i] != -1) {
					extractData(transMgr[i], queues[queue].head, queues[queue].tail);
				}
				break;
			case TYPE_REMOVE:
				// Check if transaction already done
				if(transMgr[NUM_COMMANDS + i] != -1) {
					extractData(transMgr[NUM_COMMANDS + i], queues[queue].head, queues[queue].tail);
					continue;
				}
				// Operation started, but crashed before completing
				if(transMgr[i] != -1) {
					extractData(transMgr[i], queues[queue].head, queues[queue].tail);
				}
				break;
			case TYPE_COMMIT:
				srp_persist();
				break;
		}
	}
	/*__shared__ bool tDone;
	// Begin execution of commands
	for(int i = 0; i < NUM_COMMANDS; ++i) {
		int queue = commands[i].qNo;
		if(blockIdx.x == queue) {
			if(threadIdx.x == 0)
				tDone = false;
			__syncthreads();

			switch(commands[i].type) {
				case TYPE_INSERT:
					// Check if transaction already done
					if(threadIdx.x == 0 && transMgr[NUM_COMMANDS + i] != -1) {
						extractData(transMgr[NUM_COMMANDS + i], queues[queue].head, queues[queue].tail);
						tDone = true;
					}
					__syncthreads();
					if(tDone)
						continue;
					// Operation started, but crashed before completing
					if(threadIdx.x == 0 && transMgr[i] != -1) {
						extractData(transMgr[i], queues[queue].head, queues[queue].tail);
					}
#if !defined(RECOVERY)
					// Mark transaction started
					else if(threadIdx.x == 0) {
						transMgr[i] = packData(queues[queue].head, queues[queue].tail);
						srp_order();
					}
					// Perform transaction
					queues[queue].insert(commands[i].buffer, commands[i].numElems);
					// Mark transaction complete
					if(threadIdx.x == 0)
						transMgr[NUM_COMMANDS + i] = packData(queues[queue].head, queues[queue].tail);
#endif
					break;
				case TYPE_REMOVE:
					// Check if transaction already done
					if(threadIdx.x == 0 && transMgr[NUM_COMMANDS + i] != -1) {
						extractData(transMgr[NUM_COMMANDS + i], queues[queue].head, queues[queue].tail);
						tDone = true;
					}
					__syncthreads();
					if(tDone)
						continue;
					// Operation started, but crashed before completing
					if(threadIdx.x == 0 && transMgr[i] != -1) {
						extractData(transMgr[i], queues[queue].head, queues[queue].tail);
					}
#if !defined(RECOVERY)
					// Mark transaction started
					else if(threadIdx.x == 0) {
						transMgr[i] = packData(queues[queue].head, queues[queue].tail);
						srp_order();
					}
					// Perform transaction
					queues[queue].remove(commands[i].buffer, commands[i].numElems);
					// Mark transaction complete
					if(threadIdx.x == 0)
						transMgr[NUM_COMMANDS + i] = packData(queues[queue].head, queues[queue].tail);
#endif
					break;
				case TYPE_COMMIT:
					srp_persist();
					break;
			}
		}
	}*/
}

__global__ void initKernel(multiQ *mq, char *transMgr = NULL, char *queues = NULL) {
	mq->init((ll*)transMgr, (DATA*)queues);
}

__global__ void mainKernel(multiQ *mq, command commands[NUM_COMMANDS]) {
	mq->performOps(commands);
}


__global__ void recoveryKernel(multiQ *mq, command commands[NUM_COMMANDS]) {
	mq->recover(commands);
}

void createCommands(command *host_commands) {
	// Alternate between inserts and removals
	int type[NUMQ], size[NUMQ];
	memset(type, 0, NUMQ * sizeof(int));
	memset(size, 0, NUMQ * sizeof(int));
	
	srand(100);
	for(int i = 0; i < NUM_COMMANDS - NUMQ; ++i) {
		int qNo = rand() % NUMQ;
		host_commands[i].qNo = qNo;
		
		switch(type[qNo]) {
			case 0: {
					// Insert
					host_commands[i].type = TYPE_INSERT;
					
					int num_elems = 4096;
					host_commands[i].numElems = num_elems;
					size[qNo] += num_elems;
					
					DATA *h_buffer = new DATA[num_elems];
					for(int element = 0; element < num_elems; ++element) {
						h_buffer[element] = rand() % (1<<28);
					}
					
					DATA *d_buffer;
					cudaMalloc((void**)&d_buffer, sizeof(DATA) * num_elems);
					cudaMemcpy(d_buffer, h_buffer, sizeof(DATA) * num_elems, cudaMemcpyHostToDevice);
					host_commands[i].buffer = d_buffer;
					
					delete h_buffer;
					type[qNo] = 1;
				}
				break;
			case 1: {
					// Remove
					host_commands[i].type = TYPE_REMOVE;
					
					int num_elems = 4096;
					host_commands[i].numElems = num_elems;
					size[qNo] -= num_elems;
					
					DATA *d_buffer;
					cudaMalloc((void**)&d_buffer, sizeof(DATA) * num_elems);
					host_commands[i].buffer = d_buffer;
					type[qNo] = 0;
				}
				break;
		}
	}
	// Commit
	for(int i = 0; i < NUMQ; ++i) {
		host_commands[NUM_COMMANDS - NUMQ + i].qNo = i;
		host_commands[NUM_COMMANDS - NUMQ + i].type = TYPE_COMMIT;
	}
}

bool checkCommands(command *host_commands) {
	bool success = true;
	// Alternate between inserts and removals
	std::list<DATA> host_q[NUMQ];
	printf("Testing commands\n");
	for(int i = 0; i < NUM_COMMANDS; ++i) {
		int qNo = host_commands[i].qNo;
		
		switch(host_commands[i].type) {
			case TYPE_INSERT: {
					int num_elems = host_commands[i].numElems;
					
					DATA *h_buffer = new DATA[num_elems];					
					cudaMemcpy(h_buffer, host_commands[i].buffer, sizeof(DATA) * num_elems, cudaMemcpyDeviceToHost);
					for(int element = 0; element < num_elems; ++element)
						host_q[qNo].push_back(h_buffer[element]);
					
					delete h_buffer;
				}
				break;
			case TYPE_REMOVE: {
					int num_elems = host_commands[i].numElems;
					
					DATA *h_buffer = new DATA[num_elems];					
					cudaMemcpy(h_buffer, host_commands[i].buffer, sizeof(DATA) * num_elems, cudaMemcpyDeviceToHost);
					for(int element = 0; element < num_elems; ++element) {
						DATA value = host_q[qNo].front();
						if(value != h_buffer[element]) {
							printf("Command %d, mismatch found %d, expected %d\n", i, h_buffer[element], value);
							success = false;
						}
						host_q[qNo].pop_front();
					}					
					delete h_buffer;
					
				}
				break;
			case TYPE_COMMIT: {
					// Nothing to do for this
				}
				break;
		}
	}
	if(success)
		printf("All success\n");
	return success;
}

int main() {
	size_t tMgrSize = sizeof(ll) * NUM_COMMANDS * 2;
	size_t queueSize = sizeof(DATA) * NUMQ * ELEMS_PER_Q;
	size_t size = tMgrSize + queueSize;
	char *pData = (char *)gpm_map_file("multiqueue.dat", size, 1);
	
	multiQ *mq;
	cudaMalloc((void **)&mq, sizeof(multiQ));
	initKernel<<<NBLKS, THDS_PER_BLK>>>(mq, pData, &pData[tMgrSize]);
	// Create commands on host
	command host_commands[NUM_COMMANDS];
	createCommands(host_commands);
	
	// Copy commands to GPU
	command *device_commands;
	cudaMalloc((void**)&device_commands, sizeof(command) * NUM_COMMANDS);
	cudaMemcpy(device_commands, host_commands, sizeof(command) * NUM_COMMANDS, cudaMemcpyHostToDevice);
	
	mainKernel<<<NBLKS, THDS_PER_BLK>>>(mq, device_commands);
	cudaDeviceSynchronize();
#if defined(RECOVERY) 
    std::cout<<"Recovery begins\n"; 
	recoveryKernel<<<NBLKS, THDS_PER_BLK>>>(mq, device_commands); 
    cudaDeviceSynchronize();
    std::cout<<"Recovery ends\n"; 
#endif 
	
	return !checkCommands(host_commands);
}























