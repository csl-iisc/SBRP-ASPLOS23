//#define EMULATE_NVM_BW
#include <thrust/random.h> 
#include <thrust/device_vector.h> 
#include <curand.h> 
#include <curand_kernel.h> 
#include "libgpm.cuh"
#include "libgpmlog.cuh"
//#include "bandwidth_analysis.cuh"

#define BUCKET_SIZE 576
#define MAX_BUCKETS 300
#define MAX_KEYS ((size_t)MAX_BUCKETS * BUCKET_SIZE)
#define DATATYPE int

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
 inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
 {
    if (code != cudaSuccess)
    {
       fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
       if (abort) exit(code);
    }
 }


struct RandomGenerator
{
    int count; 
    __host__ __device__ RandomGenerator() : count(0) {}

    __host__ __device__ unsigned int hashCode(unsigned int a) 
    {
        a = (a+0x7ed55d16) + (a<<12);
        a = (a^0xc761c23c) ^ (a>>19);
        a = (a+0x165667b1) + (a<<5);
        a = (a+0xd3a2646c) ^ (a<<9);
        a = (a+0xfd7046c5) + (a<<3);
        a = (a^0xb55a4f09) ^ (a>>16);
        return a; 
    }

    __host__ __device__ int/*thrust::tuple<float, float>*/ operator()()
    {
        unsigned seed = hashCode(blockIdx.x * blockDim.x + threadIdx.x + count);
        count += blockDim.x * gridDim.x;
        thrust::default_random_engine rng(seed);
        thrust::random::uniform_real_distribution<int/*float*/> distrib; 
        return distrib(rng);/*thrust::make_tuple(distrib(rng), distrib(rng));*/
    }
};

template<typename T>
class HashNode 
{
public : 
    uint32_t key; 
    T value; 
    __host__ __device__ HashNode() {}
    HashNode (int key, T value) 
    {
        this->key   = key;
        this->value = value;
    }
    
    __host__ __device__ HashNode (const HashNode<T> &B)
    {
        this->key = B.key;
        this->value = B.value;
    }
};

template<typename T>
class HashMap 
{
    HashNode<T> *hashArr;
    int nodeCount;
    int bucketCount;
    int *bucketOffset;
    int *d_nodeCount;
    int *randSeed;
    gpmlog *nodeLog;
    gpmlog *mdLog;
    void init();
    void hashPhase1(int numberOfBuckets, int *keys, T *values, int size);
    void hashPhase2(int numberOfBuckets, int *modifiedBuckets);

    public: 
    HashMap() 
    {
        hashArr     = NULL;
        nodeCount   = 0;
        bucketCount = MAX_BUCKETS;
        
        size_t len   = MAX_KEYS * sizeof(HashNode<T>) + sizeof(int);
        hashArr      = (HashNode<T> *)gpm_map_file("parallel_hashmap.out", len, 1);
        d_nodeCount  = (int*)((char*)hashArr + MAX_KEYS * sizeof(HashNode<T>));
        
        gpuErrchk(cudaMalloc((void**)&bucketOffset, MAX_BUCKETS * sizeof(int)));        
        gpuErrchk(cudaMemset(hashArr, 0, MAX_KEYS * sizeof(HashNode<T>)));
        
        len = MAX_KEYS * sizeof(HashNode<T>);
#if defined(CONV_LOG)
        nodeLog = gpmlog_create("parallel_hash_nlog.out", len, MAX_KEYS);
#else
        nodeLog = gpmlog_create_managed("parallel_hash_nlog.out", len, MAX_BUCKETS, BUCKET_SIZE);
#endif
        len = sizeof(int);
        mdLog = gpmlog_create("parallel_hash_mdlog.out", len, 1);
    }
    
    HashMap(const char *file) 
    {
        hashArr     = NULL;
        nodeCount   = 0;
        bucketCount = MAX_BUCKETS;
        
        size_t len   = 0;
        hashArr      = (HashNode<T> *)gpm_map_file(file, len, 0);
        d_nodeCount  = (int*)((char*)hashArr + len - sizeof(int));
        cudaMemcpy(&nodeCount, d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost);
        
        nodeLog = gpmlog_open("parallel_hash_nlog.out");
        mdLog = gpmlog_open("parallel_hash_mdlog.out");
        
        //recover();
    }
    __host__ void insert (int *keys, T *values, int size);                       
    __host__ void delete_node (int *key);                                                         
    __host__ int size ();                                                                       
    __host__ T* find (int *keys, int size);
    __host__ void recover();
};

__device__ int calcBucketHash(int key, int numberOfBuckets)
{
    return key % numberOfBuckets;
}

template<class T>
__global__
void cuHashPhase1(HashNode<T> *hashArr, int *keys, T *values, int size, int bucketCount, int *bucketOffset) 
{
    int threadId = blockIdx.x * blockDim.x + threadIdx.x; 
    int bucketId; 
    while (threadId < size) {
        bucketId = calcBucketHash(keys[threadId], bucketCount);
        int position = atomicAdd(&bucketOffset[bucketId], 1);
        
        if(position > BUCKET_SIZE)
        	continue;
        gpm_memcpy_nodrain(&hashArr[bucketId * BUCKET_SIZE + position].key, &keys[threadId], sizeof(int), cudaMemcpyDeviceToDevice);
        gpm_memcpy_nodrain(&hashArr[bucketId * BUCKET_SIZE + position].value, &values[threadId], sizeof(DATATYPE), cudaMemcpyDeviceToDevice);        
        threadId += blockDim.x * gridDim.x;
    }
}

template<class T>
    __global__
void cuHashPhase2(HashNode<T> *hashArr, int *bucketOffset, int randSeed, int *successful, int *modifiedBucket) 
{
    __shared__ HashNode<T> table[3][192];
        
    int a = randSeed;
    uint32_t c[3][2]; 
    c[0][0] = (a+0x7ed55d16) + (a<<12);
    c[1][0] = (a^0xc761c23c) ^ (a>>19);
    c[2][0] = (a+0x165667b1) + (a<<5);
    c[0][1] = (a+0xd3a2646c) ^ (a<<9);
    c[1][1] = (a+0xfd7046c5) + (a<<3);
    c[2][1] = (a^0xb55a4f09) ^ (a>>16);

    //insert a condition to exit
    //when we start assessing a thread which is beyond the offset of that particular block
    //Exit
    if (threadIdx.x >= bucketOffset[blockIdx.x] || !modifiedBucket[blockIdx.x]) {
        return;
    }
    table[threadIdx.x / 192][threadIdx.x % 192].key = 0;
    __syncthreads();
    
    PMEM_READ_OP( HashNode<T> tempKey = hashArr[blockIdx.x * BUCKET_SIZE + threadIdx.x] , 8 )
    PMEM_WRITE_OP( hashArr[blockIdx.x * BUCKET_SIZE + threadIdx.x].key = 0 , 4 )
    int location    = -1;
    int tableNumber = 0;

    for (int i=0; i<25; i++) {
        for (int j=0; j<3; j++) {
        	
            int tableOffset = ((c[j][0] + c[j][1] * tempKey.key) % 1900813) % 192;
            if (location == -1 || table[tableNumber][location].key != tempKey.key) {
                location = -1;
                table[j][tableOffset] = tempKey;
            }
            __syncthreads();
            if (table[j][tableOffset].key == tempKey.key) {
                location = tableOffset;
                tableNumber = j;
            }
        }
        __syncthreads();
    } 
    if (location == -1 || table[tableNumber][location].key != tempKey.key) {
        *successful = 0;
    }
    else {
        //PMEM_WRITE_OP( hashArr[tableNumber * 192 + location] = tempKey, sizeof(HashNode<T>) )
        gpm_memcpy_nodrain(&hashArr[blockIdx.x * BUCKET_SIZE + tableNumber * 192 + location], 
        	&tempKey, sizeof(HashNode<T>), cudaMemcpyDeviceToDevice);
        srp_persist();
#ifdef GPM_EPOCH_MODEL
	__threadfence(); 
#endif 
    }
}

template<class T>
    __host__
void HashMap<T>::hashPhase1(int numberOfBuckets, int *keys, T *values, int size)
{
    cuHashPhase1<<<numberOfBuckets, BUCKET_SIZE>>>(this->hashArr, keys, values, size, this->bucketCount, this->bucketOffset);
}   

template<class T>
    __host__
void HashMap<T>::hashPhase2(int numberOfBuckets, int* modifiedBucket)
{
    int *d_successful, *h_successful = (int*)malloc(sizeof(int));
    gpuErrchk(cudaMalloc((void **)&d_successful, sizeof(int)));
    int seed = 0; // Make this randomly generated every iteration
    //do {
        gpuErrchk(cudaMemset(d_successful, 1, sizeof(int)));
        cuHashPhase2<<<numberOfBuckets, BUCKET_SIZE>>>(this->hashArr, this->bucketOffset, seed, d_successful, modifiedBucket);
        *h_successful = 1;
        gpuErrchk(cudaDeviceSynchronize());
        
        gpuErrchk(cudaMemcpy(h_successful, d_successful, sizeof(int), cudaMemcpyDeviceToHost));
        printf("Successful? %s\n", *h_successful ? "True" : "False");
    //} while(!(*h_successful));
    //gpuErrchk(cudaMemcpy(randSeed, &seed, sizeof(int), cudaMemcpyHostToDevice));
}   


__global__
void markBuckets(int *keys, int numberOfKeys, int *buckets, int numberOfBuckets)
{
    for(int i = threadIdx.x + blockDim.x * blockIdx.x; 
        i < numberOfKeys; i += blockDim.x * gridDim.x) {
        int bucket = calcBucketHash(keys[i], numberOfBuckets);
        buckets[bucket] = 1;
    }
}

template<class T>
    __global__
void compactBuckets(HashNode<T> *hashArr, int *bucketOffset, int numberOfBuckets, bool rehash, int *modifiedBucket, gpmlog *nodeLog)
{
    int position = blockIdx.x * BUCKET_SIZE + threadIdx.x;
    if(!rehash && !modifiedBucket[blockIdx.x])
        return;
    
    PMEM_READ_OP( HashNode<T> node = hashArr[position] , 8 )
    
#if defined(CONV_LOG)
    gpmlog_insert(nodeLog, &hashArr[position], sizeof(HashNode<T>), position);
#else
    gpmlog_insert(nodeLog, &hashArr[position], sizeof(HashNode<T>));
#endif
    srp_persist();
#ifdef GPM_EPOCH_MODEL
	__threadfence(); 
#endif 
    gpm_memset_nodrain(&hashArr[position], 0, sizeof(HashNode<T>));
    __syncthreads();
	
    if(node.key == 0)
        return;
    
    if(rehash) {
        int bucketId = calcBucketHash(node.key, numberOfBuckets);
        
        int pos = atomicAdd(&bucketOffset[bucketId], 1);
        //PMEM_WRITE_OP( hashArr[bucketId * BUCKET_SIZE + pos] = node, sizeof(HashNode<T>) )
        gpm_memcpy_nodrain(&hashArr[bucketId * BUCKET_SIZE + pos], &node, sizeof(HashNode<T>), cudaMemcpyDeviceToDevice);
    }
    else {
        int pos = atomicAdd(&bucketOffset[blockIdx.x], 1);
        //PMEM_WRITE_OP( hashArr[blockIdx.x * BUCKET_SIZE + pos] = node, sizeof(HashNode<T>) )
        gpm_memcpy_nodrain(&hashArr[blockIdx.x  * BUCKET_SIZE + pos], &node, sizeof(HashNode<T>), cudaMemcpyDeviceToDevice);
    }
}

__global__
void logHashMap(gpmlog *mdLog, int nodeCount)
{
    int TID = threadIdx.x + blockIdx.x * blockDim.x;
    
    if(TID == 0)
        gpmlog_insert(mdLog, &nodeCount, sizeof(int), 0);
    
    srp_persist();        
#ifdef GPM_EPOCH_MODEL
	__threadfence(); 
#endif 
}

__global__
void clearLog1(gpmlog *mdLog)
{
    if(threadIdx.x + blockIdx.x * blockDim.x == 0) {
        gpmlog_clear(mdLog, 0);
    }
    srp_persist();
}

__global__
void clearLog2(gpmlog *nodeLog)
{
#if defined(CONV_LOG)
    gpmlog_clear(nodeLog, threadIdx.x + blockIdx.x * blockDim.x);
#else
    gpmlog_clear(nodeLog);
#endif
}


template<class T>
__global__ void recoverFromLog(gpmlog *nodeLog, int *bucketOffsets, HashNode<T> *hashArr, int numberOfBuckets)
{
    for(int bucket = blockIdx.x; bucket < numberOfBuckets; bucket += gridDim.x) {
        int position = bucket * BUCKET_SIZE + threadIdx.x;
        if(gpmlog_get_size(nodeLog, position) == sizeof(HashNode<T>)) {
            gpmlog_read(nodeLog, &hashArr[position], sizeof(HashNode<T>), position);
            if(hashArr[position].key != 0)
            	atomicAdd(&bucketOffsets[bucket], 1);
        }
        gpmlog_clear(nodeLog, position);
    }
    srp_persist();
}

__global__ void getNodeCount(gpmlog *mdLog, int *count, bool *abort)
{
    *abort = false;
    if(gpmlog_get_size(mdLog, 0) == 0) {
        *abort = true;
        return;
    }
    
    gpmlog_read(mdLog, count, sizeof(int));
}


__global__
void clearLog(int numberOfBuckets, gpmlog *nodeLog)
{
    for(int bucket = blockIdx.x; bucket < numberOfBuckets; bucket += gridDim.x) {
        int position = bucket * BUCKET_SIZE + threadIdx.x;
        gpmlog_clear(nodeLog, position);
    }
}

template<class T>
    __host__
void HashMap<T>::recover()
{
    bool *abort;
    cudaMalloc((void**)&abort, sizeof(bool));
    int *count;
    cudaMalloc((void**)&count, sizeof(int));
    
    getNodeCount<<<1, 1>>>(mdLog, count, abort);

    bool host_abort;
    cudaMemcpy(&host_abort, abort, sizeof(bool), cudaMemcpyDeviceToHost);
    if(host_abort) {
        cudaMemcpy(&nodeCount, d_nodeCount, sizeof(int), cudaMemcpyDeviceToHost);
        clearLog<<<50, BUCKET_SIZE>>>(nodeCount / 409 + 1, nodeLog);
        // Todo: Recalculate bucketOffsets, but we shouldn't be running this condition anyway.
        assert(false);
    }
    
    cudaMemcpy(&nodeCount, count, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemset(hashArr, 0, MAX_KEYS * sizeof(HashNode<T>));
    recoverFromLog<T><<<50, BUCKET_SIZE>>>(nodeLog, bucketOffset, hashArr, nodeCount / 409 + 1);
}

template<class T>
    __host__
void HashMap<T>::insert(int *keys, T *values, int size)
{
    int numberOfBuckets = (nodeCount + size)/309 + 1;
    bool rehash = false;
    if (bucketCount < numberOfBuckets) {
        bucketCount = numberOfBuckets;
        rehash = true;
    }
    
    int *modifiedBucket;
    cudaMalloc((void **)&modifiedBucket, sizeof(int) * bucketCount);
    cudaMemset(modifiedBucket, 0, sizeof(int) * bucketCount);

	// Log details of no. of nodes and bucket offsets
    logHashMap<<<1, 1>>>(mdLog, nodeCount);
    
    nodeCount += size;
    cudaMemcpy(d_nodeCount, &nodeCount, sizeof(int), cudaMemcpyHostToDevice);
    
    cudaMemset(bucketOffset, 0, sizeof(int) * bucketCount);
    markBuckets<<<32, 1024>>>(keys, size, modifiedBucket, bucketCount);
    compactBuckets <<<bucketCount, BUCKET_SIZE>>> (hashArr, bucketOffset, bucketCount, rehash, modifiedBucket, nodeLog);
    hashPhase1(bucketCount, keys, values, size);
    hashPhase2(bucketCount, modifiedBucket);
#ifndef RECOVERY
    clearLog1<<<1, 1>>>(mdLog);
    clearLog2<<<bucketCount, BUCKET_SIZE>>>(nodeLog);
#else
    cout << "Recovery begins\n"; 
	recover();
    cudaDeviceSynchronize();
    cout << "Recovery ends\n"; 
#endif
    cudaDeviceSynchronize();
    cudaFree(modifiedBucket);
}


template<class T>
__global__ 
void findKernel(HashNode<T> *hashArr, int *keys, T *values, int size, int *bucketOffset, int numberOfBuckets)
{
    int a = 0;
    uint32_t c[3][2]; 
    c[0][0] = (a+0x7ed55d16) + (a<<12);
    c[1][0] = (a^0xc761c23c) ^ (a>>19);
    c[2][0] = (a+0x165667b1) + (a<<5);
    c[0][1] = (a+0xd3a2646c) ^ (a<<9);
    c[1][1] = (a+0xfd7046c5) + (a<<3);
    c[2][1] = (a^0xb55a4f09) ^ (a>>16);
    
    for(int i = threadIdx.x + blockIdx.x * blockDim.x; i < size; i += blockDim.x * gridDim.x) {
        int bucketId = calcBucketHash(keys[i], numberOfBuckets);

        for (int j=0; j<3; j++) {
            int tableOffset = ((c[j][0] + c[j][1] * (uint32_t)keys[i]) % 1900813) % 192;
            if (hashArr[bucketId * BUCKET_SIZE + j * 192 + tableOffset].key == keys[i]) {
                values[i] = hashArr[bucketId * BUCKET_SIZE + j * 192 + tableOffset].value;
                break;
            }
        }
    }
}

template<class T>
    __host__
T* HashMap<T>::find(int *keys, int size)
{
    T* deviceVals;
    gpuErrchk(cudaMalloc((void **)&deviceVals, sizeof(T) * size));
    findKernel<T><<<30, BUCKET_SIZE>>> (hashArr, keys, deviceVals, size, bucketOffset, bucketCount);
    cudaDeviceSynchronize();
    T* values = new T[size];
    gpuErrchk(cudaMemcpy(values, deviceVals, sizeof(T) * size, cudaMemcpyDeviceToHost));
    gpuErrchk(cudaFree(deviceVals));
    return values;
}

//genrate a set of random keys to populate the hashmap
void genRandomKeys(int **deviceKeys, DATATYPE **deviceValues, int numberOfKeys, int start) 
{    
    int *hashKey = new int[numberOfKeys];
    DATATYPE *hashVals = new DATATYPE[numberOfKeys];
    for(int i = start; i < start + numberOfKeys; ++i)
    {
        hashKey[i - start] = i + 1;
        hashVals[i - start] = rand() % 100000;
    }
    gpuErrchk(cudaMalloc((void **)deviceKeys, sizeof(int) * numberOfKeys));
    gpuErrchk(cudaMemcpy(*deviceKeys, hashKey, sizeof(int) * numberOfKeys, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMalloc((void **)deviceValues, sizeof(DATATYPE) * numberOfKeys));
    gpuErrchk(cudaMemcpy(*deviceValues, hashVals, sizeof(DATATYPE) * numberOfKeys, cudaMemcpyHostToDevice));
    delete hashKey;
    delete hashVals;
}

void insertRandomKeys(int numberOfKeys, HashMap<DATATYPE> *hashMap, int start) 
{
    int *keys;
    DATATYPE *values;
    genRandomKeys(&keys, &values, numberOfKeys, start);
    hashMap->insert(keys, values, numberOfKeys);
#ifndef RECOVERY
    DATATYPE *values1 = new DATATYPE[numberOfKeys];
    cudaMemcpy(values1, values, sizeof(DATATYPE) * numberOfKeys, cudaMemcpyDeviceToHost);
    DATATYPE *values2 = new DATATYPE[numberOfKeys];
    values2 = hashMap->find(keys, numberOfKeys);
    int *keys2 = new int[numberOfKeys];
    cudaMemcpy(keys2, keys, sizeof(DATATYPE) * numberOfKeys, cudaMemcpyDeviceToHost);
    for(int i = 0; i < numberOfKeys; ++i)
        if(values1[i] != values2[i]) {
            printf("Val not matching at %d, expected %d, found %d\n", i, values1[i], values2[i]);
            fflush(stdout);
            assert(false);
        }
    delete values1;
    delete values2;
#endif
    gpuErrchk(cudaFree(keys));
    gpuErrchk(cudaFree(values));
}

int main(int argc, char **argv) 
{  
    /*if(argc < 2) {
        printf("Needs number of keys as argument\n");
        return 0;
    }*/
    int numberOfKeys = 25000;//atoi(argv[1]);
    HashMap<DATATYPE> *hashMap = new HashMap<DATATYPE>; 
    //insertRandomKeys allows user to insert any number of random keys into the hashmap.
    printf("%d keys with max size %ld\n", numberOfKeys, MAX_KEYS);
    insertRandomKeys(numberOfKeys, hashMap, numberOfKeys);
    //The insert function also allows the user to insert keys into the hashmap, but the user has to provide both the key and the value vector for it.
    return 0;
}
