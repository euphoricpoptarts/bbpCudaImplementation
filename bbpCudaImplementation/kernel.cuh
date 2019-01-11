#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#define uint32 unsigned int
#define uint64 unsigned long long
#define fastModLimit 0xfffffff
#define fastModULTRAINSTINCT 0x3ffffffffff
//#define QUINTILLION

__device__  __constant__ const uint64 twoTo63Power = 0x8000000000000000;
__device__ int printOnce = 0;

struct sJ {
	uint64 s[2] = { 0, 0 };
};

__device__ __host__ void sJAdd(sJ* addend, const sJ* augend);
void bbpPassThrough(int threads, int blocks, sJ *c, uint64 *progress, uint64 startingExponent, uint64 begin, uint64 end, uint64 strideMultiplier);
void reduceSJPassThrough(int threads, int blocks, sJ *c, int offset, int stop);