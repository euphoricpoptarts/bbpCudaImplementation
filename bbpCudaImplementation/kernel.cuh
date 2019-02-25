#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#define uint32 unsigned int
#define uint64 unsigned long long
#define fastModLimit 0xfffffff
#define fastModULTRAINSTINCT 0x7ffffffffff
//#define QUINTILLION

struct uint128 {
	uint64 msw = 0, lsw = 0;
};

__device__ __host__ void sJAdd(uint128* addend, const uint128* augend);
void bbpPassThrough(int threads, int blocks, uint128 *c, uint64 *progress, uint64 startingExponent, uint64 begin, uint64 end, uint64 strideMultiplier);
void reduceUint128ArrayPassThrough(int threads, int blocks, uint128 *c, int offset, int stop);