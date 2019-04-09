#include "kernel.cuh"

__device__  __constant__ const uint64 twoTo63Power = 0x8000000000000000;
__device__ int printOnce = 0;

__global__ void bbpKernel(uint128 *c, uint64 *progress, uint64 startingExponent, uint64 begin, uint64 end, uint64 stride);

//adds all elements of addend and augend, storing in addend
__device__ __host__ void sJAdd(uint128* addend, const uint128* augend) {
	addend->lsw += augend->lsw;
	addend->msw += augend->msw;
	if (addend->lsw < augend->lsw) addend->msw++;
}

//uses 32 bit multiplications to compute the highest 64 and lowest 64 bits of squaring a 64 bit number
//in assembly in order to access carry bit
//saves work with realization that (hi + lo)^2 = hi^2 + 2*hi*lo + lo^2
//compare to non-squaring multiplication (hi1 + lo1)*(hi2 + lo2) = hi1*hi2 + hi1*lo2 + lo1*hi2 + lo1*lo2
//one fewer multiplication is needed
__device__ void square64By64(uint64 multiplicand, uint64 & lo, uint64 & hi) {

	asm("{\n\t"
		".reg .u64          m0, m1, m2;\n\t"
		".reg .u32          t0, t1, t2, t3, v0, v1;\n\t"
		"mov.b64           {v0, v1}, %2;\n\t" //splits a into hi and lo 32 bit words
		"mul.wide.u32       m0, v0, v0;    \n\t" //m0 = alo*alo
		"mul.wide.u32       m1, v0, v1;    \n\t" //m1 = alo*ahi
		"mul.wide.u32       m2, v1, v1;    \n\t" //m2 = ahi*ahi
		"mov.b64           {t0, t1}, m0;\n\t"
		"mov.b64           {t2, t3}, m2;\n\t"
		"add.cc.u64         m1, m1, m1;\n\t" //because (ahi + alo)^2 = ahi^2 + 2*alo*ahi + alo^2, we must double m1
		"addc.u32           t3,  t3,  0;\n\t"
		"mov.b64           {v0, v1}, m1;\n\t"
		"add.cc.u32         t1, t1, v0;\n\t"
		"addc.cc.u32        t2, t2, v1;\n\t"
		"addc.u32           t3, t3, 0;\n\t"
		"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"mov.b64            %1, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(lo), "=l"(hi)
		: "l"(multiplicand));
}

__device__ void subtractModIfMoreThanMod(uint64 & value, const uint64 & mod) {
	asm("{\n\t"
		".reg .u64        t0;\n\t"
		"sub.u64          t0, %1, %2;\n\t"
		"min.u64          %0, t0, %1;\n\t"
		"}"
		: "=l"(value)
		: "l"(value), "l"(mod));
}

//using R=2^32, performs a 2 step montgomery reduction on the 128-bit number represented by hi and lo
//assembly is used to access carry bit
__device__ void montgomeryAddAndShift32Bit(uint64 & hi, uint64 & lo, const uint64 & mod, const uint32 & mprime) {
	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t0, t1, t2, t3, z0, m0, m1;\n\t"
		"mov.b64           {m0, m1}, %3;\n\t" //splits mod into m0 and m1
		"mov.b64           {t0, t1}, %1;\n\t" //splits lo into hi and lo 32 bit words
		"mov.b64           {t2, t3}, %2;\n\t" //splits hi into hi and lo 32 bit words

		//montgomery reduction on least significant 32-bit word
		"mul.lo.u32         z0, %4, t0;\n\t"
		"mad.lo.cc.u32      t0, z0, m0, t0;\n\t"
		"madc.hi.cc.u32     t1, z0, m0, t1;\n\t"
		"addc.cc.u32        t2,  0, t2;\n\t"
		"addc.u32           t3,  0, t3;\n\t"
		"mad.lo.cc.u32      t1, z0, m1, t1;\n\t"
		"madc.hi.cc.u32     t2, z0, m1, t2;\n\t"
		"addc.u32           t3,  0, t3;\n\t"

		//montgomery reduction on second least significant 32-bit word
		"mul.lo.u32         z0, %4, t1;\n\t"
		"mad.lo.cc.u32      t1, z0, m0, t1;\n\t"
		"madc.hi.cc.u32     t2, z0, m0, t2;\n\t"
		"addc.u32           t3,  0, t3;\n\t"
		"mad.lo.cc.u32      t2, z0, m1, t2;\n\t"
		"madc.hi.u32        t3, z0, m1, t3;\n\t"
		"mov.b64            %0, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(hi)
		: "l"(lo), "l"(hi), "l"(mod), "r"(mprime));
}

__device__ void add128Bit(uint64 & addendHi, uint64 & addendLo, uint64 augendHi, uint64 augendLo) {
	asm("{\n\t"
		"add.cc.u64         %1, %3, %5;\n\t"
		"addc.u64           %0, %2, %4;\n\t"
		"}"
		: "=l"(addendHi), "=l"(addendLo)
		: "l"(addendHi), "l"(addendLo), "l"(augendHi), "l"(augendLo));
}

__device__ void sub128Bit(uint64 & addendHi, uint64 & addendLo, uint64 augendHi, uint64 augendLo) {
	asm("{\n\t"
		"sub.cc.u64         %1, %3, %5;\n\t"
		"subc.u64           %0, %2, %4;\n\t"
		"}"
		: "=l"(addendHi), "=l"(addendLo)
		: "l"(addendHi), "l"(addendLo), "l"(augendHi), "l"(augendLo));
}

//finds output such that (n * output) % 2^64 = -1
//found this approach used here: http://plouffe.fr/simon/1-s2.0-S0167819118300334-main.pdf
//an explanation of the approach: http://marc-b-reynolds.github.io/math/2017/09/18/ModInverse.html
//saves from 15-25% of the total computation time over xbinGCD method (on the lower side of that for larger digit computations)
__device__ void modInverseNewtonsMethod(uint64 n, uint64 & output) {
	//n * 3 xor 2
	output = ((n << 1) + n) ^ 2LLU;

#pragma unroll
	for (int i = 0; i < 4; i++) {
		output = output * (2 - (n * output));
	}

	//we have (n * output) % 2^64 = 1, so we need to invert it
	output = -output;
}

//montgomery multiplication routine identical to above except for only being used when abar and bbar are known in advance to be the same
//uses a faster multiplication routine for squaring than is possible while not squaring
__device__ void montgomerySquare(uint64 abar, uint64 mod, uint32 mprime, uint64 & output) {

	uint64 tlo = 0;

	square64By64(abar, tlo, output);

	montgomeryAddAndShift32Bit(output, tlo, mod, mprime);

	//can be removed if mod < 2^62
	//see this paper: https://pdfs.semanticscholar.org/0e6a/3e8f30b63b556679f5dff2cbfdfe9523f4fa.pdf
#ifdef QUINTILLION
	subtractModIfMoreThanMod(output, mod);
#endif
}

__device__ void fixedPointDivisionExact(const uint64 & mod, const uint64 & r, const uint64 & mPrime, uint128 * result, int negative) {
	if (!r) return;

	uint64 q0 = (-r)*mPrime;
	uint64 q1 = -(1LLU) - __umul64hi(mod, q0);
	q1 *= mPrime;

	if(!negative) add128Bit(result->msw, result->lsw, q1, q0);
	else sub128Bit(result->msw, result->lsw, q1, q0);
}

__device__ void fixedPointDivisionExactWithShift(const uint64 & mod, const uint64 & r, const uint64 & mPrime, uint128 * result, int shift, int negative) {
	if (!r) return;

	uint64 q0 = (-r)*mPrime;
	uint64 q1 = -(1LLU) - __umul64hi(mod, q0);
	q1 *= mPrime;

	q0 >>= shift;
	if(shift <= 64) q0 = q0 + (q1 << (64 - shift));
	else q0 = q0 + (q1 >> (shift - 64));
	q1 >>= shift;

	if (!negative) add128Bit(result->msw, result->lsw, q1, q0);
	else sub128Bit(result->msw, result->lsw, q1, q0);
}

/*
  calculates left-to-right binary exponentiation with respect to a modulus
  adds the 128 bit number representing ((2^exp)%mod)/mod to result
  exponentation may be partially precomputed
  @param exp: the exponent (shocking)
  @param mod: modulus under which the exponentation should occur
  @param result: pointer to the structure containing the sum of previous modular exponentations to add this result to
  @param negative: 1 if this is a negative term in the summation, 0 if it is positive
  @param montgomeryStart: 2^(64 + precomputed bits of exponent) % mod
	This is 2 in montgomery space with no precomputation, or the precomputed part of the exponent in montgomery space with precomputation.
  @param shiftToLittleBit: 63 if no precomputation has occurred (and 1 less than 63 for every precomputed bit of the exponent)
  uses montgomery multiplication to avoid modulus operations
*/
__device__ __noinline__ void modExpLeftToRight(uint64 exp, const uint64 & mod, uint128 * result, const int & negative, uint64 montgomeryStart, int shiftToLittleBit) {
	uint64 output = 1;
	uint64 mPrime;

	modInverseNewtonsMethod(mod, mPrime);

	uint32 mPrime32 = mPrime;

	//exp = exp - subtract;

	int shift = 0;

	if (exp < 128) {
		shift = 128 - exp;
		exp = 128;
	}

	//this makes it unnecessary to convert out of montgomery space
	exp -= 64;

	output = montgomeryStart;

	shiftToLittleBit -= __clzll(exp);

	while (shiftToLittleBit-- != 0) {

		montgomerySquare(output, mod, mPrime32, output);
		
		output <<= (exp >> shiftToLittleBit) & 1;

	}

	//remove these if you don't mind a slight decrease in precision
#ifndef QUINTILLION
	subtractModIfMoreThanMod(output, mod << 1);
#endif
	subtractModIfMoreThanMod(output, mod);

	if (shift) {
		fixedPointDivisionExactWithShift(mod, output, -mPrime, result, shift, negative);
	}
	else {
		fixedPointDivisionExact(mod, output, -mPrime, result, negative);
	}
}

//finds montgomeryStart so that 2^(63 + loops) % startMod = montgomeryStart
//finds div so that montgomeryStart + n*div is congruent to 2^(63 + loops) % (startMod - n*modCoefficient)
//this is possible because montgomery multiplication does not require we know 2^(63 + loops) % mod exactly, but requires we know a number congruent to 2^(63 + loops) % mod (as long as this number is less than 2^63)
//div is inversely proportional to startMod ( div = 2^(63 + loops) / startMod )
//montgomeryStart + n*div is < 2*mod for mod > 2^( (63 + loops + log(n*modCoefficient) ) / 2)
__device__ __noinline__ void fastModApproximator(uint64 endMod, uint64 startExp, uint64 endExp, uint64 modCoefficient, uint64 & montgomeryStart, uint64 & div, int & shiftToLittleBit, int fasterModViable) {
		div = twoTo63Power / endMod;

		//selects the most significant four (or five if mod is large enough) bits of startExp and endExp
		//if these chosen bits match, then these bits are added to loops
		//this part of the exponent will then be precomputed, so that the modular exponentiation routine may skip these bits
		int sixty4MinusBitsToCompare = 60;
		if (fasterModViable) sixty4MinusBitsToCompare = 59;
		int largest4BitsShift = sixty4MinusBitsToCompare - __clzll(startExp);
		int loops = 2;
		if ((startExp >> largest4BitsShift) == (endExp >> largest4BitsShift)) {
			shiftToLittleBit = sixty4MinusBitsToCompare;
			loops = 1 + (startExp >> largest4BitsShift);
		}


		for (int i = 0; i < loops; i++) {
			div <<= 1;
			if (-(div * endMod) > endMod) div++;
		}
		montgomeryStart = 0 - (div * endMod);// 2^(63 + loops) - div*startMod = 2^(63 + loops) % startMod
		div *= modCoefficient;
}

//computes strideMultiplier # of summation terms
__device__ void bbp(uint64 startingExponent, uint64 start, uint64 end, uint64 & startingMod, uint64 modCoefficient, int negative, uint128* output, uint64* progress) {

	//depending on the size of the smallest mod a thread will operate on
	//these variables determine which optimizations are viable
	//need to check that startExp used in fastModApproximator is greater than 64 (compare to 128 because we haven't subtracted 64 yet to prevent underflow)
	//as startExp must have at least 5 bits for fastModApproximator to work (128 looks sleeker than 96)
	int fastModViable = (modCoefficient * start + startingMod) > fastModLimit && (startingExponent - start*10LLU) > 128;
	int fasterModViable = (modCoefficient * start + startingMod) > fastModULTRAINSTINCT;
	uint64 montgomeryStart, div;
	//shiftToLittleBit is used to find how many total squarings are needed in exponentiation
	//63 computes all necessary squarings
	//every 1 less will skip a squaring
	int shiftToLittleBit = 63;

	if(fastModViable) fastModApproximator(modCoefficient * end + startingMod, startingExponent - 64 - start*10LLU, startingExponent - 64 - end*10LLU, modCoefficient, montgomeryStart, div, shiftToLittleBit, fasterModViable);
	
	//go backwards so we can add div instead of subtracting it
	//subtracting produces a likelihood of underflow (whereas addition will not cause overflow for any mod where 2^8 < mod < (2^64 - 2^8) )
	//also condition is (k + 1) > start as opposed to k >= start because if start is 0 then k >= start has no end condition
	for (uint64 k = end; (k + 1) > start; k--) {
		uint64 exp = startingExponent - (k*10LLU);
		uint64 mod = modCoefficient * k + startingMod;
		if(!fastModViable) {
			montgomeryStart = twoTo63Power % mod;
			montgomeryStart <<= 1;
			subtractModIfMoreThanMod(montgomeryStart, mod);
			montgomeryStart <<= 1;
			subtractModIfMoreThanMod(montgomeryStart, mod);
		}

		modExpLeftToRight(exp, mod, output, negative, montgomeryStart, shiftToLittleBit);

		negative ^= 1;
		montgomeryStart += div;
	}

	//send some progress information to the host device
	//once per 2^20/strideMultiplier threads
	if (((end + 1) & 0xfffff) <= (start & 0xfffff) && end > start) {
		atomicMax(progress, end);
	}
}

//determine from thread and block position which parts of summation to calculate
//only one of the threads per kernel (AND ONLY ON GPU0) will report progress
//stride over all parts of summation in bbp formula where k <= startingExponent (between all threads of all launches)
__global__ void bbpKernel(uint128 *c, uint64 *progress, uint64 startingExponent, uint64 begin, uint64 end, uint64 strideMultiplier)
{
	int gridId = threadIdx.x + blockDim.x * blockIdx.x;
	int divider = (blockDim.x * gridDim.x) / 7;
	uint64 start = begin + (gridId % divider)*strideMultiplier;
	uint64 mod = 0, modCoefficient = 4;
	end = ullmin(end, start + strideMultiplier - 1);
	int negative = end & 1;
	switch (gridId / divider) {
	case 0:
		mod = 1;//4k + 1
		startingExponent -= 3;
		negative ^= 1;
		break;
	case 1:
		mod = 3;//4k + 3
		startingExponent -= 8;
		negative ^= 1;
		break;
	case 2:
		mod = 1;//10k + 1
		modCoefficient = 10;
		break;
	case 3:
		mod = 3;//10k + 3
		modCoefficient = 10;
		startingExponent -= 2;
		negative ^= 1;
		break;
	case 4:
		mod = 5;//10k + 5
		modCoefficient = 10;
		startingExponent -= 6;
		negative ^= 1;
		break;
	case 5:
		mod = 7;//10k + 7
		modCoefficient = 10;
		startingExponent -= 6;
		negative ^= 1;
		break;
	case 6:
		mod = 9;//10k + 9
		modCoefficient = 10;
		startingExponent -= 8;
	}
	__shared__ uint64 modArr[128];
	modArr[threadIdx.x] = mod;
	bbp(startingExponent, start, end, modArr[threadIdx.x], modCoefficient, negative, c + gridId, progress);
}

//stride over current leaves of reduce tree
__global__ void reduceUint128ArrayKernel(uint128 *c, int offset, int stop) {
	int stride = blockDim.x * gridDim.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	while (i < stop) {
		int augend = i + offset;
		sJAdd(c + i, c + augend);
		i += stride;
	}
}

void bbpPassThrough(int threads, int blocks, uint128 *c, uint64 *progress, uint64 startingExponent, uint64 begin, uint64 end, uint64 strideMultiplier) {
	bbpKernel << <blocks, threads >> > (c, progress, startingExponent, begin, end, strideMultiplier);
}

void reduceUint128ArrayPassThrough(int threads, int blocks, uint128 *c, int offset, int stop) {
	reduceUint128ArrayKernel<<<blocks, threads>>>(c, offset, stop);
}