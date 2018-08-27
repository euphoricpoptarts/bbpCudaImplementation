
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <deque>
#include <atomic>
#include <iostream>
#include <filesystem>
#include <string>
#include <algorithm>

#define uint64 unsigned long long

namespace chr = std::chrono;

const int totalGpus = 2;

//warpsize is 32 so optimal value is probably always a multiple of 32
const int threadCountPerBlock = 128;
//this is more difficult to optimize but seems to not like odd numbers
const int blockCount = 2240;

__device__ __constant__ const uint64 baseSystem = 1024;
//__device__  __constant__ const int baseExpOf2 = 10;

__device__  __constant__ const uint64 int64MaxBit = 0x8000000000000000;

//__device__ int printOnce = 0;

struct sJ {
	double s[7] = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
};

typedef struct {
	volatile uint64 *currentProgress;
	volatile uint64 *deviceProg;
	sJ previousCache;
	double previousTime;
	sJ status[totalGpus];
	volatile uint64 nextStrideBegin[totalGpus];
	uint64 maxProgress;
	volatile int quit = 0;
	cudaError_t error;
	chr::high_resolution_clock::time_point * begin;
	volatile std::atomic<int> dataWritten;
} PROGRESSDATA, *PPROGRESSDATA;

typedef struct {
	sJ output;
	uint64 digit;
	uint64 beginFrom;
	int gpu = 0;
	int totalGpus = 0;
	int size = 0;
	cudaError_t error;
	volatile uint64 *deviceProg;
	sJ * status;
	volatile uint64 * nextStrideBegin;
	volatile std::atomic<int> * dataWritten;
} BBPLAUNCHERDATA, *PBBPLAUNCHERDATA;

PPROGRESSDATA setupProgress();
void progressCheck(PPROGRESSDATA data);
void cudaBbpLauncher(PBBPLAUNCHERDATA dataV);

//adds all elements of addend and augend, storing in addend
__device__ __host__ void sJAdd(sJ* addend, const sJ* augend) {
	for (int i = 0; i < 7; i++) {
		addend->s[i] += augend->s[i];
		if (addend->s[i] >= 1.0) addend->s[i] -= 1.0;
	}
}

//uses 32 bit multiplications to compute the highest 64 and lowest 64 bits of multiplying 2 64 bit numbers together
__device__ void multiply64By64(uint64 multiplicand, uint64 multiplier, uint64 * lo, uint64 * hi) {

	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t0, t1, t2, t3, v0, v1, v2, v3;\n\t"
		"mov.b64           {v0, v1}, %2;\n\t" //splits a into hi and lo 32 bit words
		"mov.b64           {v2, v3}, %3;\n\t" //splits b into hi and lo 32 bit words
		"mul.lo.u32         t0, v0, v2;    \n\t" //lolo = lo(alo*blo)
		"mul.hi.u32         t1, v0, v2;    \n\t" //lohi = hi(alo*blo)
		"mad.lo.cc.u32      t1, v0, v3, t1;\n\t" //lohi = lo(alo*bhi) + hi(alo*blo) (with carry flag)
		"madc.hi.cc.u32     t2, v0, v3,  0;\n\t" //hilo = hi(alo*bhi) + 1 carry (with carry flag, as carry may need to propagate)
		"madc.hi.u32        t3, v1, v3,  0;\n\t" //hihi = hi(ahi*bhi) + 1 carry (no need to set carry)
		"mad.lo.cc.u32      t1, v1, v2, t1;\n\t" //lohi = lo(ahi*blo) + lo(alo*bhi) + hi(alo*blo) (with carry flag)
		"madc.hi.cc.u32     t2, v1, v2, t2;\n\t" //hilo = hi(ahi*blo) + hi(alo*bhi) + 2 carries (with carry flag)
		"addc.u32           t3, t3, 0;\n\t" //hihi = hi(ahi*bhi) + 2 carries (no need to set carry)
		"mad.lo.cc.u32      t2, v1, v3, t2;\n\t" //hilo = lo(ahi*bhi) + hi(ahi*blo) + hi(alo*bhi) + 2 carries (with carry flag)
		"addc.u32           t3, t3, 0;\n\t" //hihi = hi(ahi*bhi) + 3 carries
		"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"mov.b64            %1, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(*lo), "=l"(*hi)
		: "l"(multiplicand), "l"(multiplier));
}

//uses 32 bit multiplications to compute the lowest 64 bits of multiplying 2 64 bit numbers together
//faster than multiplicand*multiplier
__device__ void multiply64By64LoOnly(uint64 multiplicand, uint64 multiplier, uint64 * lo) {

	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t0, t1, v0, v1, v2, v3;\n\t"
		"mov.b64           {v0, v1}, %1;\n\t" //splits a into hi and lo 32 bit words
		"mov.b64           {v2, v3}, %2;\n\t" //splits b into hi and lo 32 bit words
		"mul.lo.u32         t0, v0, v2;    \n\t" //lolo = lo(alo*blo)
		"mul.hi.u32         t1, v0, v2;    \n\t" //lohi = hi(alo*blo)
		"mad.lo.cc.u32      t1, v0, v3, t1;\n\t" //lohi = lo(alo*bhi) + hi(alo*blo) (with carry flag)
		"mad.lo.u32         t1, v1, v2, t1;\n\t" //lohi = lo(ahi*blo) + lo(alo*bhi) + hi(alo*blo) (with carry flag)
		"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"}"
		: "=l"(*lo)
		: "l"(multiplicand), "l"(multiplier));
}

//uses 32 bit multiplications to compute the highest 64 bits of multiplying 2 64 bit numbers together
//adding it to value currently in hi
__device__ void multiply64By64PlusHi(uint64 multiplicand, uint64 multiplier, uint64 * hi) {

	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t1, t2, t3, v0, v1, v2, v3;\n\t"
		"mov.b64           {t2, t3}, %3;\n\t"
		"mov.b64           {v0, v1}, %1;\n\t" //splits a into hi and lo 32 bit words
		"mov.b64           {v2, v3}, %2;\n\t" //splits b into hi and lo 32 bit words
		//"mul.lo.u32         t0, v0, v2;    \n\t" //lolo = lo(alo*blo)
		"mul.hi.u32         t1, v0, v2;    \n\t" //lohi = hi(alo*blo)
		"mad.lo.cc.u32      t1, v0, v3, t1;\n\t" //lohi = lo(alo*bhi) + hi(alo*blo) (with carry flag)
		"madc.hi.cc.u32     t2, v0, v3, t2;\n\t" //hilo = starting_value + hi(alo*bhi) + 1 carry (with carry flag)
		"madc.hi.u32        t3, v1, v3, t3;\n\t" //hihi = starting_value + hi(ahi*bhi) + 1 carry (no need to set carry)
		"mad.lo.cc.u32      t1, v1, v2, t1;\n\t" //lohi = lo(ahi*blo) + lo(alo*bhi) + hi(alo*blo) (with carry flag)
		"madc.hi.cc.u32     t2, v1, v2, t2;\n\t" //hilo = starting_value + hi(ahi*blo) + hi(alo*bhi) + 2 carries (with carry flag)
		"addc.u32           t3, t3, 0;\n\t" //hihi = starting_value + hi(ahi*bhi) + 2 carries (no need to set carry)
		"mad.lo.cc.u32      t2, v1, v3, t2;\n\t" //hilo = starting_value + lo(ahi*bhi) + hi(ahi*blo) + hi(alo*bhi) + 2 carries (with carry flag)
		"addc.u32           t3, t3, 0;\n\t" //hihi = starting_value + hi(ahi*bhi) + 3 carries
		//"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"mov.b64            %0, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(*hi)
		: "l"(multiplicand), "l"(multiplier), "l"(*hi));
}

//uses 32 bit multiplications to compute the highest 64 and lowest 64 bits of multiplying 2 64 bit numbers together
//adds the results to the contents of lo
//basically a 128 bit mad with 64 bit inputs
__device__ void multiply64By64PlusLo(uint64 multiplicand, uint64 multiplier, uint64 * lo, uint64 * hi) {
	
	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t0, t1, t2, t3, v0, v1, v2, v3;\n\t"
		"mov.b64           {t0, t1}, %4;\n\t" //splits lo into t0 and t1
		"mov.b64           {v0, v1}, %2;\n\t" //splits a into hi and lo 32 bit words
		"mov.b64           {v2, v3}, %3;\n\t" //splits b into hi and lo 32 bit words
		"mad.lo.cc.u32      t0, v0, v2, t0;\n\t" //lolo = starting_value + lo(alo*blo) (with carry flag)
		"madc.hi.cc.u32     t1, v0, v2, t1;\n\t" //lohi = starting_value + hi(alo*blo) + 1 carry (with carry flag)
		"madc.hi.cc.u32     t2, v0, v3,  0;\n\t" //hilo = hi(alo*bhi) + 1 carry (with carry flag, as carry may need to propagate)
		"madc.hi.u32        t3, v1, v3,  0;\n\t" //hihi = hi(ahi*bhi) + 1 carry (no need to set carry)
		"mad.lo.cc.u32      t1, v0, v3, t1;\n\t" //lohi = starting_value + lo(alo*bhi) + hi(alo*blo) + 1 carry (with carry flag)
		"madc.hi.cc.u32     t2, v1, v2, t2;\n\t" //hilo = hi(ahi*blo) + hi(alo*bhi) + 2 carries (with carry flag)
		"addc.u32           t3, t3, 0;\n\t" //hihi = hi(ahi*bhi) + 2 carries (no need to set carry)
		"mad.lo.cc.u32      t1, v1, v2, t1;\n\t" //lohi = starting_value + lo(ahi*blo) + lo(alo*bhi) + hi(alo*blo) + 1 carry (with carry flag)
		"madc.lo.cc.u32     t2, v1, v3, t2;\n\t" //hilo = lo(ahi*bhi) + hi(ahi*blo) + hi(alo*bhi) + 3 carries (with carry flag)
		"addc.u32           t3, t3, 0;\n\t" //hihi = hi(ahi*bhi) + 3 carries
		"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"mov.b64            %1, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(*lo), "=l"(*hi)
		: "l"(multiplicand), "l"(multiplier), "l"(*lo));
}

//uses 32 bit multiplications to compute the highest 64 and lowest 64 bits of multiplying a 32 and 64 bit number together
//adds the results to the contents of lo
__device__ void multiply32By64PlusLo(uint64 multiplicand, uint64 multiplier, uint64 * lo, uint64 * hi) {

	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t0, t1, t2, t3, v0, v1, v2, v3;\n\t"
		"mov.b64           {t0, t1}, %4;\n\t" //splits lo into t0 and t1
		"mov.b64           {v0, v1}, %2;\n\t" //splits a into hi and lo 32 bit words (although a has no high bits set, we just won't use v1)
		"mov.b64           {v2, v3}, %3;\n\t" //splits b into hi and lo 32 bit words
		"mad.lo.cc.u32      t0, v0, v2, t0;\n\t" //lolo = starting_value + lo(alo*blo) (with carry flag)
		"madc.hi.cc.u32     t1, v0, v2, t1;\n\t" //lohi = starting_value + hi(alo*blo) + 1 carry (with carry flag)
		"madc.hi.cc.u32     t2, v0, v3,  0;\n\t" //hilo = hi(alo*bhi) + 1 carry (with carry flag, as carry may need to propagate)
		"mad.lo.cc.u32      t1, v0, v3, t1;\n\t" //lohi = starting_value + lo(alo*bhi) + hi(alo*blo) + 1 carry (with carry flag)
		"addc.cc.u32           t2, t2, 0;\n\t" //hilo = hi(alo*bhi) + 2 carries (with carry flag, as carry may need to propagate)
		"addc.u32           t3, 0, 0;\n\t" //just incase the last line produced a carry
		"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"mov.b64            %1, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(*lo), "=l"(*hi)
		: "l"(multiplicand), "l"(multiplier), "l"(*lo));
}

//adds augend to addend
//if an overflow is detected, add maxMod to augend
//if that overflows, add it again (as long as the mod for which maxMod is defined is < 2^63, this can't overflow)
//this function allows the program to avoid calculating any modulus operations in modMultiply64Bit except once at the end
//doing this saves anywhere from 25-40% of runtime (with larger savings coming from larger digit calculations)
__device__ void addWithCarryConvertedToMod(uint64 & addend, const uint64 & augend, const uint64 & maxMod) {
	asm("{\n\t"
		".reg .u32         t0;\n\t"
		".reg .pred         %p;\n\t"
		"add.cc.u64        %0, %0, %1;\n\t" //addend = addend + augend
		"addc.u32          t0, 0, 0;\n\t"
		"setp.eq.u32       %p, 1, t0;\n\t"
		"@%p add.cc.u64   %0, %0, %2;\n\t" //if carry-flag set, addend = addend + augend + maxMod - 2^64
		"addc.u32          t0, 0, 0;\n\t"
		"setp.eq.u32       %p, 1, t0;\n\t"
		"@%p add.cc.u64   %0, %0, %2;\n\t" //if carry-flag set, addend = addend + augend + 2*maxMod - 2^65
		"}"
		: "=l"(addend)
		: "l"(augend), "l"(maxMod));
}

__device__ void multiplyAdd64Hi(const uint64 & multiplicand, const uint64 & multiplier, uint64 * accumulate) {
	asm("{\n\t"
		"mad.hi.u64          %0, %1, %2, %3;\n\t"
		"}"
		: "=l"(*accumulate)
		: "l"(multiplicand), "l"(multiplier), "l"(*accumulate));
}

//calculates the 128 bit product of multiplicand and multiplier
//takes the highest 64 bits and multiplies it by maxMod (2^64 % mod) and adds it to the low 64 bits, repeating until the highest 64 bits are zero
//this takes (log2(mod)) / (64 - log2(mod)) steps
//maxMod is constant with respect to each mod, therefore best place to calculate is in modExp functions
__device__ void modMultiply64Bit(uint64 multiplicand, uint64 multiplier, const uint64 & mod, const uint64 & maxMod, uint64 & output) {
	uint64 hi = 0, result = 0;// , lo;
	multiply64By64PlusLo(multiplicand, multiplier, &result, &hi);
	while (hi) {
		if(hi > 0xFFFFFFFF) multiply64By64PlusLo(hi, maxMod, &result, &hi);
		else multiply32By64PlusLo(hi, maxMod, &result, &hi);
	}
	if(result >= mod) result %= mod;
	output = result;
}

//greatest common denominator method pulled from http://www.hackersdelight.org/hdcodetxt/mont64.c.txt
//modified for use-case where R is always 2^64

/* C program implementing the extended binary GCD algorithm. C.f.
http://www.ucl.ac.uk/~ucahcjm/combopt/ext_gcd_python_programs.pdf. This
is a modification of that routine in that we find s and t s.t.
gcd(a, b) = s*a - t*b,
rather than the same expression except with a + sign.
This routine has been greatly simplified to take advantage of the
facts that in the MM use, argument a is a power of 2, and b is odd. Thus
there are no common powers of 2 to eliminate in the beginning. The
parent routine has two loops. The first drives down argument a until it
is 1, modifying u and v in the process. The second loop modifies s and
t, but because a = 1 on entry to the second loop, it can be easily seen
that the second loop doesn't alter u or v. Hence the result we want is u
and v from the end of the first loop, and we can delete the second loop.
The intermediate and final results are always > 0, so there is no
trouble with negative quantities. Must have a either 0 or a power of 2
<= 2**63. A value of 0 for a is treated as 2**64. b can be any 64-bit
value.
Parameter a is half what it "should" be. In other words, this function
does not find u and v st. u*a - v*b = 1, but rather u*(2a) - v*b = 1. */

__device__ void xbinGCD(uint64 b, uint64 *pv)
{
	uint64 alpha, beta, u, v;
	//printf("Doing GCD(%llx, %llx)\n", a, b);

	u = 1; v = 0;
	alpha = int64MaxBit; beta = b;         // Note that alpha is
								 // even and beta is odd.

								 /* The invariant maintained from here on is:
								 2a = u*2*alpha - v*beta. */

								 // printf("Before, a u v = %016llx %016llx %016llx\n", a, u, v);
	int stop = 64;
	while (stop) {
		stop--;
		if ((u & 1) == 0) {             // Delete a common
			u = u >> 1; v = v >> 1;      // factor of 2 in
		}                               // u and v.
		else {
			/* We want to set u = (u + beta) >> 1, but
			that can overflow, so we use Dietz's method. */
			u = ((u ^ beta) >> 1) + (u & beta);
			v = (v >> 1) + alpha;
		}
		//    printf("After,  a u v = %016llx %016llx %016llx\n", a, u, v);
	}

	// printf("At end,    a u v = %016llx %016llx %016llx\n", a, u, v);
	*pv = v;
	return;
}


//montgomery multiplication method from http://www.hackersdelight.org/hdcodetxt/mont64.c.txt
//slightly modified to use more efficient 64 bit multiply-adds in PTX assembly
__device__ void montgomeryMult(uint64 abar, uint64 bbar, uint64 mod, uint64 mprime, uint64 & output) {

	uint64 tlo = 0, tm = 0;
	//INT_64 thi = 0, tlo = 0, tm = 0, tmmhi = 0, tmmlo = 0, uhi = 0, ulo = 0, ov = 0;

	/* t = abar*bbar. */

	multiply64By64(abar, bbar, &tlo, &output);

	//unless tlo is zero here, there will always be a carry from tm*mod + tlo
	//this would only be a problem if thi was 2^64 - 1
	//which can never occur if mod is representable in an unsigned long long
	output += !!tlo;

	/* Now compute u = (t + ((t*mprime) & mask)*m) >> 64.
	The mask is fixed at 2**64-1. Because it is a 64-bit
	quantity, it suffices to compute the low-order 64
	bits of t*mprime, which means we can ignore thi. */

	//tm = tlo * mprime;
	multiply64By64LoOnly(tlo, mprime, &tm);
	
	//there is an optimization to be made here, tm = lo64(tlo * mprime)
	//so tm * mod = lo64(tlo * mprime) * mod
	//but mprime*mod is constant for a given mod
	//is there a way to reduce the amount of work from this?
	//multiply64By64PlusLo(tm, mod, &tlo, &tmmhi);
	multiply64By64PlusHi(tm, mod, &output);//tlo is not used
	//also if mod is < 2^63 this can't overflow
	
	//assumes mod < 2^63, WILL NOT WORK if mod > 2^63 because overflow can exist in above addition in that case
	//if (thi >= mod) thi -= mod;
	//in addition to mitigating most GPUs' poor conditional branching performance, unconditional code execution is also resistant to side-channel attacks
	output = output - (mod & -((output >= mod)));
}

//using left-to-right binary exponentiation
//the position of the highest bit in exponent is passed into the function as a parameter (it is more efficient to find it outside)
//uses montgomery multiplication to reduce difficulty of modular multiplication (runs in 55% of runtime of non-montgomery modular multiplication)
//montgomery multiplication suggested by njuffa
//now uses quarternary exponentiation, in an effort to halve the number of multplications required when exponent and bitmask match
__device__ void modExpLeftToRight(const uint64 & exp, const uint64 & mod, int shiftToLittleBits, uint64 & output) {

	if (!exp) {
		//no need to set output to anything as it is already 1
		return;
	}

	uint64 mPrime;

	//finds rInverse*2^64 - mPrime*mod = 1
	xbinGCD(mod, &mPrime);
	uint64 baseNonZeroPowers[3];

	uint64 maxMod = int64MaxBit % mod;

	maxMod <<= 1;
	
	if (maxMod > mod) maxMod -= mod;

	//baseSystem*2^64 % mod
	modMultiply64Bit(maxMod, baseSystem, mod, maxMod, baseNonZeroPowers[0]);

	montgomeryMult(baseNonZeroPowers[0], baseNonZeroPowers[0], mod, mPrime, baseNonZeroPowers[1]);//baseNonZeroPowers[1] = baseBar^2
	montgomeryMult(baseNonZeroPowers[1], baseNonZeroPowers[0], mod, mPrime, baseNonZeroPowers[2]);//baseNonZeroPowers[2] = baseBar^3

	int quarternaryDigit = ((exp >> shiftToLittleBits) & 3);
	output = baseNonZeroPowers[quarternaryDigit - 1];

	while (shiftToLittleBits) {

		montgomeryMult(output, output, mod, mPrime, output);//result^2
		montgomeryMult(output, output, mod, mPrime, output);//result^4

		shiftToLittleBits -= 2;

		quarternaryDigit = (exp >> shiftToLittleBits) & 3;

		if (quarternaryDigit) {
			montgomeryMult(output, baseNonZeroPowers[quarternaryDigit - 1], mod, mPrime, output);//result*base
		}
	}

	//convert result out of montgomery form
	montgomeryMult(output, 1, mod, mPrime, output);
}

//find ( baseSystem^n % mod ) / mod and add to partialSum
//experimented with placing forceinline and noinline on various functions again
//with new changes, noinline now has most effect here, no idea why
__device__ __noinline__ void fractionalPartOfSum(const uint64 & exp, const uint64 & mod, double *partialSum, int shift, const int & negative) {
	uint64 expModResult = 1;
	modExpLeftToRight(exp, mod, shift, expModResult);
	//if k is odd, then sumTerm will be negative
	//which just means we need to invert it about the modulus
	if (negative) expModResult = mod - expModResult;

	double sumTerm = (((double)expModResult) / ((double)mod));
	
	*partialSum += sumTerm;
	if((*partialSum) >= 1.0) *partialSum -= 1.0;
}

//stride over all parts of summation in bbp formula where k <= n
//to compute partial sJ sums
__device__ void bbp(uint64 n, uint64 start, uint64 end, uint64 stride, sJ* output, volatile uint64* progress, int progressCheck) {
	for (uint64 k = start; k <= end; k += stride) {
		uint64 exp = n - k;
		//shift represents number of bits needed to shift highest set bit pair in exp
		//into the lowest 2 bits
		int shift = 62 - __clzll(exp);
		//if shift is negative set to zero
		shift *= (shift > 0);
		//if shift is odd round up to nearest multiple of 2
		shift += shift & 1;
		uint64 mod = 4 * k + 1;
		fractionalPartOfSum(exp, mod, output->s, shift, k & 1);
		mod += 2;//4k + 3
		fractionalPartOfSum(exp, mod, output->s + 1, shift, k & 1);
		mod = 10 * k + 1;
		fractionalPartOfSum(exp, mod, output->s + 2, shift, k & 1);
		mod += 2;//10k + 3
		fractionalPartOfSum(exp, mod, output->s + 3, shift, k & 1);
		mod += 2;//10k + 5
		fractionalPartOfSum(exp, mod, output->s + 4, shift, k & 1);
		mod += 2;//10k + 7
		fractionalPartOfSum(exp, mod, output->s + 5, shift, k & 1);
		mod += 2;//10k + 9
		fractionalPartOfSum(exp, mod, output->s + 6 , shift, k & 1);
		if (!progressCheck) {
			//only 1 thread (with gridId 0 on GPU0) ever updates the progress
			*progress = k;
		}
	}
}

//determine from thread and block position where to begin stride
//only one of the threads per kernel (AND ONLY ON GPU0) will report progress
__global__ void bbpKernel(sJ *c, volatile uint64 *progress, uint64 digit, int gpuNum, uint64 begin, uint64 end, uint64 stride)
{
	int gridId = threadIdx.x + blockDim.x * blockIdx.x;
	uint64 start = begin + gridId + blockDim.x * gridDim.x * gpuNum;
	int progressCheck = gridId + blockDim.x * gridDim.x * gpuNum;
	bbp(digit, start, end, stride, c + gridId, progress, progressCheck);
}

//stride over current leaves of reduce tree
__global__ void reduceSJKernel(sJ *c, int offset, int stop) {
	int stride = blockDim.x * gridDim.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	while (i < stop) {
		int augend = i + offset;
		sJAdd(c + i, c + augend);
		i += stride;
	}
}

//standard tree-based parallel reduce
cudaError_t reduceSJ(sJ *c, unsigned int size) {
	cudaError_t cudaStatus;
	while (size > 1) {
		int nextSize = (size + 1) >> 1;

		//size is odd
		if (size&1) reduceSJKernel<< <32, 32 >> >(c, nextSize, nextSize - 1);
		//size is even
		else reduceSJKernel<< <32, 32 >> >(c, nextSize, nextSize);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduceSJKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceSJKernel!\n", cudaStatus);
			return cudaStatus;
		}

		size = nextSize;
	}
	return cudaStatus;
}

uint64 finalizeDigit(sJ input, uint64 n) {
	double reducer = 1.0;

	//unfortunately 64 is not a power of 16, so if n is < 2
	//then division is unavoidable
	//this division must occur before any modulus are taken
	if(n == 0) reducer /= 64.0;
	else if (n == 1) reducer /= 4.0;

	//logic relating to 1024 not being a power of 16 and having to divide by 64
	int loopLimit = (2 * n - 3) % 5;
	if (n < 2) n = 0;
	else n = (2 * n - 3) / 5;

	double trash = 0.0;
	double *s = input.s;
	for (int i = 0; i < 7; i++) s[i] *= reducer;
	
	if (n < 16000) {
		for (int i = 0; i < 5; i++) {
			n++;
			double sign = 1.0;
			double nD = (double)n;
			if (n & 1) sign = -1.0;
			reducer /= (double)baseSystem;
			s[0] += sign * reducer / (4.0 * nD + 1.0);
			s[1] += sign * reducer / (4.0 * nD + 3.0);
			s[2] += sign * reducer / (10.0 * nD + 1.0);
			s[3] += sign * reducer / (10.0 * nD + 3.0);
			s[4] += sign * reducer / (10.0 * nD + 5.0);
			s[5] += sign * reducer / (10.0 * nD + 7.0);
			s[6] += sign * reducer / (10.0 * nD + 9.0);
		}
	}

	//multiply sJs by coefficients from Bellard's formula and then find their fractional parts
	double coeffs[7] = { -32.0, -1.0, 256.0, -64.0, -4.0, -4.0, 1.0 };
	for (int i = 0; i < 7; i++) {
		s[i] = modf(coeffs[i] * s[i], &trash);
		if (s[i] < 0.0) s[i]++;
	}

	double hexDigit = 0.0;
	for (int i = 0; i < 7; i++) hexDigit += s[i];
	hexDigit = modf(hexDigit, &trash);
	if (hexDigit < 0) hexDigit++;

	//16^n is divided by 64 and then combined into chunks of 1024^m
	//where m is = (2n - 3)/5
	//because 5 may not evenly divide this, the remaining 4^((2n - 3)%5)
	//must be multiplied into the formula at the end
	for (int i = 0; i < loopLimit; i++) hexDigit *= 4.0;
	hexDigit = modf(hexDigit, &trash);

	//shift left by 8 hex digits
	for (int i = 0; i < 12; i++) hexDigit *= 16.0;
	printf("hexDigit = %.8f\n", hexDigit);
	return (uint64)hexDigit;
}

int checkForProgressCache(uint64 digit, uint64 * contFrom, sJ * cache, double * previousTime) {
	std::string target = "digit" + std::to_string(digit) + "Base";
	std::string pToFile;
	std::vector<std::string> matching;
	int found = 0;
	for (auto& element : std::experimental::filesystem::directory_iterator("progressCache")) {
		std::string name = element.path().filename().string();
		//filename begins with desired string
		if (name.compare(0, target.length(), target) == 0) {
			matching.push_back(element.path().string());
			found = 1;
		}
	}
	if (found) {
		//sort and choose alphabetically last result
		std::sort(matching.begin(), matching.end());
		pToFile = matching.back();

		int chosen = 0;
		while (!chosen) {
			chosen = 1;
			std::cout << "A cache of a previous computation for this digit exists." << std::endl;
			std::cout << "Would you like to reload the most recent cache (" << pToFile << ")? y\\n" << std::endl;
			char choice;
			std::cin >> choice;
			if (choice == 'y') {
				std::cout << "Loading cache and continuing computation." << std::endl;
				FILE * cacheF = fopen(pToFile.c_str(), "r");

				if (cacheF == NULL) {
					std::cout << "Could not open " << pToFile << "!" << std::endl;
					std::cout << "Beginning computation without reloading." << std::endl;
					return 1;
				}

				int readLines = 0;

				readLines += fscanf(cacheF, "%llu", contFrom);
				readLines += fscanf(cacheF, "%la", previousTime);
				for (int i = 0; i < 7; i++) readLines += fscanf(cacheF, "%la", &cache->s[i]);
				fclose(cacheF);
				//9 lines of data should have been read, 1 continuation point, 1 time, and 7 data points
				if (readLines != 9) {
					std::cout << "Data reading failed!" << std::endl;
					return 1;
				}
			}
			else if (choice == 'n') {
				std::cout << "Beginning computation without reloading." << std::endl;
			}
			else {
				std::cout << "Invalid input" << std::endl;
				// Ignore to the end of line
				std::cin.clear();
				std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
				chosen = 0;
			}
		}
	}
	else {
		std::cout << "No progress cache file found. Beginning computation without reloading." << std::endl;
	}
	return 0;
}

int main()
{
	try {
		const int arraySize = threadCountPerBlock * blockCount;
		uint64 hexDigitPosition;
		std::cout << "Input hexDigit to calculate (1-indexed):" << std::endl;
		std::cin >> hexDigitPosition;
		//subtract 1 to convert to 0-indexed
		hexDigitPosition--;

		uint64 sumEnd = 0;

		//convert from number of digits in base16 to base1024
		//because of the 1/64 in formula, we must subtract log16(64) which is 1.5, so carrying the 2 * (digitPosition - 1.5) = 2 * digitPosition - 3
		//this is because division messes up with respect to modulus, so use the 16^digitPosition to absorb it
		if (hexDigitPosition < 2) sumEnd = 0;
		else sumEnd = ((2LLU * hexDigitPosition) - 3LLU) / 5LLU;

		uint64 beginFrom = 0;
		sJ cudaResult;
		double previousTime = 0.0;
		if (checkForProgressCache(sumEnd, &beginFrom, &cudaResult, &previousTime)) {
			return 1;
		}

		std::thread handles[totalGpus];
		BBPLAUNCHERDATA gpuData[totalGpus];

		PPROGRESSDATA prog = setupProgress();

		chr::high_resolution_clock::time_point start = chr::high_resolution_clock::now();

		if (prog->error != cudaSuccess) return 1;
		prog->begin = &start;
		prog->maxProgress = sumEnd;
		prog->previousCache = cudaResult;
		prog->previousTime = previousTime;

		std::thread progThread(progressCheck, prog);

		for (int i = 0; i < totalGpus; i++) {

			gpuData[i].digit = sumEnd;
			gpuData[i].gpu = i;
			gpuData[i].totalGpus = totalGpus;
			gpuData[i].size = arraySize;
			gpuData[i].deviceProg = prog->deviceProg;
			gpuData[i].status = &(prog->status[i]);
			gpuData[i].dataWritten = &(prog->dataWritten);
			gpuData[i].nextStrideBegin = &(prog->nextStrideBegin[i]);
			gpuData[i].beginFrom = beginFrom;

			handles[i] = std::thread(cudaBbpLauncher, &(gpuData[i]));
		}

		cudaError_t cudaStatus;

		for (int i = 0; i < totalGpus; i++) {

			handles[i].join();

			cudaStatus = gpuData[i].error;
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaBbpLaunch failed on gpu%d!\n", i);
				return 1;
			}

			sJ output = gpuData[i].output;

			//sum results from gpus
			sJAdd(&cudaResult, &output);
		}

		//tell the progress thread to quit
		prog->quit = 1;

		progThread.join();

		free(prog);

		uint64 hexDigit = finalizeDigit(cudaResult, hexDigitPosition);

		chr::high_resolution_clock::time_point end = chr::high_resolution_clock::now();

		printf("pi at hexadecimal digit %llu is %012llX\n",
			hexDigitPosition + 1, hexDigit);

		//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
		printf("Computed in %.8f seconds\n", previousTime + (chr::duration_cast<chr::duration<double>>(end - start)).count());

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
		}

		return 0;
	}
	catch(...) {
		printf("oops xD\n");
		return 1;
	}
}

PPROGRESSDATA setupProgress() {
	PPROGRESSDATA threadData = new PROGRESSDATA();

	std::atomic_init(&threadData->dataWritten, 0);

	//these variables are linked between host and device memory allowing each to communicate about progress
	volatile uint64 *currProgHost, *currProgDevice;

	//allow device to map host memory for progress ticker
	threadData->error = cudaSetDeviceFlags(cudaDeviceMapHost);
	if (threadData->error != cudaSuccess) {
		fprintf(stderr, "cudaSetDeviceFlags failed with error: %s\n", cudaGetErrorString(threadData->error));
		return threadData;
	}

	// Allocate Host memory for progress ticker
	threadData->error = cudaHostAlloc((void**)&currProgHost, sizeof(uint64), cudaHostAllocMapped);
	if (threadData->error != cudaSuccess) {
		fprintf(stderr, "cudaHostAalloc failed!");
		return threadData;
	}

	//create link between between host and device memory for progress ticker
	threadData->error = cudaHostGetDevicePointer((uint64 **)&currProgDevice, (uint64 *)currProgHost, 0);
	if (threadData->error != cudaSuccess) {
		fprintf(stderr, "cudaHostGetDevicePointer failed!");
		return threadData;
	}

	*currProgHost = 0;

	threadData->deviceProg = currProgDevice;
	threadData->currentProgress = currProgHost;
	threadData->quit = 0;

	return threadData;
}

//this function is meant to be run by an independent thread to output progress to the console
void progressCheck(PPROGRESSDATA progP) {

	std::deque<double> progressQ;
	std::deque<chr::high_resolution_clock::time_point> timeQ;
	int count = 0;
	while(!progP->quit) {
		count++;
		double progress = (double)(*(progP->currentProgress)) / (double)progP->maxProgress;

		chr::high_resolution_clock::time_point now = chr::high_resolution_clock::now();
		progressQ.push_front(progress);
		timeQ.push_front(now);

		//progressQ and timeQ should be same size at all times
		if (progressQ.size() > 100) {
			progressQ.pop_back();
			timeQ.pop_back();
		}

		double progressInPeriod = progressQ.front() - progressQ.back();
		double elapsedPeriod = chr::duration_cast<chr::duration<double>>(timeQ.front() - timeQ.back()).count();
		double progressPerSecond = progressInPeriod / elapsedPeriod;

		double timeEst = (1.0 - progress) / (progressPerSecond);
		//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
		double time = progP->previousTime + (chr::duration_cast<chr::duration<double>>(now - *progP->begin)).count();
		//only print every 10th cycle or 0.1 seconds
		if (count == 10) {
			count = 0;
			printf("Current progress is %3.3f%%. Estimated total runtime remaining is %8.3f seconds. Avg rate is %1.5f%%. Time elapsed is %8.3f seconds.\n", 100.0*progress, timeEst, 100.0*progressPerSecond, time);
		}

		int expected = totalGpus;

		if (std::atomic_compare_exchange_strong(&progP->dataWritten, &expected, 0)) {

			//ensure all sJs in status are from same stride
			//this should always be the case since each 1000 strides are separated by about 90 seconds currently
			//it would be very unlikely for one gpu to get 1000 strides ahead of another, unless the GPUs were not the same
			int sJsAligned = 1;
			uint64 contProcess = progP->nextStrideBegin[0];
			for (int i = 1; i < totalGpus; i++) sJsAligned &= (progP->nextStrideBegin[i] == contProcess);
			
			if (sJsAligned) {

				char buffer[100];

				double savedProgress = (double) (contProcess - 1LLU) / (double)progP->maxProgress;

				snprintf(buffer, sizeof(buffer), "progressCache/digit%lluBase1024Progress%09.6f.dat", progP->maxProgress, 100.0*savedProgress);

				//would like to do this with ofstream and std::hexfloat
				//but msvc is a microsoft product so...
				FILE * file;
				file = fopen(buffer, "w+");
				if(file != NULL) {
					printf("Writing data to disk\n");
					fprintf(file,"%llu\n",contProcess);
					fprintf(file, "%a\n", time);
					sJ currStatus = progP->previousCache;
					for (int i = 0; i < totalGpus; i++) {
						sJAdd(&currStatus, progP->status + i);
					}
					for(int i = 0; i < 7; i++) fprintf(file, "%a\n", currStatus.s[i]);
					fclose(file);
				}
				else {
					fprintf(stderr, "Error opening file %s\n", buffer);
				}
			}
			else {
				fprintf(stderr, "sJs are misaligned, could not write to disk!\n");
				for (int i = 0; i < totalGpus; i++) fprintf(stderr, "sJ %d alignment is %llu\n", i, progP->nextStrideBegin[i]);
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}
}

// Helper function for using CUDA
void cudaBbpLauncher(PBBPLAUNCHERDATA data)//cudaError_t addWithCuda(sJ *output, unsigned int size, TYPE digit)
{
	int size = data->size;
	int gpu = data->gpu;
	int totalGpus = data->totalGpus;
	uint64 digit = data->digit;
	volatile uint64 * currProgDevice = data->deviceProg;
	sJ *dev_c = 0;
	sJ* c = new sJ[1];
	sJ *dev_ex = 0;

	cudaError_t cudaStatus;
	
	uint64 stride, launchWidth, neededLaunches;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(gpu);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffer for temp vector
	cudaStatus = cudaMalloc((void**)&dev_ex, size * sizeof(sJ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffer for output vector
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(sJ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	stride =  (uint64) size * (uint64) totalGpus;

	launchWidth = stride * 64LLU;

	//need to round up
	//because bbp condition for stopping is <= digit, number of total elements in summation is 1 + digit
	//even when digit/launchWidth is an integer, it is necessary to add 1
	neededLaunches = ((digit - data->beginFrom) / launchWidth) + 1LLU;

	for (uint64 launch = 0; launch < neededLaunches; launch++) {

		uint64 begin = data->beginFrom + (launchWidth * launch);
		uint64 end = data->beginFrom + (launchWidth * (launch + 1)) - 1;
		if (end > digit) end = digit;

		// Launch a kernel on the GPU with one thread for each element.
		bbpKernel << <blockCount, threadCountPerBlock >> > (dev_c, currProgDevice, digit, gpu, begin, end, stride);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "bbpKernel launch failed on gpu%d: %s\n", gpu, cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bbpKernel on gpu 1!\n", cudaStatus);
			goto Error;
		}

		//on every 1000th launch write data to status buffer for progress thread to save
		if (launch % 1000 == 0 && launch) {

			//copy current results into temp array to reduce and update status
			cudaStatus = cudaMemcpy(dev_ex, dev_c, size * sizeof(sJ), cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed in status update!\n");
				goto Error;
			}

			cudaStatus = reduceSJ(dev_ex, size);

			if (cudaStatus != cudaSuccess) {
				goto Error;
			}

			// Copy result (reduced into first element) from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(c, dev_ex, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed in status update!\n");
				goto Error;
			}

			*(data->status) = c[0];
			*(data->nextStrideBegin) = data->beginFrom + (launchWidth * (launch + 1LLU));
			std::atomic_fetch_add(data->dataWritten, 1);
		}

		//give the rest of the computer some gpu time to reduce system choppiness
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	cudaStatus = reduceSJ(dev_c, size);

	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	// Copy result (reduced into first element) from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	(*data).output = c[0];

Error:
	free(c);
	cudaFree(dev_c);
	cudaFree(dev_ex);

	(*data).error = cudaStatus;
}
