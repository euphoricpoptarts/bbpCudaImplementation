
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <Windows.h>
#include <deque>

#define TYPE unsigned long long
#define INT_64 unsigned long long

struct sJ {
	double s1 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
	double s4k1 = 0.0, s4k3 = 0.0, s10k1 = 0.0, s10k3 = 0.0, s10k5 = 0.0, s10k7 = 0.0, s10k9 = 0.0;
};

typedef struct {
	volatile INT_64 *currentProgress;
	TYPE maxProgress;
	int quit = 0;
} PROGRESSDATA, *PPROGRESSDATA;

typedef struct {
	sJ output;
	INT_64 digit;
	int gpu = 0;
	int totalGpus = 0;
	int size = 0;
	cudaError_t error;
} BBPLAUNCHERDATA, *PBBPLAUNCHERDATA;

DWORD WINAPI cudaBbpLauncher(LPVOID dataV);

//warpsize is 32 so optimal value is probably always a multiple of 32
const int threadCountPerBlock = 64;
//this is more difficult to optimize but seems to not like odd numbers
const int blockCount = 560;

__device__ __constant__ const TYPE baseSystem = 1024;
__device__  __constant__ const int baseExpOf2 = 10;

__device__ const int typeSize = sizeof(TYPE) * 8 - 1;
__device__ const TYPE multiplyModCond = 0x4000000000000000;//2^62
//__device__ const int int64Size = sizeof(INT_64) * 8 - 1;
__device__ const INT_64 int64ModCond = 0x40000000;
__device__  __constant__ const INT_64 int64MaxBit = 0x8000000000000000;

__device__ int printOnce = 0;

//not actually quick
__device__ void quickMod(INT_64 input, const INT_64 mod, INT_64 *output) {

	/*INT_64 copy = input;
	INT_64 test = input % mod;*/
	INT_64 temp = mod;
	while (temp < input && !(temp&int64MaxBit)) temp <<= 1;
	if (temp > input) temp >>= 1;
	while (input >= mod && temp >= mod) {
		if (input >= temp) input -= temp;
		temp >>= 1;
	}
	/*if (input != test && !atomicAdd(&printOnce,1))
	{
	printf("input %llu mod %llu error\n", copy, mod);
	printOnce = 1;
	}*/
	*output = input;
}

//binary search to find highest 1 bit in multiplier
__device__ void findMultiplierHighestBit(const TYPE multiplier, TYPE *output) {

	//if no bits are 1 then highest bit doesn't exist
	if (!multiplier) {
		*output = 0;
		return;
	}

	int highestBitLocMax = typeSize;
	int highestBitLocMin = 0;

	int middle = (highestBitLocMax + highestBitLocMin) >> 1;

	TYPE highestBit = 1L;
	highestBit <<= middle;

	int less = highestBit <= multiplier;

	while (!((highestBit << 1) > multiplier && less)) {
		if (less) highestBitLocMin = middle + 1;
		else highestBitLocMax = middle - 1;
		middle = (highestBitLocMax + highestBitLocMin) >> 1;
		//this might not look necessary but it is
		highestBit = 1L;
		highestBit <<= middle;
		less = highestBit <= multiplier;
	}

	/*unsigned long long highestBit2 = 0x8000000000000000;

	while (highestBit2 > multiplier) highestBit2 >>= 1;

	if (highestBit != highestBit2 && !printOnce) {
	printf("multiplier %d error; highestBit %d; highestBit2 %d\n", multiplier, highestBit, highestBit2);
	printOnce = 1;
	}*/

	*output = highestBit;
}

//hacker's delight method to find highest bit in a long long (it just works)
//http://graphics.stanford.edu/~seander/bithacks.html
//just barely faster than built-in CUDA __clzll
__device__ void findMultiplierHighestBitHackersDelight(TYPE multiplier, TYPE *output) {

	multiplier |= multiplier >> 1;
	multiplier |= multiplier >> 2;
	multiplier |= multiplier >> 4;
	multiplier |= multiplier >> 8;
	multiplier |= multiplier >> 16;
	multiplier |= multiplier >> 32;

	*output = multiplier ^ (multiplier >> 1);

}

__device__ void modMultiplyLeftToRight(const TYPE multiplicand, const TYPE multiplier, TYPE mod, TYPE *output) {
	*output = multiplicand;

	TYPE highestBitMask = 0;

	findMultiplierHighestBitHackersDelight(multiplier, &highestBitMask);

	while (highestBitMask > 1) {
		//only perform modulus operation during loop if result is >= (TYPE maximum + 1)/4 (in order to prevent overflowing)
		if (*output >= multiplyModCond) *output %= mod;
		*output <<= 1;
		highestBitMask >>= 1;
		if (multiplier&highestBitMask)	*output += multiplicand;
	}

	//modulus must be taken after loop as it hasn't necessarily been taken during last loop iteration
	*output %= mod;
}

__device__ void modMultiplyRightToLeft(INT_64 multiplicand, INT_64 multiplier, INT_64 mod, INT_64 *output) {
	INT_64 result = 0;

	INT_64 mask = 1;

	while (multiplier > 0) {
		if (multiplier&mask) {
			result += multiplicand;

			//only perform modulus operation during loop if result is >= (INT_64 maximum + 1)/2 (in order to prevent overflowing)
			if (result >= mod) result -= mod;
		}
		multiplicand <<= 1;
		if (multiplicand >= mod) multiplicand -= mod;
		multiplier >>= 1;
	}

	//modulus must be taken after loop as it hasn't necessarily been taken during last loop iteration
	result %= mod;
	*output = result;
}

//leverages a machine instruction that returns the highest 64 bits of the multiplication operation
//multiplicand and multiplier should always be less than mod (may work correctly even if this is not the case)
//maxMod is constant with respect to each mod, therefore best place to calculate is in modExp functions
__device__ void modMultiply64Bit(INT_64 multiplicand, INT_64 multiplier, INT_64 mod, INT_64 maxMod, INT_64* output) {
	INT_64	hi = __umul64hi(multiplicand, multiplier);
	INT_64 result = (multiplicand * multiplier) % mod;
	while (hi) {
		INT_64 lo = hi * maxMod;

		//multiplyModCond should be (2^64)/(number of loop iterations)
		//where loop iterations are roughly 64/(64 - log2(mod))
		//THEREFORE THIS SHOULD NOT BE A COMPILE TIME CONSTANT
		//but a runtime variable set at launch based upon the maximum mod that will be passed to this function
		//for 2^40 number of loops is 2, for 2^50 number of loops is 4
		if (lo > multiplyModCond) lo %= mod;

		result +=  lo;
		hi = __umul64hi(hi, maxMod);
	}
	if(result >= mod) result %= mod;
	*output = result;
}

//an experiment to see if reducing the number of arguments saves any time
__device__ void modSquare64Bit(INT_64 *number, INT_64 mod, INT_64 maxMod) {
	INT_64	hi = __umul64hi(*number, *number);
	*number = (*number * *number) % mod;
	while (hi) {
		INT_64 lo = hi * maxMod;

		//multiplyModCond should be (2^64)/(number of loop iterations)
		//where loop iterations are roughly 64/(64 - log2(mod))
		//THEREFORE THIS SHOULD NOT BE A COMPILE TIME CONSTANT
		//but a runtime variable set at launch based upon the maximum mod that will be passed to this function
		//for 2^40 number of loops is 2, for 2^50 number of loops is 4
		if (lo > multiplyModCond) lo %= mod;

		*number += lo;
		hi = __umul64hi(hi, maxMod);
	}
	if(*number >= mod) *number %= mod;
}

//leverages a machine instruction that returns the highest 64 bits of the multiplication operation
//multiplicand and multiplier should always be less than mod (may work correctly even if this is not the case)
//uses bitshifts and subtraction to avoid multiplications and modulus respectively inside the loop
//loops more times than other version
//slower than other version (but could be faster if mod is close to 2^63)
__device__ void modMultiply64BitAlt(INT_64 multiplicand, INT_64 multiplier, INT_64 mod, const int modMaxBitPos, INT_64 *output) {
	INT_64	hi = __umul64hi(multiplicand, multiplier);
	INT_64 result = (multiplicand * multiplier) % mod;
	int count = 64;
	while (count > 0) {

		//determine the number of bits to shift hi so that 2*mod > hi > mod
		int dif = __clzll(hi) - modMaxBitPos;

		if (dif > count) dif = count;

		//hi is the highest 64 bits of multiplicand*multiplier
		//so hi is actually hi*2^64
		//takes bits from 2^64 and gives them to hi until 2^64 is reduced to 2^0
		//each step of loop only gives as many bits to hi as to satisfy 2*mod > hi > mod
		hi <<= dif;

		if(hi >= mod) hi -= mod;

		count -= dif;
	}
	*output = (result + hi);
	if(*output > mod) *output -= mod;
}

//perform right-to-left binary exponention taking modulus of both base and result at each step
//64 bit integers are required to accurately find the modular exponents of numbers when mod is >= ~10e6
//however, with CUDA 64 bit integers are implemented at compile time as two 32 bit integers
//this produces about a 10x slowdown over computations using 32 bit integers
__device__ void modExp(unsigned long long base, long exp, long mod, long *output) {
	const unsigned long mask = 1;
	unsigned long long result = 1;

	//only perform modulus operation during loop if result or base is >= 2^32 (in order to prevent either from overflowing)
	//this saves 30% computation time over performing modulus in every loop iteration
	const unsigned long long modCond = 0x100000000;//2^32

	while (exp > 0) {
		if (exp&mask) {
			result *= base;
			if (result >= modCond) result %= mod;
		}
		base *= base;
		if (base >= modCond) base %= mod;
		exp >>= 1;
	}

	//modulus must be taken after loop as it hasn't necessarily been taken during last loop iteration
	result %= mod;
	*output = result;
}

//using left-to-right binary exponentiation
//the position of the highest bit in exponent is passed into the function as a parameter (it is more efficient to find it outside)
//this version allows base to be constant, thus reducing total number of moduli which must be calculated
//geometric mean of multiplication inputs is also substantially lower, allowing faster average multiplications
//experimented with placing forceinline and noinline on various functions, noinline here had the biggest improvement, no idea why
__device__ __noinline__ void modExpLeftToRight(const TYPE exp, TYPE mod, TYPE highestBitMask, TYPE* output) {

	if (!exp) {
		//no need to set output to anything as it is already 1
		return;
	}

	INT_64 result = baseSystem;

	INT_64 maxMod = int64MaxBit % mod;

	maxMod <<= 1;

	if (maxMod > mod) maxMod %= mod;

	while (highestBitMask > 1) {
		
		//modMultiply expect inputs to be less than mod
		if (result >= mod) result %= mod;


		modMultiply64Bit(result, result, mod, maxMod, &result);//result^2
		//modSquare64Bit(output, mod, maxMod);//result^2

		highestBitMask >>= 1;
		if (exp&highestBitMask)	result <<= baseExpOf2;//result * base
	}

	//modulus may need to be taken after loop as it hasn't necessarily been taken during last loop iteration
	result %= mod;

	*output = result;
}

//find ( baseSystem^n % mod ) / mod and add to partialSum
__device__ void fractionalPartOfSum(TYPE exp, TYPE mod, double *partialSum, TYPE highestBitMask, int negative) {
	TYPE expModResult = 1;
	modExpLeftToRight(exp, mod, highestBitMask, &expModResult);
	double sumTerm = (((double)expModResult) / ((double)mod));
	
	//if n is odd, then sumTerm will be negative
	//add 1 to it to find its positive fractional part
	if (negative) sumTerm = 1.0 - sumTerm;
	*partialSum += sumTerm;
	if((*partialSum) > 1.0) *partialSum -= (int)(*partialSum);
}

//stride over all parts of summation in bbp formula where k <= n
//to compute partial sJ sums
__device__ void bbp(TYPE n, TYPE start, INT_64 end, int gridId, TYPE stride, sJ* output, volatile INT_64* progress, int progressCheck) {

	double trash = 0.0;
	TYPE highestExpBit = 1;
	while (highestExpBit <= n)	highestExpBit <<= 1;
	for (TYPE k = start; k <= end; k += stride) {
		while (highestExpBit > (n - k))  highestExpBit >>= 1;
		TYPE mod = 4 * k + 1;
		fractionalPartOfSum(n - k, mod, &((*output).s4k1), highestExpBit, k & 1);
		mod += 2;//4k + 3
		fractionalPartOfSum(n - k, mod, &((*output).s4k3), highestExpBit, k & 1);
		mod = 10 * k + 1;
		fractionalPartOfSum(n - k, mod, &((*output).s10k1), highestExpBit, k & 1);
		mod += 2;//10k + 3
		fractionalPartOfSum(n - k, mod, &((*output).s10k3), highestExpBit, k & 1);
		mod += 2;//10k + 5
		fractionalPartOfSum(n - k, mod, &((*output).s10k5), highestExpBit, k & 1);
		mod += 2;//10k + 7
		fractionalPartOfSum(n - k, mod, &((*output).s10k7), highestExpBit, k & 1);
		mod += 2;//10k + 9
		fractionalPartOfSum(n - k, mod, &((*output).s10k9), highestExpBit, k & 1);
		if (!progressCheck) {
			//only 1 thread (with gridId 0 on GPU0) ever updates the progress
			*progress = k;
		}
	}
}

//determine from thread and block position where to begin stride
//only one of the threads per kernel (AND ONLY ON GPU0) will report progress
__global__ void bbpKernel(sJ *c, volatile INT_64 *progress, TYPE digit, int gpuNum, INT_64 begin, INT_64 end, INT_64 stride)
{
	int gridId = threadIdx.x + blockDim.x * blockIdx.x;
	TYPE start = begin + gridId + blockDim.x * gridDim.x * gpuNum;
	int progressCheck = gridId + blockDim.x * gridDim.x * gpuNum;
	bbp(digit, start, end, gridId, stride, c + gridId, progress, progressCheck);
}

//stride over current leaves of reduce tree
__global__ void reduceSJKernel(sJ *c, int offset, int stop) {
	int stride = blockDim.x * gridDim.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	while (i < stop) {
		int augend = i + offset;
		c[i].s1 += c[augend].s1;
		c[i].s4 += c[augend].s4;
		c[i].s5 += c[augend].s5;
		c[i].s6 += c[augend].s6;
		c[i].s4k1 += c[augend].s4k1;
		c[i].s4k3 += c[augend].s4k3;
		c[i].s10k1 += c[augend].s10k1;
		c[i].s10k3 += c[augend].s10k3;
		c[i].s10k5 += c[augend].s10k5;
		c[i].s10k7 += c[augend].s10k7;
		c[i].s10k9 += c[augend].s10k9;
		if (c[i].s4k1 > 1) c[i].s4k1 -= (int)c[i].s4k1;
		if (c[i].s4k3 > 1) c[i].s4k3 -= (int)c[i].s4k3;
		if (c[i].s10k1 > 1) c[i].s10k1 -= (int)c[i].s10k1;
		if (c[i].s10k3 > 1) c[i].s10k3 -= (int)c[i].s10k3;
		if (c[i].s10k5 > 1) c[i].s10k5 -= (int)c[i].s10k5;
		if (c[i].s10k7 > 1) c[i].s10k7 -= (int)c[i].s10k7;
		if (c[i].s10k9 > 1) c[i].s10k9 -= (int)c[i].s10k9;
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

//compute four steps of sJ sums for i > n and add to sJ sums found previously
//combine sJs according to bbp formula
//multiply by 16^5 to extract five digits of pi starting at n
long finalizeDigit(sJ input, TYPE n) {
	double reducer = 1.0;
	double s1 = input.s1;
	double s4 = input.s4;
	double s5 = input.s5;
	double s6 = input.s6;
	double trash = 0.0;
	if (n < 16000) {
		for (int i = 0; i < 4; i++) {
			n++;
			reducer /= (double)baseSystem;
			double eightN = 8.0 * n;
			s1 += reducer / (eightN + 1.0);
			s4 += reducer / (eightN + 4.0);
			s5 += reducer / (eightN + 5.0);
			s6 += reducer / (eightN + 6.0);
		}
	}
	//remove any integer part of s1-s6
	s1 = modf(s1, &trash);
	s4 = modf(s4, &trash);
	s5 = modf(s5, &trash);
	s6 = modf(s6, &trash);
	double hexDigit = 4.0*s1 - 2.0*s4 - s5 - s6;
	hexDigit = modf(hexDigit, &trash);
	if (hexDigit < 0) hexDigit++;
	hexDigit *= baseSystem*baseSystem*baseSystem*baseSystem*baseSystem;
	printf("hexDigit = %.8f\n", hexDigit);
	return (long)hexDigit;
}

long finalizeDigitAlt(sJ input, TYPE n) {
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
	double s4k1 = input.s4k1 * reducer;//modf(input.s4k1, &trash);
	double s4k3 = input.s4k3 * reducer;//modf(input.s4k3, &trash);
	double s10k1 = input.s10k1 * reducer;//modf(input.s10k1, &trash);
	double s10k3 = input.s10k3 * reducer;//modf(input.s10k3, &trash);
	double s10k5 = input.s10k5 * reducer;//modf(input.s10k5, &trash);
	double s10k7 = input.s10k7 * reducer;//modf(input.s10k7, &trash);
	double s10k9 = input.s10k9 * reducer;//modf(input.s10k9, &trash);
	
	if (n < 16000) {
		for (int i = 0; i < 5; i++) {
			n++;
			double sign = 1.0;
			double nD = (double)n;
			if (n & 1) sign = -1.0;
			reducer /= (double)baseSystem;
			s4k1 += sign * reducer / (4.0 * nD + 1.0);
			s4k3 += sign * reducer / (4.0 * nD + 3.0);
			s10k1 += sign * reducer / (10.0 * nD + 1.0);
			s10k3 += sign * reducer / (10.0 * nD + 3.0);
			s10k5 += sign * reducer / (10.0 * nD + 5.0);
			s10k7 += sign * reducer / (10.0 * nD + 7.0);
			s10k9 += sign * reducer / (10.0 * nD + 9.0);
		}
	}

	//multiply sJs by coefficients from Bellard's formula and then find their fractional parts
	s4k1 = modf(-32.0*s4k1, &trash);
	if (s4k1 < 0) s4k1++;
	s4k3 = modf(-1.0*s4k3, &trash);
	if (s4k3 < 0) s4k3++;
	s10k1 = modf(256.0*s10k1, &trash);
	if (s10k1 < 0) s10k1++;
	s10k3 = modf(-64.0*s10k3, &trash);
	if (s10k3 < 0) s10k3++;
	s10k5 = modf(-4.0*s10k5, &trash);
	if (s10k5 < 0) s10k5++;
	s10k7 = modf(-4.0*s10k7, &trash);
	if (s10k7 < 0) s10k7++;
	s10k9 = modf(s10k9, &trash);
	if (s10k9 < 0) s10k9++;

	double hexDigit = s4k1 + s4k3 + s10k1 + s10k3 + s10k5 + s10k7 + s10k9;
	hexDigit = modf(hexDigit, &trash);
	if (hexDigit < 0) hexDigit++;

	//16^n is divided by 64 and then combined into chunks of 1024^m
	//where m is = (2n - 3)/5
	//because 5 may not evenly divide this, the remaining 4^((2n - 3)%5)
	//must be multiplied into the formula at the end
	for (int i = 0; i < loopLimit; i++) hexDigit *= 4.0;
	hexDigit = modf(hexDigit, &trash);

	//shift left by 5 hex digits
	hexDigit *= 16.0*16.0*16.0*16.0*16.0;
	printf("hexDigit = %.8f\n", hexDigit);
	return (long)hexDigit;
}

int main()
{
	try {
		const int arraySize = threadCountPerBlock * blockCount;
		const TYPE digitPosition = 9999999999;
		const int totalGpus = 2;
		HANDLE handles[totalGpus];
		BBPLAUNCHERDATA gpuData[totalGpus];

		clock_t start = clock();

		for (int i = 0; i < totalGpus; i++) {

			//convert from number of digits in base16 to base1024
			//because of the 1/64 in formula, we must subtract log16(64) which is 1.5, so carrying the 2 * (digitPosition - 1.5) = 2 * digitPosition - 3
			//this is because division messes up with respect to modulus, so use the 16^digitPosition to absorb it
			if (digitPosition < 2) gpuData[i].digit = 0;
			else gpuData[i].digit = ((2LLU * digitPosition) - 3LLU) / 5LLU;
			gpuData[i].gpu = i;
			gpuData[i].totalGpus = totalGpus;
			gpuData[i].size = arraySize;


			handles[i] = CreateThread(NULL, 0, *cudaBbpLauncher, (LPVOID)&(gpuData[i]), 0, NULL);

			if (handles[i] == NULL) {
				fprintf(stderr, "gpu%dThread failed to launch\n", i);
				return 1;
			}
		}

		sJ cudaResult;

		cudaError_t cudaStatus;

		for (int i = 0; i < totalGpus; i++) {

			WaitForSingleObject(handles[i], INFINITE);
			CloseHandle(handles[i]);

			cudaError_t cudaStatus = gpuData[i].error;
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaBbpLaunch failed on gpu%d!\n", i);
				return 1;
			}

			sJ output = gpuData[i].output;

			//sum results from gpus
			cudaResult.s1 += output.s1;
			cudaResult.s4 += output.s4;
			cudaResult.s5 += output.s5;
			cudaResult.s6 += output.s6;
			cudaResult.s4k1 += output.s4k1;
			cudaResult.s4k3 += output.s4k3;
			cudaResult.s10k1 += output.s10k1;
			cudaResult.s10k3 += output.s10k3;
			cudaResult.s10k5 += output.s10k5;
			cudaResult.s10k7 += output.s10k7;
			cudaResult.s10k9 += output.s10k9;
		}

		long hexDigit = finalizeDigitAlt(cudaResult, digitPosition);

		clock_t end = clock();

		printf("pi at hexadecimal digit %llu is %05X\n",
			digitPosition + 1, hexDigit);

		printf("Computed in %.8f seconds\n", (double)(end - start) / CLOCKS_PER_SEC);

		// cudaDeviceReset must be called before exiting in order for profiling and
		// tracing tools such as Nsight and Visual Profiler to show complete traces.
		cudaStatus = cudaDeviceReset();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceReset failed!");
			return 1;
		}

		return 0;
	}
	catch(...) {
		printf("oops xD\n");
		return 1;
	}
}

//this function is meant to be run by an independent thread to output progress to the console
DWORD WINAPI progressCheck(LPVOID data) {
	PPROGRESSDATA progP = (PPROGRESSDATA)data;

	double lastProgress = 0;

	std::deque<double> progressQ;

	while(!(*progP).quit) {
		double progress = (double)(*((*progP).currentProgress)) / (double)(*progP).maxProgress;

		progressQ.push_front(progress - lastProgress);

		if (progressQ.size() > 10) progressQ.pop_back();

		double progressAvg = 0.0;

		for (std::deque<double>::iterator it = progressQ.begin(); it != progressQ.end(); *it++) progressAvg += *it;

		progressAvg /= (double) progressQ.size();

		double timeEst = 0.1*(1.0 - progress) / (progressAvg);
		lastProgress = progress;
		printf("Current progress is %3.3f%%. Estimated total runtime remaining is %8.3f seconds. Avg rate is %1.5f%%.\n", 100.0*progress, timeEst, progressAvg*100.0);

		Sleep(100);
	}
	return 0;
}

// Helper function for using CUDA
DWORD WINAPI cudaBbpLauncher(LPVOID dataV)//cudaError_t addWithCuda(sJ *output, unsigned int size, TYPE digit)
{
	PBBPLAUNCHERDATA data = (PBBPLAUNCHERDATA)dataV;
	int size = (*data).size;
	int gpu = (*data).gpu;
	int totalGpus = (*data).totalGpus;
	INT_64 digit = (*data).digit;
	sJ *dev_c = 0;
	sJ* c = new sJ[size];

	cudaError_t cudaStatus;

	PROGRESSDATA threadData;
	HANDLE thread;
	//these variables are linked between host and device memory allowing each to communicate about progress
	volatile INT_64 *currProgHost, *currProgDevice;

	if (gpu == 0) {

		//allow device to map host memory for progress ticker
		cudaStatus = cudaSetDeviceFlags(cudaDeviceMapHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDeviceFlags failed with error: %s\n", cudaGetErrorString(cudaStatus));
			goto Error;
		}

		// Allocate Host memory for progress ticker
		cudaStatus = cudaHostAlloc((void**)&currProgHost, sizeof(INT_64), cudaHostAllocMapped);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaHostAalloc failed!");
			goto Error;
		}

		//create link between between host and device memory for progress ticker
		cudaStatus = cudaHostGetDevicePointer((INT_64 **)&currProgDevice, (INT_64 *)currProgHost, 0);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaHostGetDevicePointer failed!");
			goto Error;
		}

		*currProgHost = 0;

		threadData = { currProgHost, digit, 0 };

		thread = CreateThread(NULL, 0, *progressCheck, (LPVOID)&threadData, 0, NULL);

		if (thread == NULL) {
			fprintf(stderr, "progressCheck thread creation failed\n");
			goto Error;
		}
	}

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(gpu);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffer for output vector    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(sJ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	INT_64 stride =  (INT_64) size * (INT_64) totalGpus;

	INT_64 launchWidth = stride * 128LLU;

	//need to round up
	//because bbp condition for stopping is <= digit, number of total elements in summation is 1 + digit
	//even when digit/launchWidth is an integer, it is necessary to add 1
	INT_64 neededLaunches = (digit / launchWidth) + 1LLU;

	for (INT_64 launch = 0; launch < neededLaunches; launch++) {

		INT_64 begin = launchWidth * launch;
		INT_64 end = launchWidth * (launch + 1) - 1;
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

		//give the rest of the computer some gpu time to reduce system choppiness
		Sleep(1);
	}

	if (gpu == 0) {

		//tell the progress thread to quit
		threadData.quit = 1;

		WaitForSingleObject(thread, INFINITE);
		CloseHandle(thread);
	}

	cudaStatus = reduceSJ(dev_c, size);

	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	// Copy result vector from GPU 0 buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(sJ), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	(*data).output = c[0];

Error:
	free(c);
	cudaFree(dev_c);

	(*data).error = cudaStatus;
	return 0;
}
