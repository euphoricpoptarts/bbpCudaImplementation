
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <math.h>
#include <time.h>
#include <Windows.h>

#define TYPE unsigned long long
#define INT_64 unsigned long long

struct sJ {
	double s1, s4, s5, s6;
};

typedef struct progressData {
	volatile INT_64 *currentProgress;
	TYPE maxProgress;
	int quit = 0;
} PROGRESSDATA, *PPROGRESSDATA;

cudaError_t addWithCuda(sJ *c, unsigned int size, TYPE digit);

//warpsize is 32 so optimal value is probably always a multiple of 32
const int threadCountPerBlock = 64;
//this is more difficult to optimize but seems to not like odd numbers
const int blockCount = 560;

__device__ const TYPE baseSystem = 16;
__device__ const int baseExpOf2 = 4;

__device__ const int typeSize = sizeof(TYPE) * 8 - 1;
__device__ const TYPE multiplyModCond = 0x2000000000000000;//2^61
__device__ const int int64Size = sizeof(INT_64) * 8 - 1;
__device__ const INT_64 int64ModCond = 0x40000000;
__device__ const INT_64 int64MaxBit = 0x8000000000000000;

__device__ int printOnce = 0;

//not actually quick
__device__ void quickMod(INT_64 input, const INT_64 mod, INT_64 *output) {

	/*INT_64 copy = input;
	INT_64 test = input % mod;*/
	INT_64 temp = mod;
	while (temp < input && !(temp&int64MaxBit)) temp <<= 1;
	if (temp > input) temp >>= 1;
	while (input >= mod && temp >= mod) {
		if(input >= temp) input -= temp;
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
	TYPE result = multiplicand;

	//TYPE highestBitMask = 0;

	//findMultiplierHighestBit(multiplier, &highestBitMask);

	/*unsigned long zeroCount = 0;

	zeroCount = __clzll(multiplier);*/

	TYPE highestBitMask = 0;

	//highestBitMask <<= (63 - zeroCount);
	
	findMultiplierHighestBitHackersDelight(multiplier, &highestBitMask);

	while (highestBitMask > 1) {
		//only perform modulus operation during loop if result is >= (TYPE maximum + 1)/8 (in order to prevent overflowing)
		if (result >= multiplyModCond) result %= mod;
		result <<= 1;
		highestBitMask >>= 1;
		if (multiplier&highestBitMask)	result += multiplicand;
	}

	//modulus must be taken after loop as it hasn't necessarily been taken during last loop iteration
	result %= mod;
	*output = result;
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
__device__ void modExpLeftToRight(const TYPE exp, TYPE mod, TYPE highestBitMask, TYPE *output) {
	INT_64 result = baseSystem;

	//only perform modulus operation during loop if result is >= sqrt((BIG_TYPE maximum + 1)/8) (in order to prevent overflowing)
	INT_64 modCond = int64ModCond;

	while (highestBitMask > 1) {
		//this is not necessary as modMultiplyLeftToRight ensures result never overflows a 64 bit buffer
		//however performing this modulus saves time (more is less)
		//likely saves performing some moduli in modMultiplyLeftToRight or reduces overall size of arguments
		if (result >= mod) result %= mod;//quickMod(result, mod, &result);

		modMultiplyLeftToRight(result, result, mod, &result);//result *= result;
		highestBitMask >>= 1;
		if (exp&highestBitMask)	result <<= baseExpOf2;//modMultiplyLeftToRight(result, base, mod, &result);//result *= base;
	}

	//modulus must be taken after loop as it hasn't necessarily been taken during last loop iteration
	//result %= mod;//quickMod(result, mod, &result);
	*output = result;
}

//find ( 16^n % mod ) / mod and add to partialSum
__device__ void fractionalPartOfSum(TYPE exp, TYPE mod, double *partialSum, TYPE highestBitMask) {
	TYPE expModResult = 0;
	modExpLeftToRight(exp, mod, highestBitMask, &expModResult);
	*partialSum += ((double)expModResult) / ((double)mod);
}

//stride over all parts of summation in bbp formula where k <= n
//to compute partial sJ sums
__device__ void bbp(TYPE n, TYPE start, TYPE stride, sJ *output, volatile INT_64 *progress) {

	double s1 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
	double trash = 0.0;
	TYPE highestExpBit = 1;
	while (highestExpBit <= n)	highestExpBit <<= 1;
	for (TYPE k = start; k <= n; k += stride) {
		while (highestExpBit > (n - k))  highestExpBit >>= 1;
		TYPE mod = 8 * k + 1;
		fractionalPartOfSum(n - k, mod, &s1, highestExpBit);
		mod += 3;
		fractionalPartOfSum(n - k, mod, &s4, highestExpBit);
		mod += 1;
		fractionalPartOfSum(n - k, mod, &s5, highestExpBit);
		mod += 1;
		fractionalPartOfSum(n - k, mod, &s6, highestExpBit);
		//remove any integer part of s1-s6
		s1 = modf(s1, &trash);
		s4 = modf(s4, &trash);
		s5 = modf(s5, &trash);
		s6 = modf(s6, &trash);
		if (start == 0) {
			//only 1 thread ever updates the progress
			*progress = k;
		}
	}
	output[start].s1 = s1;
	output[start].s4 = s4;
	output[start].s5 = s5;
	output[start].s6 = s6;
}

//determine from thread and block position where to begin stride
//and how wide stride is
__global__ void bbpKernel(sJ *c, volatile INT_64 *progress, TYPE digit)
{
	TYPE stride = blockDim.x * gridDim.x;
	TYPE i = threadIdx.x + blockDim.x * blockIdx.x;
	bbp(digit, i, stride, c, progress);
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

int main()
{
	try {
		const int arraySize = threadCountPerBlock * blockCount;
		const TYPE digitPosition = 9999999999;
		sJ* c = new sJ[arraySize];

		clock_t start = clock();

		cudaError_t cudaStatus = addWithCuda(c, arraySize, digitPosition);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "addWithCuda failed!");
			return 1;
		}

		long hexDigit = finalizeDigit(c[0], digitPosition);

		clock_t end = clock();

		printf("pi at hexadecimal digit %llu is %X\n",
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

	while(!(*progP).quit) {
		double progress = (double)(*((*progP).currentProgress)) / (double)(*progP).maxProgress;

		if (progress - lastProgress > 1E-4) {
			double timeEst = (1.0 - progress)*0.1 / (progress - lastProgress);
			lastProgress = progress;
			printf("Current progress is %3.3f%%. Estimated total runtime remaining is %8.3f seconds.\n", 100.0*progress, timeEst);
		}

		Sleep(100);
	}
	return 0;
}

// Helper function for using CUDA
cudaError_t addWithCuda(sJ *c, unsigned int size, TYPE digit)
{
	sJ *dev_c = 0;

	//these variables are linked between host and device memory allowing each to communicate about progress
	volatile INT_64 *currProgHost, *currProgDevice;

	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	//allow device to map host memory for progress ticker
	cudaSetDeviceFlags(cudaDeviceMapHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDeviceFlags failed!");
		goto Error;
	}

	// Allocate GPU buffer for output vector    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(sJ));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
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

	PROGRESSDATA threadData = { currProgHost, digit, 0 };

	HANDLE thread = CreateThread(NULL, 0, *progressCheck, (LPVOID) &threadData, 0, NULL);

	if (thread == NULL) {
		fprintf(stderr, "progressCheck thread creation failed\n");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	bbpKernel << <blockCount, threadCountPerBlock >> >(dev_c, currProgDevice, digit);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "bbpKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bbpKernel!\n", cudaStatus);
		goto Error;
	}

	//tell the progress thread to quit
	threadData.quit = 1;

	WaitForSingleObject(thread, INFINITE);
	CloseHandle(thread);

	cudaStatus = reduceSJ(dev_c, size);

	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	// Copy result vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(sJ), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);

	return cudaStatus;
}
