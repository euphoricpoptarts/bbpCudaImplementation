
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

struct sJ {
	double s1, s4, s5, s6;
};

cudaError_t addWithCuda(sJ *c, unsigned int size, long digit);

__device__ const long baseSystem = 16;

__device__ void modExp(long long base, long exp, long long mod, long *output) {
	long long mask = 1;
	long long result = 1;
	while (exp > 0) {
		if (exp&mask) result *= base;
		base *= base;
		result %= mod;
		base %= mod;
		exp >>= 1;
	}
	*output = result;
}

__device__ void fractionalPartOfSum(long exp, long mod, double *partialSum) {
	long expModResult = 0;
	modExp(baseSystem, exp, mod, &expModResult);
	*partialSum += ((double)expModResult) / ((double)mod);
}

__device__ void bbp(long n, long start, long stride, sJ *output) {

	double s1 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
	double trash = 0.0;
	for (long k = start; k <= n; k += stride) {
		long mod = 8 * k + 1;
		fractionalPartOfSum(n - k, mod, &s1);
		mod += 3;
		fractionalPartOfSum(n - k, mod, &s4);
		mod += 1;
		fractionalPartOfSum(n - k, mod, &s5);
		mod += 1;
		fractionalPartOfSum(n - k, mod, &s6);
		//remove any integer part of s1-s6
		s1 = modf(s1, &trash);
		s4 = modf(s4, &trash);
		s5 = modf(s5, &trash);
		s6 = modf(s6, &trash);
	}
	output[start].s1 = s1;
	output[start].s4 = s4;
	output[start].s5 = s5;
	output[start].s6 = s6;
}

__global__ void bbpKernel(sJ *c, long digit)
{
	int i = threadIdx.x;
	bbp(digit, i, blockDim.x, c);
}

__global__ void reduceSJKernel(sJ *c, int stride) {
	int i = threadIdx.x;
	int augend = i + stride;
	c[i].s1 += c[augend].s1;
	c[i].s4 += c[augend].s4;
	c[i].s5 += c[augend].s5;
	c[i].s6 += c[augend].s6;
}

//standard tree-based parallel reduce
cudaError_t reduceSJ(sJ *c, unsigned int size) {
	cudaError_t cudaStatus;
	while (size > 1) {
		int nextSize = (size + 1) / 2;

		//size is odd
		if (size&1) reduceSJKernel<< <1, nextSize - 1 >> >(c, nextSize);
		//size is even
		else reduceSJKernel<< <1, nextSize >> >(c, nextSize);

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

long finalizeDigit(sJ input, long n) {
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
	const int arraySize = 256;
	const long digitPosition = 1000000;
	sJ c[arraySize];

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, arraySize, digitPosition);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	long hexDigit = finalizeDigit(c[0], digitPosition);

	printf("pi at hexadecimal digit %d is %X\n",
		digitPosition, hexDigit);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA
cudaError_t addWithCuda(sJ *c, unsigned int size, long digit)
{
	sJ *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
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

	// Launch a kernel on the GPU with one thread for each element.
	bbpKernel << <1, size >> >(dev_c, digit);

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

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(sJ), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	sJ expected;

	expected.s1 = 0;
	expected.s4 = 0;
	expected.s5 = 0;
	expected.s6 = 0;

	for (int j = 0; j < size; j++) {
		expected.s1 += c[j].s1;
		expected.s4 += c[j].s4;
		expected.s5 += c[j].s5;
		expected.s6 += c[j].s6;
	}

	cudaStatus = reduceSJ(dev_c, size);

	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(sJ), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

	if (fabs(c[0].s1 - expected.s1) > 1e-10) {
		printf("s1 not correct\n");
		printf("Expected %.8f; Actual %.8f\n", expected.s1, c[0].s1);
	}
	if (fabs(c[0].s4 - expected.s4) > 1e-10) {
		printf("s4 not correct\n");
		printf("Expected %.8f; Actual %.8f\n", expected.s4, c[0].s4);
	}
	if (fabs(c[0].s5 - expected.s5) > 1e-10) {
		printf("s5 not correct\n");
		printf("Expected %.8f; Actual %.8f\n", expected.s5, c[0].s5);
	}
	if (fabs(c[0].s6 - expected.s6) > 1e-10) {
		printf("s6 not correct\n");
		printf("Expected %.8f; Actual %.8f\n", expected.s6, c[0].s6);
	}

Error:
	cudaFree(dev_c);

	return cudaStatus;
}
