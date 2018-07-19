
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>

cudaError_t addWithCuda(long *c, unsigned int size);

__device__ void modExp(long base, long exp, long mod, long *output) {
	long mask = 1;
	long result = 1;
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
	modExp(16, exp, mod, &expModResult);
	*partialSum += ((double)expModResult) / ((double)mod);
}

__device__ void bbp(long n, long *output) {

	double s1 = 0.0, s4 = 0.0, s5 = 0.0, s6 = 0.0;
	double trash = 0.0;
	for (long k = 0; k <= n; k++) {
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
	double reducer = 1.0;
	for (int i = 0; i < 4; i++) {
		n++;
		reducer /= 16.0;
		double eightN = 8.0 * n;
		s1 += reducer / (eightN + 1.0);
		s4 += reducer / (eightN + 4.0);
		s5 += reducer / (eightN + 5.0);
		s6 += reducer / (eightN + 6.0);
	}
	//remove any integer part of s1-s6
	s1 = modf(s1, &trash);
	s4 = modf(s4, &trash);
	s5 = modf(s5, &trash);
	s6 = modf(s6, &trash);
	double hexDigit = 4.0*s1 - 2.0*s4 - s5 - s6;
	hexDigit = modf(hexDigit, &trash);
	if (hexDigit < 0) hexDigit++;
	hexDigit *= 16;
	*output = (long)hexDigit;
}

__global__ void addKernel(long *c)
{
	int i = threadIdx.x;
	bbp(i, &(c[i]));
}

int main()
{
	const int arraySize = 10;
	long c[arraySize] = { 0 };

	// Add vectors in parallel.
	cudaError_t cudaStatus = addWithCuda(c, arraySize);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addWithCuda failed!");
		return 1;
	}

	printf("pi = 3.%X%X%X%X%X%X%X%X%X%X\n",
		c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7], c[8], c[9]);

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		return 1;
	}

	return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(long *c, unsigned int size)
{
	long *dev_c = 0;
	cudaError_t cudaStatus;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffers for three vectors (two input, one output)    .
	cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(long));
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Launch a kernel on the GPU with one thread for each element.
	addKernel << <1, size >> >(dev_c);

	// Check for any errors launching the kernel
	cudaStatus = cudaGetLastError();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
		goto Error;
	}

	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
		goto Error;
	}

	// Copy output vector from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!");
		goto Error;
	}

Error:
	cudaFree(dev_c);

	return cudaStatus;
}
