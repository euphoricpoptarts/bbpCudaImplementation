#include "bbpLauncher.h"

#include <mutex>
#include <sstream>
#include <iomanip>
#include "digitData.h"
#include "kernel.cuh"

uint64 strideMultiplier = 0;
//warpsize is 32 so optimal value is almost certainly a multiple of 32
const int threadCountPerBlock = 128;
//blockCount is trickier, and is probably a multiple of the number of streaming multiprocessors in a given gpu
int blockCount = 0;
int primaryGpu = 0;
const uint64 cachePeriod = 20000;
volatile bool globalStopSignal = false;

void bbpLauncher::cacheProgress(uint64 cacheEnd, uint128 cacheData) {
	cacheMutex.lock();
	cacheQueue.emplace_back(cacheData, cacheEnd);
	cacheMutex.unlock();
}

//standard tree-based parallel reduce
cudaError_t bbpLauncher::reduceSJ(uint128 *c, unsigned int size) {
	cudaError_t cudaStatus;
	while (size > 1) {
		int nextSize = (size + 1) >> 1;

		//size is odd
		if (size & 1) reduceUint128ArrayPassThrough(32, 32, c, nextSize, nextSize - 1);
		//size is even
		else reduceUint128ArrayPassThrough(32, 32, c, nextSize, nextSize);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduceUint128ArrayKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceUint128ArrayKernel!\n", cudaStatus);
			return cudaStatus;
		}

		size = nextSize;
	}
	return cudaStatus;
}

void bbpLauncher::queryUuid() {
	cudaDeviceProp devProps;
	cudaGetDeviceProperties(&devProps, this->gpu);
	std::stringstream ss;
	ss << std::hex << std::uppercase << std::setfill('0');
	for (int i = 0; i < 16; i++) {
		unsigned int byte = (unsigned char)devProps.uuid.bytes[i];
		ss << std::setw(2) << byte;
	}
	this->uuid = ss.str();
}

bbpLauncher::bbpLauncher(digitData * data, int gpu) {
	this->data = data;
	this->gpu = gpu;
	queryUuid();
}

bbpLauncher::bbpLauncher(int gpu, int size) {
	this->gpu = gpu;
	this->size = size;
	queryUuid();
}

bbpLauncher::bbpLauncher() {}

std::string bbpLauncher::getUuid() {
	return this->uuid;
}

void bbpLauncher::setData(digitData * data) {
	this->data = data;
}

void bbpLauncher::setSize(int size) {
	this->size = size;
}

cudaError_t bbpLauncher::getError() {
	return this->error;
}

uint128 bbpLauncher::getResult() {
	return this->output;
}

std::pair<uint128, uint64> bbpLauncher::getCacheFront() {
	cacheMutex.lock();
	std::pair<uint128, uint64> frontOfQ = cacheQueue.front();
	cacheQueue.pop_front();
	cacheMutex.unlock();
	return frontOfQ;
}

bool bbpLauncher::hasCache() {
	return cacheQueue.size() > 0;
}

bool bbpLauncher::isComplete() {
	return this->complete;
}

void bbpLauncher::quit() {
	this->quitSignal = true;
}

// Helper function for using CUDA
void bbpLauncher::launch()
{
	this->complete = false;
	this->quitSignal = false;
	uint128 *dev_c = 0;
	uint128* c = new uint128[1];
	uint128 *dev_ex = 0;

	cudaError_t cudaStatus;

	uint64 launchWidth, neededLaunches, currentLaunch;

	uint64 lastWrite = 0;

	// Choose which GPU to run on, change this on a multi-GPU system.
	cudaStatus = cudaSetDevice(gpu);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		goto Error;
	}

	// Allocate GPU buffer for temp vector
	cudaStatus = cudaMalloc((void**)&dev_ex, this->size * sizeof(uint128) * 7);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	// Allocate GPU buffer for output vector
	cudaStatus = cudaMalloc((void**)&dev_c, this->size * sizeof(uint128) * 7);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMalloc failed!");
		goto Error;
	}

	launchWidth = (uint64)this->size * strideMultiplier;

	//need to round up
	//because bbp condition for stopping is <= digit, number of total elements in summation is 1 + digit
	//even when digit/launchWidth is an integer, it is necessary to add 1
	neededLaunches = ((this->data->sumEnd - this->data->sumBegin) / launchWidth) + 1LLU;
	while (!globalStopSignal && !this->quitSignal && ((currentLaunch = this->data->launchCount++) < neededLaunches)) {

		uint64 begin = this->data->sumBegin + (launchWidth * currentLaunch);
		uint64 end = this->data->sumBegin + (launchWidth * (currentLaunch + 1)) - 1;
		if (end > this->data->sumEnd) end = this->data->sumEnd;

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bbpKernel on gpu %d!\n", cudaStatus, gpu);
			goto Error;
		}
		//after exactly cachePeriod number of launches since last period between all gpus, write all data computed during and before the period to status buffer for progress thread to save
		if ((currentLaunch - lastWrite) >= cachePeriod) {

			lastWrite += cachePeriod;

			//copy current results into temp array to reduce and update status
			cudaStatus = cudaMemcpy(dev_ex, dev_c, size * sizeof(uint128) * 7, cudaMemcpyDeviceToDevice);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed in status update!\n");
				goto Error;
			}

			cudaStatus = reduceSJ(dev_ex, size * 7);

			if (cudaStatus != cudaSuccess) {
				goto Error;
			}

			// Copy result (reduced into first element) from GPU buffer to host memory.
			cudaStatus = cudaMemcpy(c, dev_ex, 1 * sizeof(uint128), cudaMemcpyDeviceToHost);
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaMemcpy failed in status update!\n");
				goto Error;
			}

			cacheProgress(this->data->sumBegin + (launchWidth * lastWrite), c[0]);
		}

		// calls the bbpKernel to compute a portion of the total bbp sum on the GPU
		bbpPassThrough(threadCountPerBlock, blockCount * 7, dev_c, this->data->deviceProg, this->data->startingExponent, begin, end, strideMultiplier);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "bbpKernel launch failed on gpu%d: %s\n", gpu, cudaGetErrorString(cudaStatus));
			goto Error;
		}

		////give the rest of the computer some gpu time to reduce system choppiness
		//if (primaryGpu) {
		//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
		//}
	}

	cudaStatus = reduceSJ(dev_c, size * 7);

	if (cudaStatus != cudaSuccess) {
		goto Error;
	}

	// Copy result (reduced into first element) from GPU buffer to host memory.
	cudaStatus = cudaMemcpy(c, dev_c, 1 * sizeof(uint128), cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy failed!\n");
		goto Error;
	}

	this->output = c[0];

Error:
	free(c);
	cudaFree(dev_c);
	cudaFree(dev_ex);

	this->error = cudaStatus;
	this->complete = true;
}