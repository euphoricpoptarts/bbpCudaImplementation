#pragma once
#include <mutex>
#include "digitData.hpp"
#include "kernel.cuh"

uint64 strideMultiplier;
//warpsize is 32 so optimal value is almost certainly a multiple of 32
const int threadCountPerBlock = 128;
//blockCount is trickier, and is probably a multiple of the number of streaming multiprocessors in a given gpu
int blockCount;
int primaryGpu;
const uint64 cachePeriod = 20000;
bool stop = false;

class bbpLauncher {
	sJ output;
	int size = 0;
	cudaError_t error;
	digitData * data;
	std::deque<std::pair<sJ, uint64>> cacheQueue;
	std::mutex cacheMutex;
	int gpu;
	bool complete = false;

	void cacheProgress(uint64 cacheEnd, sJ cacheData) {
		cacheMutex.lock();
		cacheQueue.emplace_back(cacheData, cacheEnd);
		cacheMutex.unlock();
	}

	//standard tree-based parallel reduce
	static cudaError_t reduceSJ(sJ *c, unsigned int size) {
		cudaError_t cudaStatus;
		while (size > 1) {
			int nextSize = (size + 1) >> 1;

			//size is odd
			if (size & 1) reduceSJPassThrough(32, 32, c, nextSize, nextSize - 1);
			//size is even
			else reduceSJPassThrough(32, 32, c, nextSize, nextSize);

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

public:
	bbpLauncher(digitData * data, int gpu) {
		this->data = data;
		this->gpu = gpu;
	}

	bbpLauncher() {}

	void setData(digitData * data) {
		this->data = data;
	}

	void setSize(int size) {
		this->size = size;
	}

	cudaError_t getError() {
		return this->error;
	}

	sJ getResult() {
		return this->output;
	}

	std::pair<sJ, uint64> getCacheFront() {
		cacheMutex.lock();
		std::pair<sJ, uint64> frontOfQ = cacheQueue.front();
		cacheQueue.pop_front();
		cacheMutex.unlock();
		return frontOfQ;
	}

	bool hasCache() {
		return cacheQueue.size() > 0;
	}

	bool isComplete() {
		return this->complete;
	}

	// Helper function for using CUDA
	void launch()
	{
		this->complete = false;
		sJ *dev_c = 0;
		sJ* c = new sJ[1];
		sJ *dev_ex = 0;

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
		cudaStatus = cudaMalloc((void**)&dev_ex, this->size * sizeof(sJ) * 7);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Allocate GPU buffer for output vector
		cudaStatus = cudaMalloc((void**)&dev_c, this->size * sizeof(sJ) * 7);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		launchWidth = (uint64)this->size * strideMultiplier;

		//need to round up
		//because bbp condition for stopping is <= digit, number of total elements in summation is 1 + digit
		//even when digit/launchWidth is an integer, it is necessary to add 1
		neededLaunches = ((this->data->sumEnd - this->data->sumBegin) / launchWidth) + 1LLU;
		while (!stop && ((currentLaunch = this->data->launchCount++) < neededLaunches)) {

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
				cudaStatus = cudaMemcpy(dev_ex, dev_c, size * sizeof(sJ) * 7, cudaMemcpyDeviceToDevice);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed in status update!\n");
					goto Error;
				}

				cudaStatus = reduceSJ(dev_ex, size * 7);

				if (cudaStatus != cudaSuccess) {
					goto Error;
				}

				// Copy result (reduced into first element) from GPU buffer to host memory.
				cudaStatus = cudaMemcpy(c, dev_ex, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
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
		cudaStatus = cudaMemcpy(c, dev_c, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
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
};