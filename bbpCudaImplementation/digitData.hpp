#pragma once
#include <atomic>

//this class contains all needed data to define the work for a given segment of a digit
//and to synchronize that work between multiple GPUs
class digitData {
public:
	uint64 sumEnd = 0;
	uint64 startingExponent = 0;
	uint64 sumBegin = 0;
	uint64 segmentBegin = 0;
	std::atomic<uint64> launchCount;
	cudaError_t error;
	volatile uint64 * currentProgress;
	uint64 * deviceProg;

	digitData(uint64 digitInput, uint64 segments, uint64 segmentNumber) {

		//this currently assumes that all launches will be of the same width (which is currently the case)
		//if in the future I want to change that, this will instead track the next sum term to process and will be initiated to sumBegin
		std::atomic_init(&this->launchCount, 0);

		//subtract 1 to convert to 0-indexed
		digitInput--;

		//4*hexDigitPosition converts from exponent of 16 to exponent of 2
		//adding 128 for fixed-point division algorithm
		//adding 8 for maximum size of coefficient (so that all coefficients can be expressed by subtracting an integer from the exponent)
		//subtracting 6 for the division by 64 of the whole sum
		this->startingExponent = (4LLU * digitInput) + 128LLU + 8LLU - 6LLU;

		//represents the ending index of the entire summation
		//does not include addition by 8 used above so that all nominators of summation will be a positive exponent of 2 after factoring in the coefficient
		//this leaves out a couple potentially positive exponents of 2 (could potentially just check subtraction in modExpLeftToRight and keep the addition by 8)
		uint64 endIndexOfSum = (4LLU * digitInput - 6LLU + 128LLU) / 10LLU;

		//as the range of the summation is [0, sumEnd], the count of total terms is sumEnd + 1
		//need to find ceil((endIndexOfSum + 1) / segments), this is equivalent to that but requires no floating point data types
		uint64 segmentWidth = (endIndexOfSum / segments) + 1;

		//term to begin summation from
		this->sumBegin = segmentWidth * segmentNumber;
		//this is used to compute progress of a segment
		this->segmentBegin = this->sumBegin;

		//subtract 1 as the beginning of next segment should not be included
		this->sumEnd = segmentWidth * (segmentNumber + 1) - 1;

		this->sumEnd = std::min(this->sumEnd, endIndexOfSum);

		setupProgress();
	}

	void setupProgress() {
		//these variables are linked between host and device memory allowing each to communicate about progress
		volatile uint64 *currProgHost;
		uint64 * currProgDevice;

		//allow device to map host memory for progress ticker
		this->error = cudaSetDeviceFlags(cudaDeviceMapHost);
		if (this->error != cudaSuccess) {
			fprintf(stderr, "cudaSetDeviceFlags failed with error: %s\n", cudaGetErrorString(this->error));
			return;
		}

		// Allocate Host memory for progress ticker
		this->error = cudaHostAlloc((void**)&currProgHost, sizeof(uint64), cudaHostAllocMapped);
		if (this->error != cudaSuccess) {
			fprintf(stderr, "cudaHostAalloc failed!");
			return;
		}

		//create link between between host and device memory for progress ticker
		this->error = cudaHostGetDevicePointer((uint64 **)&currProgDevice, (uint64 *)currProgHost, 0);
		if (this->error != cudaSuccess) {
			fprintf(stderr, "cudaHostGetDevicePointer failed!");
			return;
		}

		*currProgHost = 0;

		this->deviceProg = currProgDevice;
		this->currentProgress = currProgHost;
	}
};