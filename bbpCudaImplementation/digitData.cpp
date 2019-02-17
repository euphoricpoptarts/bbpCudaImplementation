#include "digitData.h"
#include <algorithm>


digitData::digitData(const digitData& toCopy) {
	//no copy should need the old currentProgress and deviceProg
	//as no owner of a copy should need those members
	this->sumEnd = toCopy.sumEnd;
	this->startingExponent = toCopy.startingExponent;
	this->sumBegin = toCopy.sumBegin;
	this->segmentBegin = toCopy.segmentBegin;
}

digitData::digitData(uint64 sumEnd, uint64 startingExponent, uint64 segmentBegin, int imdifferent)
	: sumEnd(sumEnd), startingExponent(startingExponent), sumBegin(segmentBegin), segmentBegin(segmentBegin) {
	std::atomic_init(&this->launchCount, 0);
	setupProgress();
}

digitData::digitData(uint64 digitInput, uint64 segments, uint64 segmentNumber) {

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

digitData::~digitData() {
	//if(this->currentProgress != nullptr) delete this->currentProgress;
	//if(this->deviceProg != nullptr) cudaFree(this->deviceProg);
}

void digitData::setupProgress() {
	//these variables are linked between host and device memory allowing each to communicate about progress
	volatile uint64 *currProgHost;
	uint64 * currProgDevice;

	if (!hostDeviceMapped) {
		this->error = deviceMapHost();
		if (this->error != cudaSuccess) {
			fprintf(stderr, "cudaSetDeviceFlags failed with error: %s\n", cudaGetErrorString(this->error));
			return;
		}
		hostDeviceMapped = true;
	}

	// Allocate Host memory for progress ticker
	this->error = cudaHostAlloc((void**)&currProgHost, sizeof(uint64), cudaHostAllocMapped);
	if (this->error != cudaSuccess) {
		fprintf(stderr, "cudaHostAlloc failed!");
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

cudaError_t digitData::deviceMapHost() {
	//allow device to map host memory for progress ticker
	cudaError_t error = cudaSetDeviceFlags(cudaDeviceMapHost);
	return error;
}

bool digitData::hostDeviceMapped = false;