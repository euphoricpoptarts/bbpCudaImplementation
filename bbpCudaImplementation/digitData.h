#pragma once
#include "kernel.cuh"
#include <atomic>

class digitData
{
private:
	static bool hostDeviceMapped;
	void setupProgress();
	static cudaError_t deviceMapHost();

public:
	uint64 sumEnd = 0;
	uint64 startingExponent = 0;
	uint64 sumBegin = 0;
	uint64 segmentBegin = 0;
	std::atomic<uint64> launchCount;
	cudaError_t error;
	volatile uint64 * currentProgress = nullptr;
	uint64 * deviceProg = nullptr;

	digitData(const digitData& toCopy);

	digitData(uint64 sumEnd, uint64 startingExponent, uint64 segmentBegin, int imdifferent);

	digitData(uint64 digitInput, uint64 segments, uint64 segmentNumber);

	~digitData();

};

