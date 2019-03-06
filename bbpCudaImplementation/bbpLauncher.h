#pragma once
#include <mutex>
#include <sstream>
#include <deque>
#include "digitData.h"
#include "kernel.cuh"

extern uint64 strideMultiplier;
extern const int threadCountPerBlock;
extern int blockCount;
extern int primaryGpu;
extern const uint64 cachePeriod;
extern volatile bool globalStopSignal;

class bbpLauncher {
private:
	uint128 output;
	int size = 0;
	cudaError_t error;
	digitData * data;
	std::deque<std::pair<uint128, uint64>> cacheQueue;
	std::mutex cacheMutex;
	int gpu;
	bool complete = false;
	volatile bool quitSignal = false;
	std::string uuid;

	void cacheProgress(uint64 cacheEnd, uint128 cacheData);

	static cudaError_t reduceSJ(uint128 *c, unsigned int size);

	void queryUuid();

public:
	bbpLauncher(digitData * data, int gpu);

	bbpLauncher(int gpu, int size);

	bbpLauncher();

	std::string getUuid();

	void setData(digitData * data);

	void setSize(int size);

	cudaError_t getError();

	uint128 getResult();

	std::pair<uint128, uint64> getCacheFront();

	bool hasCache();

	bool isComplete();

	void launch();

	void quit();
};