#pragma once
#include <mutex>
#include <sstream>
#include <deque>
#include "digitData.hpp"
#include "kernel.cuh"

extern uint64 strideMultiplier;
extern const int threadCountPerBlock;
extern int blockCount;
extern int primaryGpu;
extern const uint64 cachePeriod;
extern bool stop;

class bbpLauncher {
private:
	sJ output;
	int size = 0;
	cudaError_t error;
	digitData * data;
	std::deque<std::pair<sJ, uint64>> cacheQueue;
	std::mutex cacheMutex;
	int gpu;
	bool complete = false;
	std::string uuid;

	void cacheProgress(uint64 cacheEnd, sJ cacheData);

	static cudaError_t reduceSJ(sJ *c, unsigned int size);

	void queryUuid();

public:
	bbpLauncher(digitData * data, int gpu);

	bbpLauncher(int gpu, int size);

	bbpLauncher();

	std::string getUuid();

	void setData(digitData * data);

	void setSize(int size);

	cudaError_t getError();

	sJ getResult();

	std::pair<sJ, uint64> getCacheFront();

	bool hasCache();

	bool isComplete();

	void launch();
};