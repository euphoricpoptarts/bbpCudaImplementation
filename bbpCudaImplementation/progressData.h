#pragma once
#include <chrono>
#include <thread>
#include <deque>
#include <iostream>
#include <iomanip>
#include <fstream>
#ifdef __linux__
#include <experimental/filesystem>
#elif _WIN64
#include <filesystem>
#endif
#include <algorithm>
#include "kernel.cuh"
#include "bbpLauncher.h"
#include "digitData.h"

namespace chr = std::chrono;

class restClientDelegator;

//this class observes a given digitData object,
//and also a vector of bbpLaunchers
//it can make a cache file for a computation, and also reload such cache files
class progressData {
private:
	static const std::string progressFilenamePrefixTemplate;
	std::string progressFilenamePrefix;
	std::vector<bbpLauncher*> launchersTracked;
	int reloadChoice;
	digitData * digit = nullptr;
	digitData * nextWorkUnit;
	bool workAssigned = false;
	bool workRequested = false;
	bool hasDelegator = false;
	restClientDelegator * delegator;
	std::condition_variable cv;

	void writeCache(uint64 cacheEnd, uint128 cacheData, double elapsedTime);

	int reloadFromCache(std::string pToFile);

	void requestWork();

	void blockForWork();

	void sendResult(uint128 result, double time);

public:
	uint128 previousCache;
	double previousTime;
	volatile int quit = 0;
	chr::high_resolution_clock::time_point begin;

	progressData(digitData * data);

	progressData(restClientDelegator * delegator);

	void setReloadPolicy(int choice);

	void assignWork(digitData * data);

	void addLauncherToTrack(bbpLauncher * launcher);

	int checkForProgressCache(uint64 totalSegments, uint64 segment);

	void beginWorking();

	//this function is meant to be run by an independent thread to output progress to the console
	void progressCheck();
};