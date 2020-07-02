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
#include <atomic>
#include <vector>
#include <list>
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
	std::list<std::pair<std::thread, bbpLauncher *>> threadLauncherPairs;
	int reloadChoice;
	digitData * digit = nullptr;
	digitData * nextWorkUnit;
	bool workAssigned = false;
	bool workRequested = false;
	bool hasDelegator = false;
	restClientDelegator * delegator;
	std::condition_variable cv;
	std::atomic<uint64> checkToStop;

	void writeCache(uint64 cacheEnd, uint128 cacheData, double elapsedTime);
	bool fetchResultFromLaunchers(uint128& result, double& time);
	int reloadFromCache(std::string pToFile);
	void requestWork();
	void blockForWork();
	void sendResult(uint128 result, double time);
	bool areLaunchersComplete();
	bool checkForProgressCache();

public:
	uint128 previousCache;
	double previousTime;
	volatile int quit = 0;
	chr::steady_clock::time_point begin;

	progressData(digitData * data);

	progressData(restClientDelegator * delegator);

	void setReloadPolicy(int choice);

	void assignWork(digitData * data);

	void addLauncherToTrack(bbpLauncher * launcher);

	void runSingleWorkUnit();

	void beginWorking();

	void beginWorkUnit();

	//this function is meant to be run by an independent thread to output progress to the console
	void progressCheck();

	std::string controlledUuids();

	void setStopCheck(uint64 remoteId);
};