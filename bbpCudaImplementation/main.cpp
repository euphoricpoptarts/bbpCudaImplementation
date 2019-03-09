#pragma once
#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#include <string>
#include <csignal>
#include "kernel.cuh"
#include "progressData.h"
#include "bbpLauncher.h"
#include "digitData.h"
#include "restClientDelegator.h"

namespace chr = std::chrono;
uint64 segments = 1;
int startBlocks, blocksIncrement, incrementLimit;
uint64 benchmarkTarget;
int controlType;
int numRuns;

const struct {
	const std::string STRIDEMULTIPLIER = "strideMultiplier",
		BLOCKCOUNT = "blockCount",
		PRIMARYGPU = "primaryGpu",
		CONTROLTYPE = "controlType",
		BENCHMARKTRIALS = "benchmarkTrials",
		BENCHMARKTARGET = "benchmarkTarget",
		BENCHMARKSTARTINGBLOCKCOUNT = "benchmarkStartingBlockCount",
		BENCHMARKBLOCKCOUNTINCREMENT = "benchmarkBlockCountIncrement",
		BENCHMARKTOTALINCREMENTS = "benchmarkTotalIncrements",
		APIKEY = "apiKey",
		DOMAINNAME = "domain",
		PORT = "port";
} propertyNames;

std::string propertiesFile = "application.properties";
void sigint_handler(int sig);

class commandLineArgsLoader {

	bool digitGiven;
	uint64 givenDigit;
	uint64 segments;
	uint64 segmentNumber;
	int reloadChoice;

public:
	commandLineArgsLoader(int argc, char** argv) {
		this->givenDigit = 0;
		this->digitGiven = false;
		if(argc >= 2) {
			this->givenDigit = std::stoull(argv[1]);
			this->digitGiven = true;
		}
		this->segments = 0;
		if (argc >= 3) {
			this->segments = std::stoull(argv[2]);
		}
		this->segmentNumber = 0;
		if(argc >= 4 && segments > 1) {
			this->segmentNumber = std::stoull(argv[3]);
		}
		reloadChoice = 0;
		if(argc >= 5) {
			std::string reloadChoiceString = argv[4];
			if (reloadChoiceString.compare("y") == 0) {
				this->reloadChoice = 1;
			}
			else if (reloadChoiceString.compare("n") == 0) {
				this->reloadChoice = 2;
			}
		}
	}

	uint64 getDigit() {
		uint64 digit = this->givenDigit;
		if(!digitGiven) {
			std::cout << "Input hexDigit to calculate (1-indexed):" << std::endl;
			std::cin >> digit;
		}
		return digit;
	}

	uint64 getTotalSegments() {
		uint64 totalSegments = this->segments;
		if (totalSegments == 0) {
			std::cout << "Input number of segments to split computation:" << std::endl;
			std::cin >> totalSegments;
			if (totalSegments == 0) totalSegments = 1;
		}
		return totalSegments;
	}

	uint64 getSegmentNumber(uint64 maxSegment) {
		uint64 number = this->segmentNumber;
		if (number == 0) {
			if (maxSegment > 1) {
				std::cout << "Input segment number to calculate (1 - " << maxSegment << "):" << std::endl;
				std::cin >> number;
			}
			if (number == 0) number = 1;
		}
		return number - 1;
	}

	int getReloadChoice() {
		return this->reloadChoice;
	}
};

int loadProperties() {
	std::cout << "Loading properties from " << propertiesFile << std::endl;
	std::ifstream propF(propertiesFile, std::ios::in);

	if (!propF.is_open()) {
		std::cerr << "Could not open " << propertiesFile << "!" << std::endl;
		return 1;
	}

	std::map<std::string, std::string> properties;

	std::string propertyName, propertyValue;
	while (propF >> propertyName >> propertyValue) {
		properties.emplace(propertyName, propertyValue);
	}

	propF.close();

	std::string checkProps[12] = {
		propertyNames.STRIDEMULTIPLIER,
		propertyNames.BLOCKCOUNT,
		propertyNames.PRIMARYGPU,
		propertyNames.CONTROLTYPE,
		propertyNames.BENCHMARKTRIALS,
		propertyNames.BENCHMARKTARGET,
		propertyNames.BENCHMARKSTARTINGBLOCKCOUNT,
		propertyNames.BENCHMARKBLOCKCOUNTINCREMENT,
		propertyNames.BENCHMARKTOTALINCREMENTS,
		propertyNames.APIKEY,
		propertyNames.DOMAINNAME,
		propertyNames.PORT
	};

	int missedProps = 0;
	for (auto prop : checkProps) {
		if (properties.find(prop) == properties.end()) {
			std::cerr << "Property " << prop << " was not found!" << std::endl;
			missedProps++;
		}
	}
	if (missedProps) return 1;

	strideMultiplier = std::stoull(properties.at(propertyNames.STRIDEMULTIPLIER));
	blockCount = std::stoi(properties.at(propertyNames.BLOCKCOUNT));
	primaryGpu = std::stoi(properties.at(propertyNames.PRIMARYGPU));
	controlType = std::stoi(properties.at(propertyNames.CONTROLTYPE));
	numRuns = std::stoi(properties.at(propertyNames.BENCHMARKTRIALS));
	benchmarkTarget = std::stoull(properties.at(propertyNames.BENCHMARKTARGET));
	startBlocks = std::stoi(properties.at(propertyNames.BENCHMARKSTARTINGBLOCKCOUNT));
	blocksIncrement = std::stoi(properties.at(propertyNames.BENCHMARKBLOCKCOUNTINCREMENT));
	incrementLimit = std::stoi(properties.at(propertyNames.BENCHMARKTOTALINCREMENTS));
	apiKey = properties.at(propertyNames.APIKEY);
	domain = properties.at(propertyNames.DOMAINNAME);
	targetPort = properties.at(propertyNames.PORT);
	if (segments == 0) segments = 1;

	return 0;
}

int benchmark() {
	digitData data(benchmarkTarget, 1, 1);
	if (data.error != cudaSuccess) return 1;
	bbpLauncher gpuData(&data, 0);
	std::vector<std::pair<double, int>> timings;
	for (blockCount = startBlocks; blockCount <= (startBlocks + incrementLimit * blocksIncrement); blockCount += blocksIncrement) {
		double total = 0.0;
		for (int j = 0; j < numRuns; j++) {
			data.launchCount = 0;
			chr::high_resolution_clock::time_point start = chr::high_resolution_clock::now();
			gpuData.setSize(threadCountPerBlock * blockCount);
			gpuData.launch();
			chr::high_resolution_clock::time_point end = chr::high_resolution_clock::now();
			total += chr::duration_cast<chr::duration<double>>(end - start).count();
		}
		double avg = total / (double)numRuns;
		std::cout << "Average for " << blockCount << " blocks is " << avg << " seconds." << std::endl;
		std::pair<double, int> timingPair(avg, blockCount);
		timings.push_back(timingPair);
	}
	std::sort(timings.begin(), timings.end());
	std::cout << "Fastest block counts:" << std::endl;
	for (int i = 0; i < 10; i++) {
		std::cout << timings.at(i).second << " blocks at " << timings.at(i).first << " seconds." << std::endl;
	}
	return 0;
}

int controlViaClient(int totalGpus) {
	restClientDelegator delegator;
	std::vector<bbpLauncher*> launchers;
	for (int i = 0; i < totalGpus; i++) {
		launchers.push_back(new bbpLauncher(i, threadCountPerBlock * blockCount));
	}
	std::list<progressData> controllers;
	for (bbpLauncher* launcher : launchers) {
		controllers.emplace_back(&delegator);
		controllers.back().addLauncherToTrack(launcher);
	}
	std::thread delegatorThread(&restClientDelegator::monitorQueues, &delegator);

	std::list<std::thread> workers;
	for(progressData& controller : controllers){
		workers.emplace_back(&progressData::beginWorking, &controller);
	}

	for (std::thread& worker : workers) {
		worker.join();
	}
	delegatorThread.join();

	for (bbpLauncher* launcher : launchers) delete launcher;
	launchers.clear();

	return 0;
}

int main(int argc, char** argv) {

	if (loadProperties()) return 1;

	signal(SIGINT, sigint_handler);

	int totalGpus = 0;

	cudaError_t cudaStatus = cudaGetDeviceCount(&totalGpus);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount failed!\n");
		return 1;
	}
	if (!totalGpus) {
		fprintf(stderr, "No GPUs detected in system!\n");
		return 1;
	}
	if (controlType == 2) {
		return benchmark();
	}
	else if (controlType == 1) {
		return controlViaClient(totalGpus);
	}

	commandLineArgsLoader args(argc, argv);

	const int arraySize = threadCountPerBlock * blockCount;
	uint64 hexDigitPosition = args.getDigit();

	segments = args.getTotalSegments();
	
	uint64 segmentNumber = args.getSegmentNumber(segments);

	digitData data(hexDigitPosition, segments, segmentNumber);
	if (data.error != cudaSuccess) return 1;

	std::thread * handles = new std::thread[totalGpus];
	std::vector<bbpLauncher*> gpuData;
	
	progressData prog(&data);
	prog.setReloadPolicy(args.getReloadChoice());

	if (prog.checkForProgressCache(segments, segmentNumber + 1LLU)) return 1;

	chr::high_resolution_clock::time_point start = chr::high_resolution_clock::now();
	prog.begin = start;

	for (int i = 0; i < totalGpus; i++) {
		gpuData.push_back(new bbpLauncher(&data, i));
		gpuData[i]->setSize(arraySize);
		prog.addLauncherToTrack(gpuData[i]);
	}

	std::cout << prog.controlledUuids() << std::endl;

	std::thread progThread(&progressData::progressCheck, &prog);

	for (int i = 0; i < totalGpus; i++) {
		handles[i] = std::thread(&bbpLauncher::launch, gpuData[i]);
	}

	uint128 cudaResult = prog.previousCache;

	for (int i = 0; i < totalGpus; i++) {

		handles[i].join();

		cudaStatus = gpuData[i]->getError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaBbpLaunch failed on gpu%d!\n", i);
			globalStopSignal = true;
		}

		uint128 output = gpuData[i]->getResult();

		//sum results from gpus
		sJAdd(&cudaResult, &output);
	}

	//tell the progress thread to quit
	//prog.quit = 1;

	progThread.join();

	delete[] handles;
	for (bbpLauncher* launcher : gpuData) delete launcher;
	gpuData.clear();

	if (!globalStopSignal) {

		chr::high_resolution_clock::time_point end = chr::high_resolution_clock::now();

		printf("pi at hexadecimal digit %llu is %016llX %016llX\n",
			hexDigitPosition, cudaResult.msw, cudaResult.lsw);

		//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
		double totalTime = prog.previousTime + (chr::duration_cast<chr::duration<double>>(end - start)).count();
		printf("Computed in %.8f seconds\n", totalTime);

		const char * completionPathFormat = "completed/segmented%dExponent%lluSegment%dBase2Complete.dat";
		char buffer[256];
		snprintf(buffer, sizeof(buffer), completionPathFormat, segments, data.startingExponent, segmentNumber + 1);
		std::ofstream completedF(buffer, std::ios::out);
		if (completedF.is_open()) {
			completedF << std::hex << std::setfill('0');
			completedF << std::setw(16) << cudaResult.lsw << std::endl;
			completedF << std::setw(16) << cudaResult.msw << std::endl;
			completedF << std::hexfloat << std::setprecision(13) << totalTime << std::endl;
			completedF.close();
		}
		else {
			fprintf(stderr, "Error opening file %s\n", buffer);
		}
	}
	else {
		std::cout << "Quitting upon user exit command!" << std::endl;
	}

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	return 0;
}

void sigint_handler(int sig) {
	if (sig == SIGINT) {
		globalStopSignal = true;
	}
}