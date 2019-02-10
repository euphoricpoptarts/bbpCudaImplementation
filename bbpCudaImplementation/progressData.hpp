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
#include "bbpLauncher.hpp"
#include "digitData.hpp"

namespace chr = std::chrono;

class inertialDouble {
private:
	double value = 0.0;

public:
	void tug(double tugPoint) {
		if (value != tugPoint) {
			double delta = abs(value - tugPoint);
			double larger = std::max(abs(value), abs(tugPoint));
			double deltaScaler = delta / larger;
			value += ((tugPoint - value)*deltaScaler);
		}
	}

	double getValue() {
		return value;
	}
};

//this class observes a given digitData object,
//and also a vector of bbpLaunchers
//it can make a cache file for a computation, and also reload such cache files
class progressData {
private:
	static const std::string progressFilenamePrefixTemplate;
public:
	sJ previousCache;
	double previousTime;
	int reloadChoice;
	digitData * digit;
	volatile int quit = 0;
	chr::high_resolution_clock::time_point * begin;
	std::string progressFilenamePrefix;
	std::vector<bbpLauncher*> launchersTracked;

	progressData(digitData * data)
	{
		this->digit = data;
		this->quit = 0;
		this->reloadChoice = 0;
	}

	void setReloadPolicy(int choice) {
		this->reloadChoice = choice;
	}

	void addLauncherToTrack(bbpLauncher * launcher) {
		launchersTracked.push_back(launcher);
	}

	~progressData() {
		//TODO: delete the device/host pointers?
	}

	int checkForProgressCache(uint64 totalSegments, uint64 segment) {
		char buffer[256];
		snprintf(buffer, sizeof(buffer), progressFilenamePrefixTemplate.c_str(), this->digit->startingExponent, totalSegments, segment);
		this->progressFilenamePrefix = buffer;
		std::string pToFile;
		std::vector<std::string> matching;
		int found = 0;
		for (auto& element : std::experimental::filesystem::directory_iterator("progressCache")) {
			std::string name = element.path().filename().string();
			//filename begins with desired string
			if (name.compare(0, this->progressFilenamePrefix.length(), this->progressFilenamePrefix) == 0) {
				matching.push_back(element.path().string());
				found = 1;
			}
		}
		if (found) {
			//sort and choose alphabetically last result
			std::sort(matching.begin(), matching.end());
			pToFile = matching.back();

			if (this->reloadChoice == 0) {
				int chosen = 0;
				while (!chosen) {
					chosen = 1;
					std::cout << "A cache of a previous computation for this digit exists." << std::endl;
					std::cout << "Would you like to reload the most recent cache (" << pToFile << ")? y\\n" << std::endl;
					char choice;
					std::cin >> choice;
					if (choice == 'y') {}
					else if (choice == 'n') {
						std::cout << "Beginning computation without reloading." << std::endl;
						return 0;
					}
					else {
						std::cout << "Invalid input" << std::endl;
						// Ignore to the end of line
						std::cin.clear();
						std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
						chosen = 0;
					}
				}
			}
			else if (this->reloadChoice == 2) {
				return 0;
			}
			reloadFromCache(pToFile);
		}
		else {
			std::cout << "No progress cache file found. Beginning computation without reloading." << std::endl;
		}
		return 0;
	}

	int reloadFromCache(std::string pToFile) {
		std::cout << "Loading cache and continuing computation." << std::endl;

		std::ifstream cacheF(pToFile, std::ios::in);

		if (!cacheF.is_open()) {
			std::cerr << "Could not open " << pToFile << "!" << std::endl;
			return 1;
		}

		bool readSuccess = true;
		readSuccess = readSuccess && (cacheF >> std::dec >> this->digit->sumBegin);
		readSuccess = readSuccess && (cacheF >> std::hexfloat >> this->previousTime);
		for (int i = 0; i < 2; i++) readSuccess = readSuccess && (cacheF >> std::hex >> this->previousCache.s[i]);

		cacheF.close();

		if (!readSuccess) {
			std::cerr << "Cache reload failed due to improper file formatting!" << std::endl;
			return 1;
		}
		return 0;
	}

	//this function is meant to be run by an independent thread to output progress to the console
	void progressCheck() {

		std::deque<double> progressQ;
		std::deque<chr::high_resolution_clock::time_point> timeQ;
		int count = 0;
		inertialDouble progressPerSecond;
		while (!this->quit) {
			count++;

			uint64 readCurrent = *(this->digit->currentProgress);

			//currentProgress is always initialized at zero
			//when progress is reloaded or a segment after the first is started
			//the minimum for current progress should be the point from which progress was reloaded or the beginning of the segment, respectively
			readCurrent = std::max(readCurrent, this->digit->sumBegin);

			double progress = (double)(readCurrent - this->digit->segmentBegin) / (double)(this->digit->sumEnd - this->digit->segmentBegin);

			chr::high_resolution_clock::time_point now = chr::high_resolution_clock::now();
			progressQ.push_front(progress);
			timeQ.push_front(now);

			//progressQ and timeQ should be same size at all times
			if (progressQ.size() > 100) {
				progressQ.pop_back();
				timeQ.pop_back();
			}

			double progressInPeriod = progressQ.front() - progressQ.back();
			double elapsedPeriod = chr::duration_cast<chr::duration<double>>(timeQ.front() - timeQ.back()).count();
			if (elapsedPeriod > 0.0) {
				progressPerSecond.tug(progressInPeriod / elapsedPeriod);
			}
			else {
				progressPerSecond.tug(0.0);
			}

			double timeEst = (1.0 - progress) / (progressPerSecond.getValue());
			//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
			double elapsedTime = this->previousTime + (chr::duration_cast<chr::duration<double>>(now - *this->begin)).count();
			//only print every 10th cycle or 0.1 seconds
			if (count == 10) {
				count = 0;
				printf("Current progress is %3.3f%%. Estimated total runtime remaining is %8.3f seconds. Avg rate is %1.5f%%. Time elapsed is %8.3f seconds.\n", 100.0*progress, timeEst, 100.0*progressPerSecond.getValue(), elapsedTime);
			}

			bool resultsReady = true;

			resultsReady = resultsReady && (launchersTracked.size() > 0);

			for (bbpLauncher* launcher : launchersTracked) resultsReady = resultsReady && (launcher->hasCache());

			if (resultsReady) {

				uint64 contProcess = 0;
				sJ currStatus = this->previousCache;
				bool compareCacheEnd = false;
				bool cacheMisaligned = false;
				for (bbpLauncher* launcher : launchersTracked) {
					std::pair<sJ, uint64> launcherCache = launcher->getCacheFront();
					sJAdd(&currStatus, &launcherCache.first);
					if (compareCacheEnd) {
						if (contProcess != launcherCache.second) cacheMisaligned = true;
					}
					else {
						contProcess = launcherCache.second;
					}
					compareCacheEnd = true;
				}

				if (!cacheMisaligned) writeCache(contProcess, currStatus, elapsedTime);
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}

	void writeCache(uint64 cacheEnd, sJ cacheData, double elapsedTime) {
		char buffer[256];

		//minus 1 because cache range is [segmentBegin, contProcess)
		double savedProgress = (double)(cacheEnd - 1LLU - this->digit->segmentBegin) / (double)(this->digit->sumEnd - this->digit->segmentBegin);

		snprintf(buffer, sizeof(buffer), "progressCache/%s%09.6f.dat", this->progressFilenamePrefix.c_str(), 100.0*savedProgress);

		std::ofstream cacheF(buffer, std::ios::out);
		if (cacheF.is_open()) {
			printf("Writing data to disk\n");
			cacheF << cacheEnd << std::endl;
			//setprecision is required by msvc++ to output full precision doubles in hex
			//although the documentation of hexfloat clearly specifies hexfloat should ignore setprecision https://en.cppreference.com/w/cpp/io/manip/fixed
			cacheF << std::setprecision(13) << std::hexfloat << elapsedTime << std::endl;
			for (int i = 0; i < 2; i++) cacheF << std::hex << cacheData.s[i] << std::endl;
			cacheF.close();
		}
		else {
			fprintf(stderr, "Error opening file %s\n", buffer);
		}
	}
};

const std::string progressData::progressFilenamePrefixTemplate = "exponent%lluTotalSegments%lluSegment%lluProgress";