#include "progressData.h"
#ifndef NO_BOOST
#include "restClientDelegator.h"
#endif

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

void progressData::writeCache(uint64 cacheEnd, uint128 cacheData, double elapsedTime) {

    if(!this->cacheDirExists){
        std::cerr << "The progressCache directory does not exist! Can't save progress!" << std::endl;
        return;
    }

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
		cacheF << std::hex << cacheData.lsw << std::endl;
		cacheF << std::hex << cacheData.msw << std::endl;
		cacheF.close();
	}
	else {
		fprintf(stderr, "Error opening file %s\n", buffer);
	}
}

int progressData::reloadFromCache(std::string pToFile) {
	std::cout << "Loading cache and continuing computation." << std::endl;

	std::ifstream cacheF(pToFile, std::ios::in);

	if (!cacheF.is_open()) {
		std::cerr << "Could not open " << pToFile << "!" << std::endl;
		return 1;
	}

	bool readSuccess = true;
	readSuccess = readSuccess && (cacheF >> std::dec >> this->digit->sumBegin);
	readSuccess = readSuccess && (cacheF >> std::hexfloat >> this->previousTime);
	readSuccess = readSuccess && (cacheF >> std::hex >> this->previousCache.lsw);
	readSuccess = readSuccess && (cacheF >> std::hex >> this->previousCache.msw);

	cacheF.close();

	if (!readSuccess) {
		std::cerr << "Cache reload failed due to improper file formatting!" << std::endl;
		return 1;
	}
	return 0;
}


std::string progressData::controlledUuids() {
	std::string controlledUuids;
	bool first = true;
	for (bbpLauncher* launcher : launchersTracked) {
		if (!first) controlledUuids.append("-");
		controlledUuids.append(launcher->getUuid());
		first = false;
	}
	return controlledUuids;
}

void progressData::requestWork() {
#ifndef NO_BOOST
	delegator->addWorkGetToQueue(this);
	workRequested = true;
#endif
}

void progressData::sendResult(uint128 result, double time) {
#ifndef NO_BOOST
	delegator->addResultPutToQueue(digit, result, time);
	delete digit;
#endif
}

void progressData::blockForWork() {
	std::mutex mtx;
	std::unique_lock<std::mutex> ul(mtx);
	cv.wait(ul, [this] {return this->workAssigned; });
	
	//the restClientDelegator holds a pointer to the object pointed to before this assignment
	//it is the restClientDelegator's responsibility to delete it
	digit = nextWorkUnit;

	nextWorkUnit = nullptr;

	workAssigned = false;
	workRequested = false;
}

void progressData::setStopCheck(uint64 remoteId) {
	this->checkToStop = remoteId;
}

progressData::progressData(digitData * data)
{
	this->digit = data;
	this->quit = 0;
	this->reloadChoice = 0;
	this->checkToStop = 0;
}

progressData::progressData(restClientDelegator * delegator)
{
	this->delegator = delegator;
	this->hasDelegator = true;
	this->checkToStop = 0;
}

void progressData::setReloadPolicy(int choice) {
	this->reloadChoice = choice;
}

void progressData::assignWork(digitData * data) {
	this->nextWorkUnit = data;
	this->workAssigned = true;
	cv.notify_all();
}

void progressData::addLauncherToTrack(bbpLauncher * launcher) {
	launchersTracked.push_back(launcher);
}

bool progressData::checkForProgressCache() {
	char buffer[256];
	snprintf(buffer, sizeof(buffer), progressFilenamePrefixTemplate.c_str(), digit->startingExponent, digit->segments, digit->segmentNumber + 1);
	this->progressFilenamePrefix = buffer;
	std::string pToFile;
	std::vector<std::string> matching;
	int found = 0;
    std::filesystem::path cacheDir("progressCache");
    if(!std::filesystem::exists(cacheDir)){
        std::cout << "Could not find progressCache in the current directory because it does not exist. Please create progressCache as a directory in order to use progress saving feature!" << std::endl;
        return true;
    } else if(!std::filesystem::is_directory(cacheDir)){
        std::cout << "Found progressCache in the current directory but it is not a directory. Please create progressCache as a directory in order to use progress saving feature!" << std::endl;
        return true;
    }
    this->cacheDirExists = true;
	for (auto& element : std::filesystem::directory_iterator(cacheDir)) {
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
					return true;
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
			return true;
		}
		reloadFromCache(pToFile);
	}
	else {
		std::cout << "No progress cache file found. Beginning computation without reloading." << std::endl;
	}
	return true;
}

void progressData::beginWorkUnit() {
	threadLauncherPairs.clear();
	this->begin = chr::steady_clock::now();
	for (bbpLauncher* launcher : launchersTracked) {
		launcher->setData(digit);
		threadLauncherPairs.emplace_back(std::thread(&bbpLauncher::launch, launcher), launcher);
	}
}

bool progressData::fetchResultFromLaunchers(uint128& result, double& time) {
	if (areLaunchersComplete()) {
		uint128 cudaResult;
		cudaError_t cudaStatus;

		for (std::pair<std::thread, bbpLauncher *>& pair : threadLauncherPairs) {
			pair.first.join();

			cudaStatus = pair.second->getError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaBbpLaunch failed on gpu %s!\n", pair.second->getUuid().c_str());
			}

			uint128 output = pair.second->getResult();

			//sum results from gpus
			sJAdd(&cudaResult, &output);
		}
		result = cudaResult;
		time = (chr::duration_cast<chr::duration<double>>(chr::steady_clock::now() - this->begin)).count();
		return true;
	}
	else {
		for (std::pair<std::thread, bbpLauncher *>& pair : threadLauncherPairs) {
			pair.second->quit();
			pair.first.join();
		}
		return false;
	}
}

void progressData::beginWorking() {
	requestWork();
	while (!globalStopSignal) {

		blockForWork();

		this->previousTime = 0.0;
		this->quit = 0;
		beginWorkUnit();
		progressCheck();
		uint128 result;
		double time = 0;
		if (fetchResultFromLaunchers(result, time)) {
			printf("result of work-unit is %016llX %016llX\n",
				result.msw, result.lsw);
			printf("Computed in %.8f seconds\n", time);
			sendResult(result, time);
		}
		
		if (!this->workRequested) {
			requestWork();
		}
	}
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
	}
}

void progressData::runSingleWorkUnit() {
	this->quit = 0;
	if (!checkForProgressCache()) return;
    std::cout << "Beginning computation of hex digit " << this->digit->digitPos << std::endl;
	beginWorkUnit();
	progressCheck();
	uint128 result;
	double time = 0;
	if (fetchResultFromLaunchers(result, time)) {
		sJAdd(&result, &previousCache);

		printf("pi at hexadecimal digit %llu is %016llX %016llX\n",
			digit->digitPos, result.msw, result.lsw);

		//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
		double totalTime = previousTime + time;
		printf("Computed in %.8f seconds\n", totalTime);

		const char * completionPathFormat = "completed/segmented%dExponent%lluSegment%dBase2Complete.dat";
		char fileNameBuffer[256];
		snprintf(fileNameBuffer, sizeof(fileNameBuffer), completionPathFormat, digit->segments, digit->startingExponent, digit->segmentNumber + 1);
		std::ofstream completedF(fileNameBuffer, std::ios::out);
		if (completedF.is_open()) {
			completedF << std::hex << std::setfill('0');
			completedF << std::setw(16) << result.lsw << std::endl;
			completedF << std::setw(16) << result.msw << std::endl;
			completedF << std::hexfloat << std::setprecision(13) << totalTime << std::endl;
			completedF.close();
		}
		else {
			fprintf(stderr, "Error opening file %s\n", fileNameBuffer);
		}
	}
	cudaError_t cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!\n");
	}
}

//this function is meant to be run by an independent thread to output progress to the console
void progressData::progressCheck() {

	std::deque<double> progressQ;
	std::deque<chr::steady_clock::time_point> timeQ;
	chr::steady_clock::time_point lastProgOutput, lastServerProgUpdate;
	lastProgOutput = chr::steady_clock::now();
	lastServerProgUpdate = lastProgOutput;
	inertialDouble progressPerSecond;
	while (!this->quit) {

		uint64 readCurrent = *(this->digit->currentProgress);

		//currentProgress is always initialized at zero
		//when progress is reloaded or a segment after the first is started
		//the minimum for current progress should be the point from which progress was reloaded or the beginning of the segment, respectively
		readCurrent = std::max(readCurrent, this->digit->sumBegin);

		double progress = (double)(readCurrent - this->digit->segmentBegin) / (double)(this->digit->sumEnd - this->digit->segmentBegin);

		chr::steady_clock::time_point now = chr::steady_clock::now();
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
		double elapsedTime = this->previousTime + (chr::duration_cast<chr::duration<double>>(now - this->begin)).count();
		//only print every 0.1 seconds
		if (chr::duration_cast<chr::duration<double>>(now - lastProgOutput).count() >= 0.1) {
			lastProgOutput += chr::milliseconds(100);
			//printf("Current progress is %3.3f%%. Estimated total runtime remaining is %8.3f seconds. Avg rate is %1.5f%%. Time elapsed is %8.3f seconds.\n", 100.0*progress, timeEst, 100.0*progressPerSecond.getValue(), elapsedTime);
		}
		//only update server every second
		if (chr::duration_cast<chr::duration<double>>(now - lastServerProgUpdate).count() >= 1.0) {
			lastServerProgUpdate += chr::seconds(1);
#ifndef NO_BOOST
			if(this->hasDelegator) delegator->addReservationExtensionPutToQueue(this->digit, 100.0*progress, elapsedTime, this);
#endif
		}

		bool resultsReady = true;

		resultsReady = resultsReady && (launchersTracked.size() > 0);

		for (bbpLauncher* launcher : launchersTracked) resultsReady = resultsReady && (launcher->hasCache());

		if (resultsReady) {

			uint64 contProcess = 0;
			uint128 currStatus = this->previousCache;
			bool compareCacheEnd = false;
			bool cacheMisaligned = false;
			for (bbpLauncher* launcher : launchersTracked) {
				std::pair<uint128, uint64> launcherCache = launcher->getCacheFront();
				sJAdd(&currStatus, &launcherCache.first);
				if (compareCacheEnd) {
					if (contProcess != launcherCache.second) cacheMisaligned = true;
				}
				else {
					contProcess = launcherCache.second;
				}
				compareCacheEnd = true;
			}

			if (!cacheMisaligned) {
#ifndef NO_BOOST
				if(this->hasDelegator) delegator->addProgressUpdatePutToQueue(digit, currStatus, contProcess, elapsedTime);
#endif
				writeCache(contProcess, currStatus, elapsedTime);
			}
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(10));
		if (timeEst < 1.0 && this->hasDelegator && !this->workRequested) {
			requestWork();
		}

		//if (timeEst < 0.5) {
			this->quit = areLaunchersComplete();
		//}
		if (this->digit->remoteId > 0 && checkToStop.exchange(0) == this->digit->remoteId) {
			this->quit = true;
		}
	}
}

bool progressData::areLaunchersComplete() {
	bool launchersComplete = true;
	for (bbpLauncher* launcher : launchersTracked) launchersComplete = launchersComplete && launcher->isComplete();
	return launchersComplete;
}

const std::string progressData::progressFilenamePrefixTemplate = "exponent%lluTotalSegments%lluSegment%lluProgress";
