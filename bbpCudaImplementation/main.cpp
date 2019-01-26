#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <deque>
#include <mutex>
#include <atomic>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <map>
#ifdef __linux__
#include <experimental/filesystem>
#elif _WIN64
#include <filesystem>
#endif
#include <string>
#include <algorithm>
#include <csignal>
#include "kernel.cuh"

namespace chr = std::chrono;

const struct {
	const std::string STRIDEMULTIPLIER = "strideMultiplier",
		BLOCKCOUNT = "blockCount",
		SEGMENTS = "totalSegments",
		PRIMARYGPU = "primaryGpu",
		BENCHMARKBLOCKCOUNTS = "benchmarkBlockCounts",
		BENCHMARKTRIALS = "benchmarkTrials",
		BENCHMARKTARGET = "benchmarkTarget",
		BENCHMARKSTARTINGBLOCKCOUNT = "benchmarkStartingBlockCount",
		BENCHMARKBLOCKCOUNTINCREMENT = "benchmarkBlockCountIncrement",
		BENCHMARKTOTALINCREMENTS = "benchmarkTotalIncrements";
} propertyNames;

std::string propertiesFile = "application.properties";
std::string progressFilenamePrefixTemplate = "exponent%lluTotalSegments%lluSegment%lluProgress";
int totalGpus;
uint64 strideMultiplier;
//warpsize is 32 so optimal value is almost certainly a multiple of 32
const int threadCountPerBlock = 128;
//blockCount is trickier, and is probably a multiple of the number of streaming multiprocessors in a given gpu
int blockCount;
int primaryGpu;
int benchmarkBlockCounts;
int numRuns;
uint64 benchmarkTarget;
int startBlocks, blocksIncrement, incrementLimit;
const uint64 cachePeriod = 20000;
uint64 segments = 1;
bool stop = false;
void sigint_handler(int sig);

cudaError_t reduceSJ(sJ *c, unsigned int size);

//this class contains all needed data to define the work for a given segment of a digit
//and to synchronize that work between multiple GPUs
class digitData {
public:
	uint64 sumEnd = 0;
	uint64 startingExponent = 0;
	uint64 sumBegin = 0;
	uint64 segmentBegin = 0;
	std::atomic<uint64> launchCount;

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
	}
};

//this class facilitates tracking progress for a given segment of a digit
//including periodic caching to file, and reloading from a cache file
class progressData {
public:
	volatile uint64 * currentProgress;
	uint64 * deviceProg;
	sJ previousCache;
	double previousTime;
	std::deque<std::pair<sJ, uint64>> * currentResult;
	digitData * digit;
	volatile int quit = 0;
	cudaError_t error;
	chr::high_resolution_clock::time_point * begin;
	std::mutex * queueMtx;
	std::string progressFilenamePrefix;

	progressData(int gpus) {

		//these variables are linked between host and device memory allowing each to communicate about progress
		volatile uint64 *currProgHost;
		uint64 * currProgDevice;

		this->currentResult = new std::deque<std::pair<sJ, uint64>>[gpus];
		this->queueMtx = new std::mutex[gpus];

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
		this->quit = 0;
	}

	~progressData() {
		delete[] this->currentResult;
		delete[] this->queueMtx;
		//TODO: delete the device/host pointers?
	}

	int checkForProgressCache(digitData * data, uint64 totalSegments, uint64 segment, int reloadChoice) {
		this->digit = data;
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

			if (reloadChoice == 0) {
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
			else if (reloadChoice == 2) {
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
	}

	//this function is meant to be run by an independent thread to output progress to the console
	void progressCheck() {

		std::deque<double> progressQ;
		std::deque<chr::high_resolution_clock::time_point> timeQ;
		int count = 0;
		while (!this->quit) {
			count++;

			uint64 readCurrent = *(this->currentProgress);

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
			double progressPerSecond = progressInPeriod / elapsedPeriod;

			double timeEst = (1.0 - progress) / (progressPerSecond);
			//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
			double time = this->previousTime + (chr::duration_cast<chr::duration<double>>(now - *this->begin)).count();
			//only print every 10th cycle or 0.1 seconds
			if (count == 10) {
				count = 0;
				printf("Current progress is %3.3f%%. Estimated total runtime remaining is %8.3f seconds. Avg rate is %1.5f%%. Time elapsed is %8.3f seconds.\n", 100.0*progress, timeEst, 100.0*progressPerSecond, time);
			}

			bool resultsReady = true;

			for (int i = 0; i < totalGpus; i++) resultsReady = resultsReady && (this->currentResult[i].size() > 0);

			if (resultsReady) {

				uint64 contProcess = this->currentResult[0].front().second;

				char buffer[256];

				//minus 1 because cache range is [segmentBegin, contProcess)
				double savedProgress = (double)(contProcess - 1LLU - this->digit->segmentBegin) / (double)(this->digit->sumEnd - this->digit->segmentBegin);

				snprintf(buffer, sizeof(buffer), "progressCache/%s%09.6f.dat", this->progressFilenamePrefix.c_str(), 100.0*savedProgress);

				//would like to do this with ofstream and std::hexfloat
				//but msvc is a microsoft product so...

				std::ofstream cacheF(buffer, std::ios::out);
				if (cacheF.is_open()) {
					printf("Writing data to disk\n");
					cacheF << contProcess << std::endl;
					//setprecision is required by msvc++ to output full precision doubles in hex
					//although the documentation of hexfloat clearly specifies hexfloat should ignore setprecision https://en.cppreference.com/w/cpp/io/manip/fixed
					cacheF << std::setprecision(13) << std::hexfloat << time << std::endl;
					sJ currStatus = this->previousCache;
					for (int i = 0; i < totalGpus; i++) {
						this->queueMtx[i].lock();
						sJAdd(&currStatus, &this->currentResult[i].front().first);
						this->currentResult[i].pop_front();
						this->queueMtx[i].unlock();
					}
					for (int i = 0; i < 2; i++) cacheF << std::hex << currStatus.s[i] << std::endl;
					cacheF.close();
				}
				else {
					fprintf(stderr, "Error opening file %s\n", buffer);
				}
			}

			std::this_thread::sleep_for(std::chrono::milliseconds(10));
		}
	}
};

class bbpLauncher {
public:
	static int totalLaunchers;
	sJ output;
	int gpu = 0;
	int totalGpus = 0;
	int size = 0;
	cudaError_t error;
	digitData * data;
	progressData * prog;

	bbpLauncher() {
		this->gpu = totalLaunchers++;
	}

	void initialize(digitData * data, progressData * prog) {
		this->data = data;
		this->prog = prog;
	}

	// Helper function for using CUDA
	void launch()//cudaError_t addWithCuda(sJ *output, unsigned int size, TYPE digit)
	{
		sJ *dev_c = 0;
		sJ* c = new sJ[1];
		sJ *dev_ex = 0;

		cudaError_t cudaStatus;

		uint64 launchWidth, neededLaunches, currentLaunch;

		uint64 lastWrite = 0;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(gpu);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffer for temp vector
		cudaStatus = cudaMalloc((void**)&dev_ex, this->size * sizeof(sJ) * 7);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Allocate GPU buffer for output vector
		cudaStatus = cudaMalloc((void**)&dev_c, this->size * sizeof(sJ) * 7);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		launchWidth = (uint64)this->size * strideMultiplier;

		//need to round up
		//because bbp condition for stopping is <= digit, number of total elements in summation is 1 + digit
		//even when digit/launchWidth is an integer, it is necessary to add 1
		neededLaunches = ((this->data->sumEnd - this->data->sumBegin) / launchWidth) + 1LLU;
		while (!stop && ((currentLaunch = this->data->launchCount++) < neededLaunches)) {

			uint64 begin = this->data->sumBegin + (launchWidth * currentLaunch);
			uint64 end = this->data->sumBegin + (launchWidth * (currentLaunch + 1)) - 1;
			if (end > this->data->sumEnd) end = this->data->sumEnd;

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bbpKernel on gpu %d!\n", cudaStatus, this->gpu);
				goto Error;
			}
			//after exactly cachePeriod number of launches since last period between all gpus, write all data computed during and before the period to status buffer for progress thread to save
			if ((currentLaunch - lastWrite) >= cachePeriod) {

				lastWrite += cachePeriod;

				//copy current results into temp array to reduce and update status
				cudaStatus = cudaMemcpy(dev_ex, dev_c, size * sizeof(sJ) * 7, cudaMemcpyDeviceToDevice);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed in status update!\n");
					goto Error;
				}

				cudaStatus = reduceSJ(dev_ex, size * 7);

				if (cudaStatus != cudaSuccess) {
					goto Error;
				}

				// Copy result (reduced into first element) from GPU buffer to host memory.
				cudaStatus = cudaMemcpy(c, dev_ex, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed in status update!\n");
					goto Error;
				}

				this->prog->queueMtx[this->gpu].lock();
				this->prog->currentResult[this->gpu].emplace_back(c[0], this->data->sumBegin + (launchWidth * lastWrite));
				this->prog->queueMtx[this->gpu].unlock();
			}

			// Launch a kernel on the GPU with one thread for each element.
			bbpPassThrough(threadCountPerBlock, blockCount * 7, dev_c, this->prog->deviceProg, this->data->startingExponent, begin, end, strideMultiplier);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "bbpKernel launch failed on gpu%d: %s\n", this->gpu, cudaGetErrorString(cudaStatus));
				goto Error;
			}

			////give the rest of the computer some gpu time to reduce system choppiness
			//if (primaryGpu) {
			//	std::this_thread::sleep_for(std::chrono::milliseconds(1));
			//}
		}

		cudaStatus = reduceSJ(dev_c, size * 7);

		if (cudaStatus != cudaSuccess) {
			goto Error;
		}

		// Copy result (reduced into first element) from GPU buffer to host memory.
		cudaStatus = cudaMemcpy(c, dev_c, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy failed!\n");
			goto Error;
		}

		this->output = c[0];

	Error:
		free(c);
		cudaFree(dev_c);
		cudaFree(dev_ex);

		this->error = cudaStatus;
	}
};

int bbpLauncher::totalLaunchers = 0;

//standard tree-based parallel reduce
cudaError_t reduceSJ(sJ *c, unsigned int size) {
	cudaError_t cudaStatus;
	while (size > 1) {
		int nextSize = (size + 1) >> 1;

		//size is odd
		if (size & 1) reduceSJPassThrough(32, 32, c, nextSize, nextSize - 1);
		//size is even
		else reduceSJPassThrough(32, 32, c, nextSize, nextSize);

		// Check for any errors launching the kernel
		cudaStatus = cudaGetLastError();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "reduceSJKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
			return cudaStatus;
		}

		// cudaDeviceSynchronize waits for the kernel to finish, and returns
		// any errors encountered during the launch.
		cudaStatus = cudaDeviceSynchronize();
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching reduceSJKernel!\n", cudaStatus);
			return cudaStatus;
		}

		size = nextSize;
	}
	return cudaStatus;
}

int loadProperties() {
	std::cout << "Loading properties from " << propertiesFile << std::endl;
	std::ifstream propF(propertiesFile, std::ios::in);

	if (!propF.is_open()) {
		std::cerr << "Could not open " << propertiesFile << "!" << std::endl;
		return 1;
	}

	std::map<std::string, std::string> properties;

	while (!propF.eof()) {
		std::string propertyName, propertyValue;
		propF >> propertyName >> propertyValue;
		properties.emplace(propertyName, propertyValue);
	}

	propF.close();

	std::string checkProps[10] = {
		propertyNames.STRIDEMULTIPLIER,
		propertyNames.BLOCKCOUNT,
		propertyNames.PRIMARYGPU,
		propertyNames.BENCHMARKBLOCKCOUNTS,
		propertyNames.BENCHMARKTRIALS,
		propertyNames.BENCHMARKTARGET,
		propertyNames.BENCHMARKSTARTINGBLOCKCOUNT,
		propertyNames.BENCHMARKBLOCKCOUNTINCREMENT,
		propertyNames.BENCHMARKTOTALINCREMENTS,
		propertyNames.SEGMENTS
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
	benchmarkBlockCounts = std::stoi(properties.at(propertyNames.BENCHMARKBLOCKCOUNTS));
	numRuns = std::stoi(properties.at(propertyNames.BENCHMARKTRIALS));
	benchmarkTarget = std::stoull(properties.at(propertyNames.BENCHMARKTARGET));
	startBlocks = std::stoi(properties.at(propertyNames.BENCHMARKSTARTINGBLOCKCOUNT));
	blocksIncrement = std::stoi(properties.at(propertyNames.BENCHMARKBLOCKCOUNTINCREMENT));
	incrementLimit = std::stoi(properties.at(propertyNames.BENCHMARKTOTALINCREMENTS));
	segments = std::stoull(properties.at(propertyNames.SEGMENTS));
	if (segments == 0) segments = 1;

	return 0;
}

int benchmark() {
	digitData data(benchmarkTarget, 1, 1);
	totalGpus = 1;
	progressData prog(totalGpus);
	if (prog.error != cudaSuccess) return 1;
	bbpLauncher gpuData;
	gpuData.totalGpus = totalGpus;
	gpuData.initialize(&data, &prog);
	std::vector<std::pair<double, int>> timings;
	for (blockCount = startBlocks; blockCount <= (startBlocks + incrementLimit * blocksIncrement); blockCount += blocksIncrement) {
		double total = 0.0;
		for (int j = 0; j < numRuns; j++) {
			data.launchCount = 0;
			chr::high_resolution_clock::time_point start = chr::high_resolution_clock::now();
			gpuData.size = threadCountPerBlock * blockCount;
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

int main(int argc, char** argv) {

	if (loadProperties()) return 1;

	signal(SIGINT, sigint_handler);

	cudaError_t cudaStatus = cudaGetDeviceCount(&totalGpus);

	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount failed!\n");
		return 1;
	}
	if (!totalGpus) {
		fprintf(stderr, "No GPUs detected in system!\n");
		return 1;
	}

	if (benchmarkBlockCounts) {
		return benchmark();
	}

	const int arraySize = threadCountPerBlock * blockCount;
	uint64 hexDigitPosition;
	if (argc == 1) {
		std::cout << "Input hexDigit to calculate (1-indexed):" << std::endl;
		std::cin >> hexDigitPosition;
	}
	else {
		hexDigitPosition = std::stoull(argv[1]);
	}
	
	uint64 segmentNumber = 0;
	if (argc < 3 && segments > 1) {
		std::cout << "Input segment number to calculate (1 - " << segments << ")" << std::endl;
		std::cin >> segmentNumber;
		segmentNumber--;
	}
	else if(segments > 1) {
		segmentNumber = std::stoull(argv[2]);
		segmentNumber--;
	}

	digitData data(hexDigitPosition, segments, segmentNumber);

	std::thread * handles = new std::thread[totalGpus];
	bbpLauncher * gpuData = new bbpLauncher[totalGpus];

	int reloadChoice = 0;
	if (argc >= 4) {
		std::string reloadChoiceString = argv[3];
		if (reloadChoiceString.compare("y") == 0) {
			reloadChoice = 1;
		}
		else if (reloadChoiceString.compare("n") == 0) {
			reloadChoice = 2;
		}
	}

	progressData prog(totalGpus);
	if (prog.error != cudaSuccess) return 1;
	if (prog.checkForProgressCache(&data, segments, segmentNumber + 1LLU, reloadChoice)) return 1;

	chr::high_resolution_clock::time_point start = chr::high_resolution_clock::now();
	prog.begin = &start;

	std::thread progThread(&progressData::progressCheck, &prog);

	for (int i = 0; i < totalGpus; i++) {
		gpuData[i].totalGpus = totalGpus;
		gpuData[i].size = arraySize;
		gpuData[i].initialize(&data, &prog);

		handles[i] = std::thread(&bbpLauncher::launch, gpuData + i);
	}

	sJ cudaResult = prog.previousCache;

	for (int i = 0; i < totalGpus; i++) {

		handles[i].join();

		cudaStatus = gpuData[i].error;
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaBbpLaunch failed on gpu%d!\n", i);
			stop = true;
		}

		sJ output = gpuData[i].output;

		//sum results from gpus
		sJAdd(&cudaResult, &output);
	}

	//tell the progress thread to quit
	prog.quit = 1;

	progThread.join();

	delete[] handles;
	delete[] gpuData;

	if (!stop) {

		chr::high_resolution_clock::time_point end = chr::high_resolution_clock::now();

		printf("pi at hexadecimal digit %llu is %016llX %016llX\n",
			hexDigitPosition, cudaResult.s[1], cudaResult.s[0]);

		//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
		printf("Computed in %.8f seconds\n", prog.previousTime + (chr::duration_cast<chr::duration<double>>(end - start)).count());

		const char * completionPathFormat = "completed/segmented%dExponent%lluSegment%dBase2Complete.dat";
		char buffer[256];
		snprintf(buffer, sizeof(buffer), completionPathFormat, segments, data.startingExponent, segmentNumber + 1);
		std::ofstream completedF(buffer, std::ios::out);
		if (completedF.is_open()) {
			completedF << std::hex << std::setfill('0') << std::setw(16);
			for (int i = 0; i < 2; i++) completedF << cudaResult.s[i] << std::endl;
			completedF.close();
		}
		else {
			fprintf(stderr, "Error opening file %s\n", buffer);
		}
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
		stop = true;
	}
}