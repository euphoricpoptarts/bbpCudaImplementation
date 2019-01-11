#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <deque>
#include <mutex>
#include <atomic>
#include <iostream>
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

std::string propertiesFile = "application.properties";
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
bool stop = false;
void sigint_handler(int sig);

cudaError_t reduceSJ(sJ *c, unsigned int size);

class digitData {
public:
	uint64 sumEnd = 0;
	uint64 startingExponent = 0;
	uint64 beginFrom = 0;

	digitData(uint64 digitInput) {
		//subtract 1 to convert to 0-indexed
		digitInput--;

		//4*hexDigitPosition converts from exponent of 16 to exponent of 2
		//adding 128 for fixed-point division algorithm
		//adding 8 for maximum size of coefficient (so that all coefficients can be expressed by subtracting an integer from the exponent)
		//subtracting 6 for the division by 64 of the whole sum
		this->startingExponent = (4LLU * digitInput) + 128LLU + 2LLU;

		//the end of the sum does not have the addition by 8 so that all calculations will be a positive exponent of 2 after factoring in the coefficient
		//this leaves out a couple potentially positive exponents of 2 (could potentially just check subtraction in modExpLeftToRight and keep the addition by 8)
		this->sumEnd = (4LLU * digitInput - 6LLU + 128LLU) / 10LLU;
	}
};

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
	std::atomic<uint64> launchCount;

	progressData(int gpus) {
		std::atomic_init(&this->launchCount, 0);

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

	int checkForProgressCache(digitData * data) {
		this->digit = data;
		std::string target = "exponent" + std::to_string(this->digit->startingExponent) + "Base";
		std::string pToFile;
		std::vector<std::string> matching;
		int found = 0;
		for (auto& element : std::experimental::filesystem::directory_iterator("progressCache")) {
			std::string name = element.path().filename().string();
			//filename begins with desired string
			if (name.compare(0, target.length(), target) == 0) {
				matching.push_back(element.path().string());
				found = 1;
			}
		}
		if (found) {
			//sort and choose alphabetically last result
			std::sort(matching.begin(), matching.end());
			pToFile = matching.back();

			int chosen = 0;
			while (!chosen) {
				chosen = 1;
				std::cout << "A cache of a previous computation for this digit exists." << std::endl;
				std::cout << "Would you like to reload the most recent cache (" << pToFile << ")? y\\n" << std::endl;
				char choice;
				std::cin >> choice;
				if (choice == 'y') {
					std::cout << "Loading cache and continuing computation." << std::endl;
					FILE * cacheF = fopen(pToFile.c_str(), "r");

					if (cacheF == NULL) {
						std::cout << "Could not open " << pToFile << "!" << std::endl;
						return 1;
					}

					int readLines = 0;

					readLines += fscanf(cacheF, "%llu", &this->digit->beginFrom);
					readLines += fscanf(cacheF, "%la", &this->previousTime);
					for (int i = 0; i < 2; i++) readLines += fscanf(cacheF, "%llX", &this->previousCache.s[i]);
					fclose(cacheF);
					//4 lines of data should have been read, 1 continuation point, 1 time, and 2 data points
					if (readLines != 4) {
						std::cout << "Data reading failed!" << std::endl;
						return 1;
					}
				}
				else if (choice == 'n') {
					std::cout << "Beginning computation without reloading." << std::endl;
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
		else {
			std::cout << "No progress cache file found. Beginning computation without reloading." << std::endl;
		}
		return 0;
	}

	//this function is meant to be run by an independent thread to output progress to the console
	void progressCheck() {

		std::deque<double> progressQ;
		std::deque<chr::high_resolution_clock::time_point> timeQ;
		int count = 0;
		while (!this->quit) {
			count++;
			double progress = (double)(*(this->currentProgress)) / (double)this->digit->sumEnd;

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

				char buffer[100];

				double savedProgress = (double)(contProcess - 1LLU) / (double)this->digit->sumEnd;

				snprintf(buffer, sizeof(buffer), "progressCache/exponent%lluBase2Progress%09.6f.dat", this->digit->startingExponent, 100.0*savedProgress);

				//would like to do this with ofstream and std::hexfloat
				//but msvc is a microsoft product so...
				FILE * file;
				file = fopen(buffer, "w+");
				if (file != NULL) {
					printf("Writing data to disk\n");
					fprintf(file, "%llu\n", contProcess);
					fprintf(file, "%a\n", time);
					sJ currStatus = this->previousCache;
					for (int i = 0; i < totalGpus; i++) {
						this->queueMtx[i].lock();
						sJAdd(&currStatus, &this->currentResult[i].front().first);
						this->currentResult[i].pop_front();
						this->queueMtx[i].unlock();
					}
					for (int i = 0; i < 2; i++) fprintf(file, "%llX\n", currStatus.s[i]);
					fclose(file);
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
		neededLaunches = ((this->data->sumEnd - this->data->beginFrom) / launchWidth) + 1LLU;
		while (!stop && ((currentLaunch = this->prog->launchCount++) < neededLaunches)) {

			uint64 begin = this->data->beginFrom + (launchWidth * currentLaunch);
			uint64 end = this->data->beginFrom + (launchWidth * (currentLaunch + 1)) - 1;
			if (end > this->data->sumEnd) end = this->data->sumEnd;

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
				this->prog->currentResult[this->gpu].emplace_back(c[0], this->data->beginFrom + (launchWidth * lastWrite));
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

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bbpKernel on gpu %d!\n", cudaStatus, this->gpu);
				goto Error;
			}

			//give the rest of the computer some gpu time to reduce system choppiness
			if (primaryGpu) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
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
	FILE * propF = fopen(propertiesFile.c_str(), "r");

	if (propF == NULL) {
		std::cout << "Could not open " << propertiesFile << "!" << std::endl;
		return 1;
	}

	int readLines = 0;

	readLines += fscanf(propF, "%llu", &strideMultiplier);
	readLines += fscanf(propF, "%d", &blockCount);
	readLines += fscanf(propF, "%d", &primaryGpu);
	readLines += fscanf(propF, "%d", &benchmarkBlockCounts);
	readLines += fscanf(propF, "%d", &numRuns);
	readLines += fscanf(propF, "%llu", &benchmarkTarget);
	readLines += fscanf(propF, "%d", &startBlocks);
	readLines += fscanf(propF, "%d", &blocksIncrement);
	readLines += fscanf(propF, "%d", &incrementLimit);
	if (readLines != 9) {
		std::cout << "Properties loading failed!" << std::endl;
		return 1;
	}

	return 0;
}

int benchmark() {
	digitData data(benchmarkTarget);
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
			prog.launchCount = 0;
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

int main() {

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
	std::cout << "Input hexDigit to calculate (1-indexed):" << std::endl;
	std::cin >> hexDigitPosition;

	digitData data(hexDigitPosition);

	std::thread * handles = new std::thread[totalGpus];
	bbpLauncher * gpuData = new bbpLauncher[totalGpus];

	progressData prog(totalGpus);
	if (prog.error != cudaSuccess) return 1;
	if (prog.checkForProgressCache(&data)) return 1;

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