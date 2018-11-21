
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "device_atomic_functions.h"

#include <stdio.h>
#include <math.h>
#include <chrono>
#include <thread>
#include <deque>
#include <atomic>
#include <iostream>
#ifdef __linux__
#include <experimental/filesystem>
#elif _WIN64
#include <filesystem>
#endif
#include <string>
#include <algorithm>

#define uint32 unsigned int
#define uint64 unsigned long long

namespace chr = std::chrono;

std::string propertiesFile = "application.properties";
int totalGpus;
uint64 strideMultiplier;
//warpsize is 32 so optimal value is probably always a multiple of 32
const int threadCountPerBlock = 128;
//this is more difficult to optimize but seems to not like odd numbers
int blockCount;
__device__  __constant__ const uint64 int64MaxBit = 0x8000000000000000;
__device__ int printOnce = 0;
int primaryGpu;

struct sJ {
	uint64 s[2] = { 0, 0};
};

__global__ void bbpKernel(sJ *c, volatile uint64 *progress, uint64 startingExponent, int gpuNum, uint64 begin, uint64 end, uint64 stride);
cudaError_t reduceSJ(sJ *c, unsigned int size);

//adds all elements of addend and augend, storing in addend
__device__ __host__ void sJAdd(sJ* addend, const sJ* augend) {
	addend->s[0] += augend->s[0];
	addend->s[1] += augend->s[1];
	if (addend->s[0] < augend->s[0]) addend->s[1]++;
}

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
	volatile uint64 * deviceProg;
	sJ previousCache;
	double previousTime;
	sJ * status;
	volatile uint64 * nextStrideBegin;
	uint64 maxProgress;
	volatile int quit = 0;
	cudaError_t error;
	chr::high_resolution_clock::time_point * begin;
	volatile std::atomic<int> dataWritten;

	progressData(int gpus) {
		std::atomic_init(&this->dataWritten, 0);

		//these variables are linked between host and device memory allowing each to communicate about progress
		volatile uint64 *currProgHost, *currProgDevice;

		this->status = new sJ[gpus];
		this->nextStrideBegin = new uint64[gpus];

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
		delete[] this->status;
		delete[] this->nextStrideBegin;
		//TODO: delete the device/host pointers?
	}

	int checkForProgressCache(digitData * data) {
		this->maxProgress = data->sumEnd;
		std::string target = "digit" + std::to_string(data->sumEnd) + "Base";
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

					readLines += fscanf(cacheF, "%llu", &data->beginFrom);
					readLines += fscanf(cacheF, "%la", &this->previousTime);
					for (int i = 0; i < 2; i++) readLines += fscanf(cacheF, "%llX", &this->previousCache.s[i]);
					fclose(cacheF);
					//9 lines of data should have been read, 1 continuation point, 1 time, and 7 data points
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
			double progress = (double)(*(this->currentProgress)) / (double)this->maxProgress;

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

			int expected = totalGpus;

			if (std::atomic_compare_exchange_strong(&this->dataWritten, &expected, 0)) {

				//ensure all sJs in status are from same stride
				//this should always be the case since each 1000 strides are separated by about 90 seconds currently
				//it would be very unlikely for one gpu to get 1000 strides ahead of another, unless the GPUs were not the same
				int sJsAligned = 1;
				uint64 contProcess = this->nextStrideBegin[0];
				for (int i = 1; i < totalGpus; i++) sJsAligned &= (this->nextStrideBegin[i] == contProcess);

				if (sJsAligned) {

					char buffer[100];

					double savedProgress = (double)(contProcess - 1LLU) / (double)this->maxProgress;

					snprintf(buffer, sizeof(buffer), "progressCache/digit%lluBase1024Progress%09.6f.dat", this->maxProgress, 100.0*savedProgress);

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
							sJAdd(&currStatus, this->status + i);
						}
						for (int i = 0; i < 2; i++) fprintf(file, "%llX\n", currStatus.s[i]);
						fclose(file);
					}
					else {
						fprintf(stderr, "Error opening file %s\n", buffer);
					}
				}
				else {
					fprintf(stderr, "sJs are misaligned, could not write to disk!\n");
					for (int i = 0; i < totalGpus; i++) fprintf(stderr, "sJ %d alignment is %llu\n", i, this->nextStrideBegin[i]);
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

		uint64 stride, launchWidth, neededLaunches;

		// Choose which GPU to run on, change this on a multi-GPU system.
		cudaStatus = cudaSetDevice(gpu);
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
			goto Error;
		}

		// Allocate GPU buffer for temp vector
		cudaStatus = cudaMalloc((void**)&dev_ex, this->size * sizeof(sJ));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		// Allocate GPU buffer for output vector
		cudaStatus = cudaMalloc((void**)&dev_c, this->size * sizeof(sJ));
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaMalloc failed!");
			goto Error;
		}

		stride = (uint64)this->size * (uint64)this->totalGpus;

		launchWidth = stride * strideMultiplier;

		//need to round up
		//because bbp condition for stopping is <= digit, number of total elements in summation is 1 + digit
		//even when digit/launchWidth is an integer, it is necessary to add 1
		neededLaunches = ((this->data->sumEnd - this->data->beginFrom) / launchWidth) + 1LLU;

		for (uint64 launch = 0; launch < neededLaunches; launch++) {

			uint64 begin = this->data->beginFrom + (launchWidth * launch);
			uint64 end = this->data->beginFrom + (launchWidth * (launch + 1)) - 1;
			if (end > this->data->sumEnd) end = this->data->sumEnd;

			// Launch a kernel on the GPU with one thread for each element.
			bbpKernel << <blockCount, threadCountPerBlock >> > (dev_c, this->prog->deviceProg, this->data->startingExponent, this->gpu, begin, end, stride);

			// Check for any errors launching the kernel
			cudaStatus = cudaGetLastError();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "bbpKernel launch failed on gpu%d: %s\n", gpu, cudaGetErrorString(cudaStatus));
				goto Error;
			}

			// cudaDeviceSynchronize waits for the kernel to finish, and returns
			// any errors encountered during the launch.
			cudaStatus = cudaDeviceSynchronize();
			if (cudaStatus != cudaSuccess) {
				fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching bbpKernel on gpu 1!\n", cudaStatus);
				goto Error;
			}

			//on every 1000th launch write data to status buffer for progress thread to save
			if (launch % 1000 == 0 && launch) {

				//copy current results into temp array to reduce and update status
				cudaStatus = cudaMemcpy(dev_ex, dev_c, size * sizeof(sJ), cudaMemcpyDeviceToDevice);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed in status update!\n");
					goto Error;
				}

				cudaStatus = reduceSJ(dev_ex, size);

				if (cudaStatus != cudaSuccess) {
					goto Error;
				}

				// Copy result (reduced into first element) from GPU buffer to host memory.
				cudaStatus = cudaMemcpy(c, dev_ex, 1 * sizeof(sJ), cudaMemcpyDeviceToHost);
				if (cudaStatus != cudaSuccess) {
					fprintf(stderr, "cudaMemcpy failed in status update!\n");
					goto Error;
				}

				this->prog->status[this->gpu] = c[0];
				this->prog->nextStrideBegin[this->gpu] = this->data->beginFrom + (launchWidth * (launch + 1LLU));
				std::atomic_fetch_add(&this->prog->dataWritten, 1);
			}

			//give the rest of the computer some gpu time to reduce system choppiness
			if (primaryGpu) {
				std::this_thread::sleep_for(std::chrono::milliseconds(1));
			}
		}

		cudaStatus = reduceSJ(dev_c, size);

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

//uses 32 bit multiplications to compute the highest 64 and lowest 64 bits of squaring a 64 bit number
__device__ void square64By64(uint64 multiplicand, uint64 * lo, uint64 * hi) {

	asm("{\n\t"
		".reg .u64          m0, m1, m2;\n\t"
		".reg .u32          t0, t1, t2, t3, v0, v1, c0;\n\t"
		"mov.b64           {v0, v1}, %2;\n\t" //splits a into hi and lo 32 bit words
		"mul.wide.u32       m0, v0, v0;    \n\t" //m0 = alo*alo
		"mul.wide.u32       m1, v0, v1;    \n\t" //m1 = alo*ahi
		"mul.wide.u32       m2, v1, v1;    \n\t" //m2 = ahi*ahi
		"mov.b64           {t0, t1}, m0;\n\t"
		"mov.b64           {t2, t3}, m2;\n\t"
		"add.cc.u64         m1, m1, m1;\n\t" //because (ahi + alo)^2 = ahi^2 + 2*alo*ahi + alo^2, we must double m1
		"addc.u32           c0,  0,  0;\n\t" //preserve carryout
		"mov.b64           {v0, v1}, m1;\n\t"
		"add.cc.u32         t1, t1, v0;\n\t"
		"addc.cc.u32        t2, t2, v1;\n\t"
		"addc.u32           t3, t3, c0;\n\t"
		"mov.b64            %0, {t0, t1};\n\t" //concatenates t0 and t1 into 1 64 bit word
		"mov.b64            %1, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(*lo), "=l"(*hi)
		: "l"(multiplicand));
}

__device__ void subtractModIfMoreThanMod(uint64 & value, const uint64 & mod) {
	asm("{\n\t"
		".reg .u64        t0;\n\t"
		"sub.u64          t0, %1, %2;\n\t"
		"min.u64          %0, t0, %1;\n\t"
		"}"
		: "=l"(value)
		: "l"(value), "l"(mod));
}

//using R=2^32, performs the 2-word montgomery reduction on the number represented by hi and lo
__device__ void montgomeryAddAndShift32Bit(uint64 & hi, uint64 & lo, const uint64 & mod, const uint32 & mprime) {
	//a : multiplicand
	//b : multiplier
	//_lo : low 32 bits of result
	//_hi : high 32 bits of result
	asm("{\n\t"
		".reg .u32          t0, t1, t2, t3, z0, m0, m1;\n\t"
		"mov.b64           {m0, m1}, %3;\n\t" //splits mod into m0 and m1
		"mov.b64           {t0, t1}, %1;\n\t" //splits lo into hi and lo 32 bit words
		"mov.b64           {t2, t3}, %2;\n\t" //splits hi into hi and lo 32 bit words
		"mul.lo.u32         z0, %4, t0;\n\t" //z0 =  lo(mprime*lolo)
		"mad.lo.cc.u32      t0, z0, m0, t0;\n\t" //t0 = lolo + lo(t0 * z0)
		"madc.hi.cc.u32     t1, z0, m0, t1;\n\t"
		"addc.cc.u32        t2,  0, t2;\n\t"
		"addc.u32           t3,  0, t3;\n\t"
		"mad.lo.cc.u32      t1, z0, m1, t1;\n\t"
		"madc.hi.cc.u32     t2, z0, m1, t2;\n\t"
		"addc.u32           t3,  0, t3;\n\t"
		"mul.lo.u32         z0, %4, t1;\n\t" //z0 =  lo(mprime*t1)
		"mad.lo.cc.u32      t1, z0, m0, t1;\n\t"
		"madc.hi.cc.u32     t2, z0, m0, t2;\n\t"
		"addc.u32           t3,  0, t3;\n\t"
		"mad.lo.cc.u32      t2, z0, m1, t2;\n\t" //t0 = lolo + lo(t0 * z0)
		"madc.hi.u32        t3, z0, m1, t3;\n\t"
		"mov.b64            %0, {t2, t3};\n\t" //concatenates t2 and t3 into 1 64 bit word
		"}"
		: "=l"(hi)
		: "l"(lo), "l"(hi), "l"(mod), "r"(mprime));
}

__device__ void add128Bit(uint64 & addendHi, uint64 & addendLo, uint64 augendHi, uint64 augendLo) {
	asm("{\n\t"
		"add.cc.u64         %1, %3, %5;\n\t"
		"addc.u64           %0, %2, %4;\n\t"
		"}"
		: "=l"(addendHi), "=l"(addendLo)
		: "l"(addendHi), "l"(addendLo), "l"(augendHi), "l"(augendLo));
}

__device__ void sub128Bit(uint64 & addendHi, uint64 & addendLo, uint64 augendHi, uint64 augendLo) {
	asm("{\n\t"
		"sub.cc.u64         %1, %3, %5;\n\t"
		"subc.u64           %0, %2, %4;\n\t"
		"}"
		: "=l"(addendHi), "=l"(addendLo)
		: "l"(addendHi), "l"(addendLo), "l"(augendHi), "l"(augendLo));
}

//finds output such that (n * output) % 2^64 = -1
//found this approach used here: http://plouffe.fr/simon/1-s2.0-S0167819118300334-main.pdf
//an explanation of the approach: http://marc-b-reynolds.github.io/math/2017/09/18/ModInverse.html
//saves from 15-25% of the total computation time over xbinGCD method (on the lower side of that for larger digit computations)
__device__ void modInverseNewtonsMethod(uint64 n, uint64 & output) {
	output = (n * 3LLU) ^ 2LLU;

	for (int i = 0; i < 4; i++) {
		output = output * (2 - (n * output));
	}

	//we have (n * output) % 2^64 = 1, so we need to invert it
	output = -output;
}

//montgomery multiplication routine identical to above except for only being used when abar and bbar are known in advance to be the same
//uses a faster multiplication routine for squaring than is possible while not squaring
__device__ void montgomerySquare(uint64 abar, uint64 mod, uint32 mprime, uint64 & output) {

	uint64 tlo = 0;// , tm = 0;

	square64By64(abar, &tlo, &output);
	/*output += !!tlo;
	multiply64By64LoOnly(tlo, mprime, &tm);
	multiply64By64PlusHi(tm, mod, &output);*/


	montgomeryAddAndShift32Bit(output, tlo, mod, mprime);

	subtractModIfMoreThanMod(output, mod);
}

__device__ void fixedPointDivisionExact(const uint64 & mod, const uint64 & r, const uint64 & mPrime, uint64 * result) {
	if (!r) return;

	uint64 q0 = (-r)*mPrime;
	uint64 q1 = -(1LLU) - __umul64hi(mod, q0);
	q1 *= mPrime;

	add128Bit(result[1], result[0], q1, q0);
}

__device__ void fixedPointDivisionExactWithShift(const uint64 & mod, const uint64 & r, const uint64 & mPrime, uint64 * result, int shift, int negative) {
	if (!r) return;

	uint64 q0 = (-r)*mPrime;
	uint64 q1 = -(1LLU) - __umul64hi(mod, q0);
	q1 *= mPrime;

	q0 >>= shift;
	if(shift <= 64) q0 = q0 + (q1 << (64 - shift));
	else q0 = q0 + (q1 >> (shift - 64));
	q1 >>= shift;

	if (!negative) {
		result[0] += q0;
		result[1] += q1;
		if (result[0] < q0) result[1]++;
	}
	else {
		uint64 check = result[0];
		result[0] -= q0;
		result[1] -= q1;
		if (result[0] > check) result[1]--;
	}

}

//using left-to-right binary exponentiation
//the position of the highest bit in exponent is passed into the function as a parameter (it is more efficient to find it outside)
//uses montgomery multiplication to reduce difficulty of modular multiplication (runs in 55% of runtime of non-montgomery modular multiplication)
//montgomery multiplication suggested by njuffa
//adds the 128 bit number representing ((2^exp)%mod)/mod to result
__device__ __noinline__ void modExpLeftToRight(uint64 exp, const uint64 & mod, uint64 * result, const int & negative, uint64 subtract, uint64 expMask) {

	uint64 output = 1;
	uint64 mPrime;

	modInverseNewtonsMethod(mod, mPrime);

	uint32 mPrime32 = mPrime;

	exp = exp - subtract;

	int shift = 0;

	if (exp < 128) {
		shift = 128 - exp;
		exp = 128;
		expMask = 128LLU;
	}

	//this makes it unnecessary to convert out of montgomery space
	exp -= 64;

	//find 2 in montgomery space
	output = int64MaxBit % mod;
	output <<= 1;
	subtractModIfMoreThanMod(output, mod);
	output <<= 1;
	subtractModIfMoreThanMod(output, mod);

	//align mask with highest set bit in exp
	while (expMask > exp) expMask >>= 1;

	//shift mask to after highest set bit
	expMask >>= 1;

	while (expMask) {

		montgomerySquare(output, mod, mPrime32, output);

		if (exp & expMask) {
			output <<= 1;
			subtractModIfMoreThanMod(output, mod);
		}

		//load next bit
		expMask >>= 1;
	}

	if (shift) {
		fixedPointDivisionExactWithShift(mod, output, -mPrime, result, shift, negative);
	}
	else {
		if (negative) output = mod - output;
		fixedPointDivisionExact(mod, output, -mPrime, result);
	}
}

//stride over all parts of summation in bbp formula where k <= startingExponent
//to compute partial sJ sums
__device__ void bbp(uint64 startingExponent, uint64 start, uint64 end, uint64 stride, sJ* output, volatile uint64* progress, int progressCheck) {
	for (uint64 k = start; k <= end; k += stride) {
		uint64 exp = startingExponent - (k*10LLU);
		//shift represents number of bits needed to shift highest set bit in exp
		//into the lowest bit
		int shiftToLittleBits = 63 - __clzll(exp);
		uint64 mask = 1LLU << shiftToLittleBits;
		uint64 mod = 4 * k + 1;
		modExpLeftToRight(exp, mod, output->s, (k & 1) ^ 1, 3, mask);
		mod += 2;//4k + 3
		modExpLeftToRight(exp, mod, output->s, (k & 1) ^ 1, 8, mask);
		mod = 10 * k + 1;
		//would need to make last parameter -2, but negative numbers are not allowed
		modExpLeftToRight(exp, mod, output->s, k & 1, 0, mask);
		mod += 2;//10k + 3
		modExpLeftToRight(exp, mod, output->s, (k & 1) ^ 1, 2, mask);
		mod += 2;//10k + 5
		modExpLeftToRight(exp, mod, output->s, (k & 1) ^ 1, 6, mask);
		mod += 2;//10k + 7
		modExpLeftToRight(exp, mod, output->s, (k & 1) ^ 1, 6, mask);
		mod += 2;//10k + 9
		modExpLeftToRight(exp, mod, output->s , k & 1, 8, mask);
		if (!progressCheck) {
			//printf("%llu\n", exp);
			//only 1 thread (with gridId 0 on GPU0) ever updates the progress
			*progress = k;
		}
	}
}

//determine from thread and block position where to begin stride
//only one of the threads per kernel (AND ONLY ON GPU0) will report progress
__global__ void bbpKernel(sJ *c, volatile uint64 *progress, uint64 startingExponent, int gpuNum, uint64 begin, uint64 end, uint64 stride)
{
	int gridId = threadIdx.x + blockDim.x * blockIdx.x;
	uint64 start = begin + gridId + blockDim.x * gridDim.x * gpuNum;
	int progressCheck = gridId + blockDim.x * gridDim.x * gpuNum;
	bbp(startingExponent, start, end, stride, c + gridId, progress, !!progressCheck);
}

//stride over current leaves of reduce tree
__global__ void reduceSJKernel(sJ *c, int offset, int stop) {
	int stride = blockDim.x * gridDim.x;
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	while (i < stop) {
		int augend = i + offset;
		sJAdd(c + i, c + augend);
		i += stride;
	}
}

//standard tree-based parallel reduce
cudaError_t reduceSJ(sJ *c, unsigned int size) {
	cudaError_t cudaStatus;
	while (size > 1) {
		int nextSize = (size + 1) >> 1;

		//size is odd
		if (size&1) reduceSJKernel<< <32, 32 >> >(c, nextSize, nextSize - 1);
		//size is even
		else reduceSJKernel<< <32, 32 >> >(c, nextSize, nextSize);

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
	if (readLines != 3) {
		std::cout << "Properties loading failed!" << std::endl;
		return 1;
	}
	return 0;
}

int main() {

	if (loadProperties()) return 1;

	cudaGetDeviceCount(&totalGpus);

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

	cudaError_t cudaStatus;

	sJ cudaResult = prog.previousCache;

	for (int i = 0; i < totalGpus; i++) {

		handles[i].join();

		cudaStatus = gpuData[i].error;
		if (cudaStatus != cudaSuccess) {
			fprintf(stderr, "cudaBbpLaunch failed on gpu%d!\n", i);
			return 1;
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

	//uint64 hexDigit = finalizeDigit(cudaResult, hexDigitPosition);

	chr::high_resolution_clock::time_point end = chr::high_resolution_clock::now();

	printf("pi at hexadecimal digit %llu is %016llX %016llX\n",
		hexDigitPosition, cudaResult.s[1], cudaResult.s[0]);

	//find time elapsed during runtime of program, and add it to recorded runtime of previous unfinished run
	printf("Computed in %.8f seconds\n", prog.previousTime + (chr::duration_cast<chr::duration<double>>(end - start)).count());

	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
	}

	return 0;
}