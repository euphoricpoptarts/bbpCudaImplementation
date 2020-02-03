# bbpCudaImplementation

## Overview
This project is a CUDA implementation of the Bailey-Borwein-Plouffe formula for calculating hexadecimal digits of pi beginning at a given digit.  
This project currently uses an alternate version of the formula discovered by Fabrice Bellard.

Progress is cached periodically in output files under a sub-directory of the bbpCudaImplementation directory titled 'progressCache'.  
Multiple CUDA-capable GPUs are detected automatically. All such GPUs will be used. GPUs of different performance levels may be used.  
The main thread dispatches an individual thread to control each GPU, and another thread to control progress tracking and caching.
This thread awaits the completion of the GPU threads and then tells the progress tracking thread to stop.  
The user can specify a benchmarking run in the configuration files, which can run a number of trials for a given digit of pi and for a range of block counts.

Optionally, the program may run under the control of a RESTful service. This service is still under construction.  
The rest-client in this program is built using the Boost Asio and Beast libraries.

## Algorithms and Novelty of Implementation
The algorithm in use has been adapted to closely match the one presented in Daisuke Takahashi's paper: http://plouffe.fr/simon/1-s2.0-S0167819118300334-main.pdf  

This implementation has several algorithmic improvements. The conditional subtractions in montgomery multiplications have been removed. This is consistent with the constraints outlined here: https://pdfs.semanticscholar.org/0e6a/3e8f30b63b556679f5dff2cbfdfe9523f4fa.pdf  
However, the bitshift required during left-to-right modular exponentiation loop decreases the domain of correctness to moduli < 2^60. This allows correct computation for pi up to the ~288 quadrillionth hex-digits.  
This improvement increases performance by about 20-25%. The domain of correctness can be extended to moduli < 2^62 by uncommenting the conditional subtraction after the previously mentioned bitshift.

The realization that squaring by montgomery multiplication can be used for inputs Y congruent to (X % M) but which satisfy (Y^2) / 2^64 < M (and if multiple sequential squarings are performed this bound can widen),
leads to further optimizations applicable to sequences of moduli.  

In detail, given some N and two moduli M1 and M2 such that M2 < M1, we calculate (N % M1) and (N / M1).  
We can calculate (N % M2) is congruent to (N % M1) + (M1 - M2)*(N / M1). Assuming that N < M2*M1/(M1 - M2), the result will be less than 2*M2.
This is acceptable because we do not need (N % M2) precisely, but a number congruent to it.  
In order to amortize the expensive operations (N % M1) and (N / M1), we will apply this to a sequence of numbers M1 > M2 > M3 ... MX.  
If the bound is satisfied for the pair M1 and MX, than it is also satisfied for any pair MI and MJ.  
In this program, note that (M1 - M2) = (M2 - M3) = (M3 - M4) ...  
We calculate (N % M1) and (M1 - M2)*(N / M1) one time, allowing us to calculate every successive (N % MI) at the cost of one addition operation.  

To calculate 2 in Montgomery space for a sequence of moduli, N shall be 2^65. This increases performance by about 6-8%.  

This can also be extended to precompute the first 4-5 bits of the exponents for a sequence of moduli.  
This is subject to condition that the most significant 4-5 bits of the exponents corresponding to each moduli are the same for the entire sequence.  
Let X be the value of these most significant bits, N shall be 2^(64 + 2^X). This increases performace by about another 5%.

## Configuration
application.properties specifies the following:  
strideMultiplier: number of sum terms computed by each thread (64 or 128 are good values)  
blockCount: blockCount  
primaryGpu: 1 for a delay between kernel launches, 0 for no delay  
controlType: 0 to input which digit of pi to calculate manually or use command line options; 1 to cede control to restful service; 2 to run benchmark  
benchmarkTrials: number of trials for each blockCount in benchmark  
benchmarkTarget: digit of pi to use for benchmark  
benchmarkStartingBlockCount: start of benchmark range (inclusive)  
benchmarkBlockCountIncrement: amount to increment block count  
benchmarkTotalIncrements: number of times to increment blockCount and rerun benchmark

## Digits Calculated and Times
Note that only the first ~25-27 digits of each are correct.

10 Quadrillionth Hex-digit: 9077 E016 4B9C 613F D6C7 F170 CAE7 263E  
Note: This computation was performed in 40 segments using 2 different machines.  
The listed time reflects the cumulative time between both machines.  
Time: 1500081.034 seconds  
Hardware: (Rig 1) 2x RTX 2080 Ti FTW3 (Factory Clocks)  
(Rig 2) 4x RTX 2080 Ti Black (Factory Clocks)

1 Quadrillionth Hex-digit: 8353 CB3F 7F0C 9ACC FA9A A215 F309 DCF2  
Note: The final two hex-digits of this computation are not the expected value, due to hardware errors from overclocking.
These last two digits are beyond the precision limit anyways, so it doesn't really matter.  
Time: 182652.895 seconds  
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

100 Trillionth Hex-digit: 0D39 BABA 1B8F ED53 DD5F 8BDE 8266 D47C  
Time: 17042.239 seconds  
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

10 Trillionth Hex-digit: A0F9 FF37 1D17 593E 0D06 D589 2CC4 B2E9  
Time: 1590.882 seconds  
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

1 Trillionth Hex-digit: 5B44 66E8 D215 388C 4E01 4CEC 50F5 B14F  
Time: 149.168 seconds  
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

100 Billionth Hex-digit: C9C3 8187 2D27 596F 81D0 E48B 95A3 C9B5  
Time: 13.823 seconds  
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

10 Billionth Hex-digit: 921C 73C6 838F B2B6 2236 30F5 1539 32AC  
Time: 1.349 seconds  
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)