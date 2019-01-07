# bbpCudaImplementation

## Overview
This project is a CUDA implementation of the Bailey-Borwein-Plouffe formula for calculating hexadecimal digits of pi beginning at a given digit.
This project currently uses an alternate version of the formula discovered by Fabrice Bellard.

Progress is cached periodically in output files under a sub-directory of the bbpCudaImplementation directory titled 'progressCache'.
Multiple CUDA-capable GPUs are detected automatically. All such GPUs will be used. GPUs of different performance levels may be used.
The main thread dispatches an individual thread to control each GPU, and another thread to control progress tracking and caching.
This thread awaits the completion of the GPU threads and then tells the progress tracking thread to stop.
The user can specify a benchmarking run in the configuration files, which can run a number of trials for a given digit of pi and for a range of block counts.

## Algorithms and Novelty of Implementation
The algorithm in use has been adapted to closely match the one presented in Daisuke Takahashi's paper: http://plouffe.fr/simon/1-s2.0-S0167819118300334-main.pdf  

This implementation has several algorithmic improvements. The conditional subtractions in montgomery multiplications have been removed. This is consistent with the constraints outlined here: https://pdfs.semanticscholar.org/0e6a/3e8f30b63b556679f5dff2cbfdfe9523f4fa.pdf  
However, the bitshift required during left-to-right modular exponentiation loop decreases the domain of correctness to moduli < 2^60. This allows correct computation for pi up to the ~288 quadrillionth hex-digits.  
This improvement increases performance by about 20-25%. The domain of correctness can be extended to moduli < 2^62 by uncommenting the conditional subtraction after the previously mentioned bitshift.

The realization that squaring by montgomery multiplication can be used for inputs n greater than the modulus but which satisfy (n^2) / 2^64 < mod (and if multiple sequential squarings are performed this bound can widen),
has led to an optimization of the transformations of 2 into the montgomery domain for a group of moduli.
Particularly, it is possible to perform this transformation exactly for one montgomery domain, and then reuse it for a number of other montgomery domains with moduli whose difference from the original moduli is small.
This allows us to use a number that is either equal or congruent about the modulus to 2 in the montgomery domain, for the cost of one addition per additional domain.
This might not be the best place to discuss exactly how this works, so I might make a blog post about it and link to it from here.
This improvement increases performance by about 6-8%.

## Configuration
application.properties specifies the following:
line 1: number of sum terms computed by each thread
line 2: blockCount
line 3: 1 for a delay between kernel launches, 0 for no delay
line 4: 1 to benchmark a range of blockCounts, 0 to access primary pi calculation
line 5: number of trials for each blockCount in benchmark
line 6: digit of pi to use for benchmark
line 7: start of benchmark range (inclusive)
line 8: amount to increment block count
line 9: number of times to increment blockCount and rerun benchmark

## Digits Calculated and Times
Note that only the first ~25-27 digits of each are correct.

1 Quadrillionth Hex-digit: 8353 CB3F 7F0C 9ACC FA9A A215 F309 DCEE
Time: 197302.296 seconds
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

10 Trillionth Hex-digit: A0F9 FF37 1D17 593E 0D06 D589 2CC4 B2E9
Time: 1721.840 seconds
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

1 Trillionth Hex-digit: 5B44 66E8 D215 388C 4E01 4CEC 50F5 B14F
Time: 160.475 seconds
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

100 Billionth Hex-digit: C9C3 8187 2D27 596F 81D0 E48B 95A3 C9B5
Time: 14.875 seconds
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)

10 Billionth Hex-digit: 921C 73C6 838F B2B6 2236 30F5 1539 32AC
Time: 1.481 seconds
Hardware: 2x RTX 2080 Ti FTW3 (2070 MHz)