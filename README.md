# bbpCudaImplementation
This project is a CUDA implementation of the Bailey-Borwein-Plouffe formula for calculating hexadecimal digits of pi beginning at a given digit.

This project currently uses an alternate version of the formula discovered by Fabrice Bellard.

The algorithm in use has been adapted to closely match the one presented in Daisuke Takahashi's paper: http://plouffe.fr/simon/1-s2.0-S0167819118300334-main.pdf  
One small improvement has been made: the elimination of the conditional subtractions in montgomery multiplications. This is consistent with the constraints outlined here: https://pdfs.semanticscholar.org/0e6a/3e8f30b63b556679f5dff2cbfdfe9523f4fa.pdf  
However, the bitshift required during left-to-right modular exponentiation loop decreases the domain of correctness to moduli < 2^60. This allows correct computation for pi up to the ~288 quadrillionth hex-digits.  
This improvement increases performance by about 20-25%. The domain of correctness can be extended to moduli < 2^62 by uncommenting the conditional subtraction after the previously mentioned bitshift.

application.properties specifies the following:  
line 1: strideMultiplier  
line 2: blockCount  
line 3: 1 for a delay between kernel launches, 0 for no delay  
line 4: 1 to benchmark a range of blockCounts, 0 to access primary pi calculation  
line 5: number of trials for each blockCount in benchmark  
line 6: digit of pi to use for benchmark  
line 7: start of benchmark range (inclusive)  
line 8: end of benchmark range (inclusive)

Progress is cached periodically in output files under a sub-directory of the bbpCudaImplementation directory titled 'progressCache'.

Multiple CUDA-capable GPUs are detected automatically. All such GPUs will be used. Multiple GPUs are assumed to be of equal capability, and this may lead to inaccuracies in time estimation if this is not true for your system.

In addition to the main thread, an individual thread is dispatched to control each GPU, and another thread controls progress tracking and caching.