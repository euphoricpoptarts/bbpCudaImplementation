# bbpCudaImplementation
This project is a CUDA implementation of the Bailey-Borwein-Plouffe formula for calculating hexadecimal digits of pi beginning at a given digit.

This project current uses an alternate version of the formula discovered by Fabrice Bellard.

Modular exponentiation is the primary bottleneck of the computation, and to mitigate this, left-to-right binary exponentiation is used to minimize the size of operands. In addition, Montgomery multiplication is used to reduce the number of modulus operations calculated.

This project can use any number of Nvidia GPUs the user may have in their system, and this can be controlled using a global constant. However, the program is optimized under the assumption that each GPU in use is of equal capability/computational capacity.

Progress is cached periodically in output files under a sub-directory of the bbpCudaImplementation directory which currently must be manually created by the user. This directory must be titled 'progressCache'.

In addition to the main thread, an individual thread is dispatched to control each GPU, and another thread controls progress tracking and caching.
