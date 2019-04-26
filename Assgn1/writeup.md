# System  Description

- Nodes: 2
  - Kind: NUMA
  - Sockets per node: 1
    - Cores per socket: 6
      - Processing units per core: 2
    - L3 cache:
      - Distribution: shared
      - Size: 15MB
    - L2 cache:
      - Distribution: per core
      - Size: 256KB
    - L1d cache:
      - Distribution: per core
      - Size: 256KB
    - L1i cache:
      - Distribution: per core
      - Size: 256KB

# Affinity

Each execution profile was ran against 10000000 elements and 100 steps.

At 12 threads per run, across the board the average time to run the program is about 12 milliseconds.

Despite the fact that each core can hold 2 threads, on the 12 core system, running with the thread count set to 24

(twice the initial amount) produced an average increase in speed by about 4 milliseconds. While true

parallelism due to hyperthreading between two threads is technically possible in a single a core, there is still

the limited bandwidth associated with both the L1, L2, and even the L3 cache.

Here is the provided source which this is tested against:

```
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
int main(int argc, char **argv){
        //Change the following 2 variables to change input parameter size
        int nlocal = atol(argv[1]);//100000000;
        int nsteps = atol(argv[2]);
        // Allocating memory
        double *in = (double*) malloc( (nlocal+2) * sizeof(double));
        double *out = (double*) malloc(nlocal * sizeof(double));
        for (int i=0; i<nlocal+2; i++)
                in[i] = 1.0;
        for (int i=0; i<nlocal; i++)
                out[i] = 0.0;
        double start_time, end_time;
        for (int step=0; step < nsteps; step++) {
                start_time = omp_get_wtime();
#pragma omp parallel for schedule(static)
                for (int i=0; i < nlocal; i++) {
                        out[i] = ( in[i]+in[i+1]+in[i+2] )/3.;
                }
#pragma omp parallel for schedule(static)
                for (int i=0; i < nlocal; i++){
                        in[i+1] = out[i];
                }
                in[0] = 0;
                in[nlocal+1] = 1;
        }
        end_time = omp_get_wtime();
        printf("%lf\n", end_time - start_time);
}
```

Note that `double *in` refers to 3x the amount of heap allocated memory as `double *out`.

Each core has its own [8-way set associative cache](https://en.wikichip.org/wiki/intel/xeon_e5/e5-2620_v4#Cache).

We know the L1 cache size is about 32k for a given core. 32 * 1024 = 32k. Divide this amount by the size of a cache line (64 bytes) multiplied
by the amount of entries for a given cache set: 8, and we have

`(32 * 1024) / (8 * 64)` = 64.


So, we have 64 separate entries which are likely to remain the cache at any given time as far as our buffer is concerned.

Because a double takes 8 bytes of space, we are left with 8 distinct elements per cache line, which implies that,

in our cache, the most we can reliably expect to have stored at a given moment is 8 * 64 = 512 contiguous elements.

There is an approximate 20,000 separate contiguous groups of 512 elements for every 10,000,000.


Bits [11:6] of the a memory address are used to represent the set of the address's corresponding cache line.


After this, which cache line the memory of the address will occupy is dependent on the tag bits of the address;

unfortunately, this is limiting, because the tag bits are physically indexed, and thus must be translated to

a corresponding physical page. Despite this, we theoretically should be able to minimize set conflicts between different

regions of memory by varying the tag bits themselves.


Bits [47:12] are used to index into the page table entries (of which there are 4 levels); each level can hold 512 entries,

with each entry representing a constant region of memory (the amount of which is dependent on the level of the page table itself).

It's likely that the cache here is being used relatively efficiently. If it is, we can expect an upper bound of 512 * 8 = 2048 elements occupying the entirety of the L1 cache.

That is, assuming we ensure the iterator and the length of the buffer are both held in registers.

Given that reads from L1 take approximately 4 nanoseconds, and because we're operating on
10,000,000 elements, we can expect an approximate 0.04 seconds to read all of the elements.

Factoring in the fact that we're dealing with virtual addresses, page mapping and unmapping, in addition to multiple processes (among other things) and the overhead associated with the scheduling of threads, we wind up with something along the lines of about 0.012378 seconds to complete the task (for the cores|spread execution profile).

The average times for each profile are listed as follows:

* cores|spread: 0.012378

  * the idea behind this profile is to spread the amount of threads we allocate throughout multiple cores. This allows for us to take advantage of multiple caches, which implies that we have potentially up to 8192 elements, assuming that, for each core, the respective matrix being iterated over has address values which map into the physical page table in such a manner that reduces conflict.

* cores|close: 0.012542
  * The average listed here is slightly slower, but the difference is essentially negligable. The "close" binding method is designed to cluster threads closer to the same core. The potential for cache thrashing due to multiple multiple threads contesting similar regions of memory is higher in this sense. That said, assuming the implementation associates thread groups contiguously, this is less of an issue, given that other threads are less likely to have conflicting memory regions.


* sockets|spread: 0.012579
  * This is interesting, because it's more or less around the same amount of time as the previous two. That said, it also makes sense, considering that the memory accesses essentially have a one to one mapping with the thread the group of indices have been accounted for; we don't have iterations which are reaching out in order to fetch data that lies 300000 words away from the current offset that's represented for a given iteration; because of this, we're less likely to have to deal with cross socket memory accesses, which incur significant latency overhead.

* sockets|close: 0.012375
 * In this profile, we're constraining the 12 threads into one socket, but whether or not the threads are spread throughout different cores (likely) is dependent on the implementation; the specification itself insinuates this [(page 605, on OMP_PLACES)](https://www.openmp.org/wp-content/uploads/OpenMP-API-Specification-5.0.pdf).











