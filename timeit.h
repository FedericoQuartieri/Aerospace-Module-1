#ifndef TIMEIT_H
#define TIMEIT_H

#include <stdio.h>
#include <time.h>

#define TIMEITN(func_call, avg_iter)                            \
do {                                                            \
    long elapsed_ns_avg = 0;                                    \
    func_call; /* Warmup run. */                                \
    struct timespec start, stop;                                \
    clock_gettime(CLOCK_MONOTONIC, &start);                     \
    for (int i = 0; i < avg_iter; ++i) {                        \
        func_call;                                              \
    }                                                           \
    clock_gettime(CLOCK_MONOTONIC, &stop);                      \
    elapsed_ns_avg = (stop.tv_sec - start.tv_sec) * 1e9 +       \
                     (stop.tv_nsec - start.tv_nsec);            \
    printf(#func_call " [" #avg_iter " runs avg] %f ms\n",      \
           elapsed_ns_avg / (1e6 * avg_iter));                  \
} while (0)

#define TIMEIT(func_call) TIMEITN(func_call, 10)

#endif
