#ifndef TIMEIT_H
#define TIMEIT_H

#include <stdio.h>
#include <string.h>
#include <time.h>

#define TIMEITN(func_call, avg_iter)                              \
do {                                                              \
    long elapsed_ns_avg = 0;                                      \
    /* func_call;*/ /* Warmup run. */                             \
    struct timespec start, stop;                                  \
    clock_gettime(CLOCK_MONOTONIC, &start);                       \
    for (int i = 0; i < avg_iter; ++i) {                          \
        func_call;                                                \
    }                                                             \
    clock_gettime(CLOCK_MONOTONIC, &stop);                        \
    elapsed_ns_avg = (stop.tv_sec - start.tv_sec) * 1e9 +         \
                     (stop.tv_nsec - start.tv_nsec);              \
    char *func_name = #func_call;                                 \
    char func_name_buff[32] = {'\0'};                             \
    strncpy(func_name_buff, func_name, 31);                       \
    char *ptr = strstr(func_name_buff, "(");                      \
    memset(ptr, ' ', 32 - (ptr - func_name_buff));                \
    func_name_buff[31] = '\0';                                    \
    printf("%s %6.2f ms [%d runs avg]\n",                         \
           func_name_buff,                                        \
           elapsed_ns_avg / (1e6 * avg_iter), avg_iter);          \
} while (0)

#define TIMEIT(func_call) TIMEITN(func_call, 10)

#endif
