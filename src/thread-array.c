#define _GNU_SOURCE

#include "thread-array.h"

#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <stdio.h>

#include "alloc.h"

/* TODO: Handle error return values. */

ThreadArray *thread_array_create(uint32_t num_threads, ArenaAllocator *arena)
{
    ThreadArray *t_array = arena_push_noalign(arena, sizeof(ThreadArray));
    ThreadArray tmp = { num_threads, {}, {}, NULL };
    memcpy(t_array, &tmp, sizeof(ThreadArray));

    pthread_attr_init(&t_array->thread_attr_);
    pthread_barrier_init(&t_array->barrier_, NULL, num_threads);

    return t_array;
}

void thread_array_destroy(ThreadArray *t_array)
{
    pthread_attr_destroy(&t_array->thread_attr_);
    pthread_barrier_destroy(&t_array->barrier_);
}

void thread_array_run(ThreadArray *t_array,
                      void *(*func)(void *),
                      ArenaAllocator *arena)
{
    arena_enter(arena);

    uint32_t num_threads = t_array->num_threads;

    Thread *threads = arena_push_noalign(
        arena, sizeof(Thread) * (num_threads - 1));

    ArenaAllocator **arena_ptrs = arena_push_noalign(
        arena, sizeof(ArenaAllocator *) * (num_threads - 1));

    /* TODO: I guess CPU affinity should be set here. */

    uint64_t arena_part_size = arena_unused(arena) / num_threads;

    for (uint32_t i = 0; i < num_threads - 1; ++i) {

        /* WARNING: Each thread should either have its
         * own partition of the arena, or a copy of the current
         * arena, provided that all threads perform the same allocations. */

        ArenaAllocator arena_part;
        arena_get_partition(arena,
                            &arena_part,
                            arena_pos(arena) + (i + 1) * arena_part_size,
                            arena_part_size);

        arena_ptrs[i] = arena_push_noalign(&arena_part, sizeof(ArenaAllocator));
        memcpy(arena_ptrs[i], &arena_part, sizeof(ArenaAllocator));

        Thread t = { i + 1, 0, t_array, arena_ptrs[i] };
        memcpy(&threads[i], &t, sizeof(Thread));
        pthread_create(&threads[i].pt_id,
                       &t_array->thread_attr_,
                       func,
                       threads + i);

        /* TODO: Figure out why pinning to cores performs weird. */
        /*
        cpu_set_t cpu_affin_mask;
        CPU_ZERO(&cpu_affin_mask);
        CPU_SET(i + 1, &cpu_affin_mask);
        pthread_setaffinity_np(threads[i].pt_id,
                               sizeof(cpu_set_t),
                               &cpu_affin_mask);
        */
    }

    ArenaAllocator arena_part;
    arena_get_partition(arena,
                        &arena_part,
                        arena_pos(arena),
                        arena_part_size);

    /* TODO: Figure out why pinning to cores performs weird. */
    /*
    cpu_set_t cpu_affin_mask;
    CPU_ZERO(&cpu_affin_mask);
    CPU_SET(0, &cpu_affin_mask);
    pthread_setaffinity_np(pthread_self(),
                           sizeof(cpu_set_t),
                           &cpu_affin_mask);
    */

    Thread t = { 0, pthread_self(), t_array, &arena_part };
    func(&t);

    for (uint32_t i = 0; i < num_threads - 1; ++i) {
        pthread_join(threads[i].pt_id, NULL);
    }

    arena_exit(arena);
}

