#ifndef THREAD_ARRAY_H
#define THREAD_ARRAY_H

#include <pthread.h>
#include <stdint.h>

#include "alloc.h"

struct ThreadArray
{
    const uint32_t num_threads;
    pthread_attr_t thread_attr_;
    pthread_barrier_t barrier_;
    void *sh_data_;
};

typedef struct ThreadArray ThreadArray;

struct Thread {
    const uint32_t t_id;
    pthread_t pt_id;
    ThreadArray *const t_array;
    ArenaAllocator *const arena;
};

typedef struct Thread Thread;

ThreadArray *thread_array_create(uint32_t num_threads, ArenaAllocator *arena);

void thread_array_destroy(ThreadArray *t_array);

void thread_array_run(ThreadArray *t_array,
                      void *(*func)(void *),
                      ArenaAllocator *arena);

static inline void thread_run_orphan(Thread *t,
                                     void *(*func)(void *),
                                     void *args)
{
    pthread_t pt_id;
    pthread_create(&pt_id, &t->t_array->thread_attr_, func, args);
}

static inline void thread_array_set_shared_data(ThreadArray *t_array,
                                                void *data)
{
    t_array->sh_data_ = data;
}

static inline ArenaAllocator *thread_get_arena(Thread *t)
{
    return t->arena;
}

static inline void *thread_get_shared_data(Thread *t)
{
    return t->t_array->sh_data_;
}

static inline void thread_wait_on_barrier(Thread *t)
{
    pthread_barrier_wait(&t->t_array->barrier_);
}

static inline uint32_t thread_get_array_size(Thread *t)
{
    return t->t_array->num_threads;
}

#endif
