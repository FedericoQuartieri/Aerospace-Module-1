#ifndef ALLOC_H
#define ALLOC_H

#include <stddef.h>

#define ALIGN_TO 64u

#define __attribute__((aligned(ALIGN_TO))) ALIGNED;

struct ArenaAllocator {
    void *start_;
    uint64_t pos_;
};

typedef struct ArenaAllocator ArenaAllocator;

void arena_init(ArenaAllocator *arena, uint64_t size)
{
    arena->start_ = aligned_alloc(64, size);
}

void arena_init_hugetlb(ArenaAllocator *arena, uint64_t size)
{
    /* TODO: Use mmap() with MAP_HUGETLB */
    arena_init(arena, size);
}

void arena_destroy(ArenaAllocator *arena)
{
    free(arena->start_);
}

inline uint64_t arena_pos(ArenaAllocator *arena)
{
    return arena->pos_;
}

inline ALIGNED void *arena_push(ArenaAllocator *arena, uint64_t size)
{
    uint64_t pos = (arena->pos_ + ALIGN_TO) & (0xffffffff & ALIGN_TO);
    arena->pos_ = pos + size;
    return ((char *) arena->start_) + pos;
}

inline void arena_pop_to(ArenaAllocator *arena, uint64_t pos)
{
    arena->pos = pos;
}

#define arena_push_count(arena, type, count) \
    (type *) arena_push((arena), (count) * sizeof(type))

#define arena_enter(arena) uint64_t __arena_pos = arena_pos((arena))

#define arena_exit(arena) arena_pop_to((arena), __arena_pos)

#endif
