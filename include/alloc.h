#ifndef ALLOC_H
#define ALLOC_H

#include <stdint.h>
#include <stdlib.h>

#define ALIGN_TO 64u

struct ArenaAllocator {
    void *start_;
    uint64_t pos_;
};

typedef struct ArenaAllocator ArenaAllocator;

static inline void arena_init(ArenaAllocator *arena, uint64_t size)
{
    /* TODO: Use mmap() here as well,
     * should be safer with huge sizes. */
    arena->start_ = aligned_alloc(ALIGN_TO, size);
    arena->pos_ = 0;
}

static inline void arena_init_hugetlb(ArenaAllocator *arena, uint64_t size)
{
    /* TODO: Use mmap() with MAP_HUGETLB */
    arena_init(arena, size);
}

static inline void arena_destroy(ArenaAllocator *arena)
{
    free(arena->start_);
}

static inline uint64_t arena_pos(ArenaAllocator *arena)
{
    return arena->pos_;
}

static inline void *arena_push(ArenaAllocator *arena, uint64_t size)
{
    uint64_t pos = (arena->pos_ + ALIGN_TO) & ~(ALIGN_TO - 1u);
    arena->pos_ = pos + size;
    return ((char *) arena->start_) + pos;
}

static inline void arena_pop_to(ArenaAllocator *arena, uint64_t pos)
{
    arena->pos_  = pos;
}

#define arena_push_count(arena, type, count) \
    arena_push((arena), (count) * sizeof(type))

#define arena_enter(arena) uint64_t __arena_pos = arena_pos((arena))

#define arena_exit(arena) arena_pop_to((arena), __arena_pos)

#endif
