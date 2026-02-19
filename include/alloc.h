#ifndef ALLOC_H
#define ALLOC_H

#include <stdint.h>
#include <stdlib.h>
#include <sys/mman.h>

#define ALIGN_TO 64ul

struct ArenaAllocator {
    void *start_;
    uint64_t pos_;
    uint64_t size_;
};

typedef struct ArenaAllocator ArenaAllocator;

static inline void arena_init_(ArenaAllocator *arena,
                               uint64_t size,
                               int hugetlb_flags)
{
    arena->start_ = mmap(NULL, size, PROT_READ | PROT_WRITE,
                         MAP_PRIVATE | MAP_ANONYMOUS | hugetlb_flags, -1, 0);
    arena->pos_ = 0;
    arena->size_ = size;
}

static inline void arena_init(ArenaAllocator *arena, uint64_t size)
{
    arena_init_(arena, size, 0);
}

static inline void arena_init_hugetlb(ArenaAllocator *arena, uint64_t size)
{
    /* WARNING: huge pages need to be enabled at boot,
     * see https://docs.kernel.org/admin-guide/mm/hugetlbpage.html */
    arena_init_(arena, size, MAP_HUGETLB);
}

static inline void arena_destroy(ArenaAllocator *arena)
{
    munmap(arena->start_, arena->size_);
}

static inline uint64_t arena_pos(ArenaAllocator *arena)
{
    return arena->pos_;
}

static inline void *arena_push(ArenaAllocator *arena, uint64_t size)
{
    /* There's a bug here that leaves ALIGN_TO unused bytes between
     * successive allocations when pos_ sits on an alignment boundary.
     * The correct version would be:
     *
     * uint64_t pos = (arena->pos_ + (ALIGN_TO - 1ul)) & ~(ALIGN_TO - 1ul);
     *
     * When using this fix, the performance of the solver gets noticeably
     * worse though, especially when using volumes whose width is a power
     * of two, most likely due to cache conflicts. */

    uint64_t pos = (arena->pos_ + ALIGN_TO) & ~(ALIGN_TO - 1ul);
    arena->pos_ = pos + size;
    return ((char *) arena->start_) + pos;
}

static inline void *arena_push_noalign(ArenaAllocator *arena, uint64_t size)
{
    uint64_t pos = arena->pos_;
    arena->pos_ = pos + size;
    return ((char *) arena->start_) + pos;
}

static inline void arena_pop_to(ArenaAllocator *arena, uint64_t pos)
{
    arena->pos_  = pos;
}

#define arena_push_count(arena, type, count) \
    (type *) arena_push((arena), (count) * sizeof(type))

#define arena_enter(arena) uint64_t __arena_pos = arena_pos((arena))

#define arena_exit(arena) arena_pop_to((arena), __arena_pos)

#endif
