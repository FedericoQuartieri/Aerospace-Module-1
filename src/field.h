#ifndef FIELD_H
#define FIELD_H

#include <string.h>

#include "alloc.h"
#include "ftype.h"
#include "utils.h"

typedef ftype *restrict field;
typedef const ftype *restrict const_field;

typedef struct { field x, y, z; } field3;
typedef struct { const_field x, y, z; } const_field3;

typedef struct { uint32_t width, height, depth; } field_size;

static inline const_field3 to_const_field3(field3 field)
{
    const_field3 const_field = { field.x, field.y, field.z };
    return const_field;
}

static inline uint64_t field_num_points(field_size size)
{
    return size.depth * size.height * size.width;
}

static inline field field_alloc(field_size size, ArenaAllocator *arena)
{
    return arena_push_count(arena, ftype, field_num_points(size));
}

static inline field3 field3_alloc(field_size size, ArenaAllocator *arena)
{
    uint64_t num_points = field_num_points(size);

    field3 field = {
        arena_push_count(arena, ftype, num_points),
        arena_push_count(arena, ftype, num_points),
        arena_push_count(arena, ftype, num_points)
    };
    return field;
}

static inline void field_fill(field_size size, ftype val, field field)
{
    uint64_t num_points = field_num_points(size);
    const_fmemset(field, val, num_points);
}

static inline void field_rand_fill(field_size size, field field)
{
    uint64_t num_points = field_num_points(size);
    rand_fmemset(field, num_points);
}

static inline void field3_fill(field_size size, ftype val, field3 field)
{
    field_fill(size, val, field.x);
    field_fill(size, val, field.y);
    field_fill(size, val, field.z);
}

static inline void field3_rand_fill(field_size size, field3 field)
{
    field_rand_fill(size, field.x);
    field_rand_fill(size, field.y);
    field_rand_fill(size, field.z);
}

static inline void field_copy(field_size size, const_field src, field dst)
{
    uint64_t alloc_size = field_num_points(size) * sizeof(ftype);
    memcpy(dst, src, alloc_size);
}

static inline void field3_copy(field_size size, const_field3 src, field3 dst)
{
    field_copy(size, src.x, dst.x);
    field_copy(size, src.y, dst.y);
    field_copy(size, src.z, dst.z);
}

#endif
