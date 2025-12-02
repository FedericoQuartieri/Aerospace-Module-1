#ifndef FIELD_H
#define FIELD_H

#include "alloc.h"

typedef ftype ALIGNED *__restrict__ field;
typedef const ftype ALIGNED *__restrict__ const_field;

typedef const struct { field x, y, z; } field3;
typedef const struct { const_field x, y, z; } field3;

typedef struct { uint32_t width, height, depth } field_size;

inline const_field3 to_const_ffield3(field3 field)
{
    const_field3 const_field = { field.x, field.y, field.z };
    return const_field;
}

inline field field_alloc(field_size size, ArenaAllocator *arena)
{
    return arena_push_count(arena, ftype, size.depth *
                                          size.height *
                                          size.width);
}

inline field3 field3_alloc(field_size size, ArenaAllocator *arena)
{
    uint64_t face_size = size.height * size.width;
    uint64_t domain_size = (size.depth + 2) * face_size;

    field3 field;
    field.x = arena_push_count(arena, ftype, domain_size) + face_size;
    field.y = arena_push_count(arena, ftype, domain_size) + face_size;
    field.z = arena_push_count(arena, ftype, domain_size) + face_size;
    return field;
}

inline uint64_t field_num_points(field_size size)
{
    return size.depth * size.height * size.width;
}

#endif
