#ifndef PRESSURE_H
#define PRESSURE_H

#include "field.h"
#include "alloc.h"

void pressure_init(field_size size, field field);

void pressure_solve(const_field3 velocity,
                    field_size size,
                    field pressure,
                    field pressure_delta,
                    ArenaAllocator *arena);

#endif
