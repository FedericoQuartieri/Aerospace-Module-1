#ifndef MOMENTUM_H
#define MOMENTUM_H

#include "field.h"
#include "alloc.h"

void momentum_init(field_size size, field3 field);

void momentum_solve(const_field porosity,
                    const_field gamma,
                    const_field pressure,
                    const_field pressure_delta,
                    field_size size,
                    field3 velocity_Dxx,
                    field3 velocity_Dyy,
                    field3 velocity_Dzz,
                    uint32_t timestep,
                    ArenaAllocator *arena);

#endif
