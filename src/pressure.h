#ifndef PRESSURE_H
#define PRESSURE_H

#include "field.h"
#include "alloc.h"
#include "thread-array.h"

void pressure_init(field_size size, field field);

void pressure_solve(const_field3 velocity,
                    field_size size,
                    field pressure,
                    field pressure_delta,
                    uint32_t timestep,
                    Thread *thread);

void pressure_correct_rot(const_field3 velocity,
                          const_field3 velocity_old,
                          field_size size,
                          field pressure,
                          ftype chi,
                          uint32_t timestep);

#endif
