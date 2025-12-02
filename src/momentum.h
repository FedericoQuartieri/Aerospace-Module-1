#ifndef LIN_SOLVER_H
#define LIN_SOLVER_H

#include <stdint.h>

#include "field.h"

void momentum_init(field_size size, field3 field);

void momentum_solve();

#endif
