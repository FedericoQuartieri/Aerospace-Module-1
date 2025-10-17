#ifndef LIN_SOLVER_H
#define LIN_SOLVER_H

#include <stddef.h>

#include "ftype.h"

void solve_wDxx_tridiag_blocks(const ftype *__restrict__ w,
                               unsigned int depth,
                               unsigned int height,
                               unsigned int width,
                               ftype *__restrict__ tmp,
                               ftype *__restrict__ f,
                               ftype *__restrict__ u);

#endif
