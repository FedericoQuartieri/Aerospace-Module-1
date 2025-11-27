#ifndef LIN_SOLVER_H
#define LIN_SOLVER_H

#include <stddef.h>
#include <stdint.h>

#include "ftype.h"

void solve_momentum_Dxx(const ftype *__restrict__ w,
                        uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ tmp,
                        ftype *__restrict__ f_x,
                        ftype *__restrict__ f_y,
                        ftype *__restrict__ f_z,
                        ftype *__restrict__ u_x,
                        ftype *__restrict__ u_y,
                        ftype *__restrict__ u_z);

void solve_momentum_Dyy(const ftype *__restrict__ w,
                        uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ tmp,
                        ftype *__restrict__ f_x,
                        ftype *__restrict__ f_y,
                        ftype *__restrict__ f_z,
                        ftype *__restrict__ u_x,
                        ftype *__restrict__ u_y,
                        ftype *__restrict__ u_z);

void solve_momentum_Dzz(const ftype *__restrict__ w,
                        uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ tmp,
                        ftype *__restrict__ f_x,
                        ftype *__restrict__ f_y,
                        ftype *__restrict__ f_z,
                        ftype *__restrict__ u_x,
                        ftype *__restrict__ u_y,
                        ftype *__restrict__ u_z);

void solve_pressure_Dxx(uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ tmp,
                        ftype *__restrict__ u_x,
                        ftype *__restrict__ u_y,
                        ftype *__restrict__ u_z,
                        ftype *__restrict__ p);

void solve_pressure_Dyy(uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ tmp,
                        ftype *__restrict__ f,
                        ftype *__restrict__ p);

void solve_pressure_Dzz(uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ tmp,
                        ftype *__restrict__ f,
                        ftype *__restrict__ p);

void solve_pressure_fused(uint32_t depth,
                          uint32_t height,
                          uint32_t width,
                          ftype *__restrict__ tmp,
                          ftype *__restrict__ u_x,
                          ftype *__restrict__ u_y,
                          ftype *__restrict__ u_z,
                          ftype *__restrict__ p);

#endif
