#ifndef FINITE_DIFF_H
#define FINITE_DIFF_H

#include "ftype.h"

void compute_grad(const ftype *__restrict__ src,
                  uint32_t depth,
                  uint32_t height,
                  uint32_t width,
                  ftype *__restrict__ dst_i,
                  ftype *__restrict__ dst_j,
                  ftype *__restrict__ dst_k);

void compute_grad_tiled(const ftype *__restrict__ src,
                        uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        uint32_t tile_depth,
                        uint32_t tile_height,
                        uint32_t tile_width,
                        ftype *__restrict__ dst_i,
                        ftype *__restrict__ dst_j,
                        ftype *__restrict__ dst_k);

void compute_grad_strided(const ftype *__restrict__ src,
                          uint32_t depth,
                          uint32_t height,
                          uint32_t width,
                          ftype *__restrict__ dst_i,
                          ftype *__restrict__ dst_j,
                          ftype *__restrict__ dst_k);

#endif
