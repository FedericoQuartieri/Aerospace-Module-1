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

void compute_grad_transpose(const ftype *__restrict__ src,
                            uint32_t depth,
                            uint32_t height,
                            uint32_t width,
                            ftype *__restrict__ dst_i,
                            ftype *__restrict__ dst_j,
                            ftype *__restrict__ dst_k);


inline __attribute__((always_inline))
void compute_grad_at(const ftype *__restrict__ src,
                     uint64_t idx,
                     uint32_t height,
                     uint32_t width,
                     vftype *__restrict__ dst_x,
                     vftype *__restrict__ dst_y,
                     vftype *__restrict__ dst_z)
{
    vftype curr = vload(src + idx);
    vftype next_x = vloadu(src + idx + 1);
    vftype next_y = vload(src + idx + width);
    vftype next_z = vload(src + idx + height * width);
    *dst_x = vsub(next_x, curr);
    *dst_y = vsub(next_y, curr);
    *dst_z = vsub(next_z, curr);
}

inline __attribute__((always_inline))
vftype compute_Dxx_at(const ftype *__restrict__ src,
                      uint64_t idx,
                      vftype center)
{
    vftype prev = vloadu((src - 1) + idx);
    //vftype center = vload(src + idx);
    vftype next = vloadu(src + idx + 1);
    return vadd(prev, vsub(next, vadd(center, center)));
}

inline __attribute__((always_inline))
vftype compute_Dyy_at(const ftype *__restrict__ src,
                      uint64_t idx,
                      uint32_t stride,
                      vftype center)
{
    vftype prev = vload((src - stride) + idx);
    //vftype center = vload(src + idx);
    vftype next = vload(src + idx + stride);
    return vadd(prev, vsub(next, vadd(center, center)));
}

#define compute_Dzz_at(...) compute_Dyy_at(__VA_ARGS__)

#endif


