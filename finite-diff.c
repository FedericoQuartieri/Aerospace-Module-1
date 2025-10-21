#include "finite-diff.h"
#include "ftype.h"

static inline __attribute__((always_inline))
uint64_t rowmaj_idx(uint64_t i,
                    uint64_t j,
                    uint64_t k,
                    uint32_t height,
                    uint32_t width)
{
    return height * width * i + width * j + k;
}

static inline __attribute__((always_inline))
void compute_local_grad(const ftype *__restrict__ src,
                        uint64_t i,
                        uint64_t j,
                        uint64_t k,
                        uint32_t height,
                        uint32_t width,
                        ftype *__restrict__ dst_i,
                        ftype *__restrict__ dst_j,
                        ftype *__restrict__ dst_k)
{
    uint64_t idx = rowmaj_idx(i, j, k, height, width);
#ifdef NO_MANUAL_VECTORIZE
    ftype curr = src[idx];
    ftype next_k = src[idx + 1];
    ftype next_j = src[idx + width];
    ftype next_i = src[idx + height * width];
    dst_k[idx] = next_k - curr;
    dst_j[idx] = next_j - curr;
    dst_i[idx] = next_i - curr;
#else
    vftype curr = vload(src + idx);
    vftype next_k = vloadu(src + idx + 1);
    vftype next_j = vload(src + idx + width);
    vftype next_i = vload(src + idx + height * width);
    vstore(dst_k + idx, vsub(next_k, curr));
    vstore(dst_j + idx, vsub(next_j, curr));
    vstore(dst_i + idx, vsub(next_i, curr));
#endif
}

void compute_grad(const ftype *__restrict__ src,
                  uint32_t depth,
                  uint32_t height,
                  uint32_t width,
                  ftype *__restrict__ dst_i,
                  ftype *__restrict__ dst_j,
                  ftype *__restrict__ dst_k)
{
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
#ifdef NO_MANUAL_VECTORIZE
            for (uint32_t k = 0; k < width; ++k) {
#else
            for (uint32_t k = 0; k < width; k += VLEN) {
#endif
                compute_local_grad(
                    src, i, j, k, height, width, dst_i, dst_j, dst_k);
            }
        }
    }
}

void compute_grad_tiled(const ftype *__restrict__ src,
                        uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        uint32_t tile_depth,
                        uint32_t tile_height,
                        uint32_t tile_width,
                        ftype *__restrict__ dst_i,
                        ftype *__restrict__ dst_j,
                        ftype *__restrict__ dst_k)
{
    for (uint32_t ti = 0; ti < depth; ti += tile_depth) {
        for (uint32_t tj = 0; tj < height; tj += tile_height) {
            for (uint32_t tk = 0; tk < width; tk += tile_width) {

                for (uint32_t i = 0; i < tile_depth; ++i) {
                    for (uint32_t j = 0; j < tile_height; ++j) {
#ifdef NO_MANUAL_VECTORIZE
                        for (uint32_t k = 0; k < tile_width; ++k) {
#else
                        for (uint32_t k = 0; k < tile_width; k += VLEN) {
#endif
                            compute_local_grad(src, ti + i, tj + j, tk + k,
                                               height, width, dst_i, dst_j, dst_k);
                        }
                    }
                }
            }
        }
    }
}

void compute_grad_strided(const ftype *__restrict__ src,
                          uint32_t depth,
                          uint32_t height,
                          uint32_t width,
                          ftype *__restrict__ dst_i,
                          ftype *__restrict__ dst_j,
                          ftype *__restrict__ dst_k)
{
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; j += VLEN) {
#ifdef NO_MANUAL_VECTORIZE
            for (uint32_t k = 0; k < width; ++k) {
#else
            for (uint32_t k = 0; k < width; k += VLEN) {
#endif
                for (uint32_t jj = 0; jj < VLEN; ++jj) {
                    compute_local_grad(src, i, j + jj, k,
                                       height, width, dst_i, dst_j, dst_k);
                }
            }
        }
    }
}

