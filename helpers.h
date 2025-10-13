#ifndef HELPERS_H
#define HELPERS_H

#include "defines.h"

static inline __attribute__((always_inline)) size_t rowmaj_idx(size_t i,
                                                               size_t j,
                                                               size_t k,
                                                               size_t height,
                                                               size_t width)
{
    size_t face_size = width * height;
    return i * face_size + j * width + k;
}
void comp_grad(const DTYPE *__restrict__ field,
               int depth,
               int height,
               int width,
               DTYPE *__restrict__ grad_i,
               DTYPE *__restrict__ grad_j,
               DTYPE *__restrict__ grad_k);




#endif // HELPERS_H