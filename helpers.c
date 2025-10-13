#include "defines.h"
#include "helpers.h"


void comp_grad(const DTYPE *__restrict__ field,
               int depth,
               int height,
               int width,
               DTYPE *__restrict__ grad_i,
               DTYPE *__restrict__ grad_j,
               DTYPE *__restrict__ grad_k)
{
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {

                size_t idx = rowmaj_idx(i, j, k, height, width);
                DTYPE value = field[idx];

                grad_i[idx] =
                    (field[rowmaj_idx(i + 1, j, k, height, width)] - value) * d_x_inverse;
                grad_j[idx] =
                    (field[rowmaj_idx(i, j + 1, k, height, width)] - value) * d_y_inverse;
                grad_k[idx] =
                    (field[rowmaj_idx(i, j, k + 1, height, width)] - value) * d_z_inverse;
            }
        }
    }
}


