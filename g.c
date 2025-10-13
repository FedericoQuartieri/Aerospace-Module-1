#include "defines.h"
#include "g.h"
#include "helpers.h"


void g(const DTYPE *f_x,
            const DTYPE *f_y,
            const DTYPE *f_z,
            const DTYPE *eta_x,
            const DTYPE *eta_y,
            const DTYPE *eta_z,
            const DTYPE *zeta_x,
            const DTYPE *zeta_y,
            const DTYPE *zeta_z,
            const DTYPE *speed_x,
            const DTYPE *speed_y,
            const DTYPE *speed_z,
            const DTYPE *grad_x,
            const DTYPE *grad_y,
            const DTYPE *grad_z,
            DTYPE *g_x,
            DTYPE *g_y,
            DTYPE *g_z,
            const DTYPE *k_values,
            int depth,
            int height,
            int width)
{
    for (int i = 1; i < depth - 1; ++i) {
        for (int j = 1; j < height - 1; ++j) {
            for (int k = 1; k < width - 1; ++k) {

                size_t idx = rowmaj_idx(i, j, k, height, width);

                g_x[idx] = f_x[idx] - grad_x[idx] - (nu_const / (2 * k_values[idx])) * speed_x[idx] +
                             (nu_const / 2) * ((eta_x[rowmaj_idx(i + 1, j, k, height, width)] -2 * eta_x[idx] +
                                                 eta_x[rowmaj_idx(i - 1, j, k, height, width)]) * (d_x_inverse_squared) + (zeta_x[rowmaj_idx(i, j + 1, k, height, width)] -2 * zeta_x[idx] +
                                                 zeta_x[rowmaj_idx(i, j - 1, k, height, width)]) * (d_y_inverse_squared) + (speed_x[rowmaj_idx(i, j, k + 1, height, width)] -2 * speed_x[idx] +
                                                 speed_x[rowmaj_idx(i, j, k - 1, height, width)]) * (d_z_inverse_squared));
                g_y[idx] = f_y[idx] - grad_y[idx] - (nu_const / (2 * k_values[idx])) * speed_y[idx] +
                             (nu_const / 2) * ((eta_y[rowmaj_idx(i + 1, j, k, height, width)] -2 * eta_y[idx] +
                                                 eta_y[rowmaj_idx(i - 1, j, k, height, width)]) * (d_x_inverse_squared) + (zeta_y[rowmaj_idx(i, j + 1, k, height, width)] -2 * zeta_y[idx] +
                                                 zeta_y[rowmaj_idx(i, j - 1, k, height, width)]) * (d_y_inverse_squared) + (speed_y[rowmaj_idx(i, j, k + 1, height, width)] -2 * speed_y[idx] +
                                                 speed_y[rowmaj_idx(i, j, k - 1, height, width)]) * (d_z_inverse_squared));
                g_z[idx] = f_z[idx] - grad_z[idx] - (nu_const / (2 * k_values[idx])) * speed_z[idx] +
                             (nu_const / 2) * ((eta_z[rowmaj_idx(i + 1, j, k, height, width)] -2 * eta_z[idx] +
                                                 eta_z[rowmaj_idx(i - 1, j, k, height, width)]) * (d_x_inverse_squared) + (zeta_z[rowmaj_idx(i, j + 1, k, height, width)] -2 * zeta_z[idx] +
                                                 zeta_z[rowmaj_idx(i, j - 1, k, height, width)]) * (d_y_inverse_squared) + (speed_z[rowmaj_idx(i, j, k + 1, height, width)] -2 * speed_z[idx] +
                                                 speed_z[rowmaj_idx(i, j, k - 1, height, width)]) * (d_z_inverse_squared));
            }
        }
    }
}
