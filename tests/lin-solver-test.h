#ifndef LIN_SOLVER_TEST_H
#define LIN_SOLVER_TEST_H

#define TOL 1e-4

static DEF_TEST(test_wD_solution,
                const ftype *__restrict__ w,
                const ftype *__restrict__ f,
                const ftype *__restrict__ u,
                uint32_t depth,
                uint32_t height,
                uint32_t width,
                uint32_t stride_i,
                uint32_t stride_j,
                uint32_t stride_k,
                ftype u0,
                ftype un,
                int is_comp_normal,
                int is_bc_dirichlet)
{
    for (uint32_t i = 0; i < depth; ++i) {
        for (uint32_t j = 0; j < height; ++j) {
            uint64_t idx = stride_i * i + stride_j * j;
            /* Check first row of the block. */
            ftype w_i = w[idx];
            ftype f_i = f[idx];
            if (is_bc_dirichlet) {
                /* Dirichlet boundary condition! */
                ASSERT_EQUALF(u0, u[idx], TOL);
            } else {
                /* Homogeneous Neumann boundary condition! */
                ASSERT_EQUALF(f_i /* -2 dx u0 */,
                              3 * u[idx] - 2 * u[idx + stride_k],
                              TOL);
            }
            //printf("%2d %2d  %f - f=%f\n", i, j, error, f_i);
            /* Check remaining rows except last one. */
            for (uint32_t k = 1; k < width - 1; ++k) {
                ftype w_i = w[idx + stride_k * k];
                ftype f_i = f[idx + stride_k * k];
                ASSERT_EQUALF(f_i /* -2 dx u0 */,
                              -w_i * u[idx + stride_k * (k - 1)] +
                              (1 + 2 * w_i) * u[idx + stride_k * k] -
                              w_i * u[idx + stride_k * (k + 1)],
                              TOL);

                //printf("%2d %2d  %f - f=%f u=%f\n", i, j, err, f_i, u[idx + stride_k * k]);
            }
            /* Check last row of the block. */
            w_i = w[idx + stride_k * (width - 1)];
            f_i = f[idx + stride_k * (width - 1)];
            /* Dirichlet boundary condition! */
            if (is_bc_dirichlet) {
                if (is_comp_normal) {
                    ASSERT_EQUALF(un, u[idx + stride_k * (width - 1)], TOL);
                } else {
                    ASSERT_EQUALF(-2 * w_i * un - f_i,
                                  (-w_i * u[idx + stride_k * (width - 2)] +
                                  (1 + 3 * w_i) *
                                  u[idx + stride_k * (width - 1)]),
                                  TOL);
                }
            } else {
                ASSERT_EQUALF(f_i /* + dx un */,
                              -1 * u[idx + stride_k * (width - 2)] +
                              2 * u[idx + stride_k * (width - 1)],
                              TOL);
            }
            //printf("%2d %2d %f\n", i, j, error);
        }
    }
}

static DEF_TEST(test_wDxx_solution,
                const ftype *__restrict__ w,
                const ftype *__restrict__ f,
                const ftype *__restrict__ u,
                ftype u0,
                ftype un,
                uint32_t depth,
                uint32_t height,
                uint32_t width,
                int is_comp_normal,
                int is_bc_dirichlet)
{
    EXPECT_SUCCESS(test_wD_solution,
                   w, f, u, depth, height, width,
                   height * width, width, 1, u0, un,
                   is_comp_normal, is_bc_dirichlet);
}

static DEF_TEST(test_wDyy_solution,
                const ftype *__restrict__ w,
                const ftype *__restrict__ f,
                const ftype *__restrict__ u,
                ftype u0,
                ftype un,
                uint32_t depth,
                uint32_t height,
                uint32_t width,
                int is_comp_normal,
                int is_bc_dirichlet)
{
    EXPECT_SUCCESS(test_wD_solution,
                   w, f, u, depth, width, height,
                   height * width, 1, width, u0, un,
                   is_comp_normal, is_bc_dirichlet);
}

static DEF_TEST(test_wDzz_solution,
                const ftype *__restrict__ w,
                const ftype *__restrict__ f,
                const ftype *__restrict__ u,
                ftype u0,
                ftype un,
                uint32_t depth,
                uint32_t height,
                uint32_t width,
                int is_comp_normal,
                int is_bc_dirichlet)
{
    EXPECT_SUCCESS(test_wD_solution,
                   w, f, u, width, height, depth,
                   1, width, height * width, u0, un,
                   is_comp_normal, is_bc_dirichlet);
}

#endif
