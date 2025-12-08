#include "ftype.h"
#include "alloc.h"
#include "field.h"

#include "test.h"
#include "lin-solver-test.h"

#include "pressure.c"

static void compute_div(const ftype *src_x,
                        const ftype *src_y,
                        const ftype *src_z,
                        uint32_t depth,
                        uint32_t height,
                        uint32_t width,
                        ftype *dst)
{
    /* Front face divergence is zero. */
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t k = 0; k < width; ++k) {
            dst[width * j + k] = 0;
        }
    }

    for (uint32_t i = 1; i < depth; ++i) {
        uint64_t face_offset = height * width * i;
        /* Top face divergence is zero. */
        for (int k = 0; k < width; ++k) {
            dst[face_offset + k] = 0;
        }

        for (uint32_t j = 1; j < height; ++j) {
            /* Left face divergence is zero. */
            dst[face_offset + width * j] = 0;
            for (uint32_t k = 1; k < width; ++k) {
                uint64_t idx = face_offset + width * j + k;
                dst[idx] = (src_x[idx] - src_x[idx - 1]) +
                           (src_y[idx] - src_y[idx - width]) +
                           (src_z[idx] - src_z[idx - height * width]);
            }
        }
    }
}

DEF_TEST(test_pressure_Dxx_solver,
         ArenaAllocator *arena,
         uint32_t depth,
         uint32_t height,
         uint32_t width)
{
    arena_enter(arena);

    field_size size = { width, height, depth };
    field gamma = field_alloc(size, arena);
    field sol = field_alloc(size, arena);
    field rhs = field_alloc(size, arena);
    field tmp = field_alloc(size, arena);
    field3 velocity = field3_alloc(size, arena);

    field_fill(size, 1.0, gamma);
    field3_rand_fill(size, velocity);

    compute_div(velocity.x, velocity.y, velocity.z,
                depth, height, width, rhs);

    solve_Dxx_blocks(velocity.x, velocity.y, velocity.z,
                     depth, height, width, tmp, sol);

    EXPECT_SUCCESS(test_wDxx_solution,
                   gamma, rhs, sol, 0.0, 0.0, depth, height, width, 1, 0);

    arena_exit(arena);
}

#define DEF_TEST_PRESSURE_D_SOLVER(axes)                                   \
DEF_TEST(test_pressure_D##axes##_solver,                                   \
         ArenaAllocator *arena,                                            \
         uint32_t depth,                                                   \
         uint32_t height,                                                  \
         uint32_t width)                                                   \
{                                                                          \
    arena_enter(arena);                                                    \
                                                                           \
    field_size size = { width, height, depth };                            \
    field gamma = field_alloc(size, arena);                                \
    field sol = field_alloc(size, arena);                                  \
    field rhs = field_alloc(size, arena);                                  \
    field rhs_ref = field_alloc(size, arena);                              \
    /* WARNING: Only max(d, h, w) tmp space would be required. */          \
    field tmp = field_alloc(size, arena);                                  \
                                                                           \
    field_fill(size, 1.0, gamma);                                          \
    field_rand_fill(size, rhs);                                            \
    field_copy(size, rhs, rhs_ref);                                        \
                                                                           \
    solve_D##axes##_blocks(depth, height, width, tmp, rhs, sol);           \
                                                                           \
    EXPECT_SUCCESS(test_wD##axes##_solution,                               \
                   gamma, rhs_ref, sol, 0.0, 0.0,                          \
                   depth, height, width, 1, 0);                            \
                                                                           \
    arena_exit(arena);                                                     \
}

DEF_TEST_PRESSURE_D_SOLVER(yy)
DEF_TEST_PRESSURE_D_SOLVER(zz)



