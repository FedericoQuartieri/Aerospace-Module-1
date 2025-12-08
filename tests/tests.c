#include <stdio.h>
#include <string.h>

#include "test.h"
#include "ftype.h"
#include "field.h"
#include "consts.h"

DEFINE_NU(1.00)
DEFINE_DT(0.01)
DEFINE_DX(0.01)

DEF_TEST(test_vtranspose)
{
#ifdef FLOAT
    float __attribute__((aligned(32))) m[64];
    for (int i = 0; i < 64; ++i) {
        m[i] = ((float) rand()) / RAND_MAX;
    }

    __m256 r0 = _mm256_load_ps(m);
    __m256 r1 = _mm256_load_ps(m + 8);
    __m256 r2 = _mm256_load_ps(m + 16);
    __m256 r3 = _mm256_load_ps(m + 24);
    __m256 r4 = _mm256_load_ps(m + 32);
    __m256 r5 = _mm256_load_ps(m + 40);
    __m256 r6 = _mm256_load_ps(m + 48);
    __m256 r7 = _mm256_load_ps(m + 56);

    vtranspose(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);

    float __attribute__((aligned(32))) t[64];

    _mm256_store_ps(t, r0);
    _mm256_store_ps(t + 8, r1);
    _mm256_store_ps(t + 16, r2);
    _mm256_store_ps(t + 24, r3);
    _mm256_store_ps(t + 32, r4);
    _mm256_store_ps(t + 40, r5);
    _mm256_store_ps(t + 48, r6);
    _mm256_store_ps(t + 56, r7);

    int status = SUCCESS;
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 8; ++j) {
            EXPECT_EQUAL(m[i * 8 + j], t[j * 8 + i]);
        }
    }

#else
    double __attribute__((aligned(32))) m[16];
    for (int i = 0; i < 16; ++i) {
        m[i] = ((double) rand()) / RAND_MAX;
    }

    __m256d r0 = _mm256_load_pd(m);
    __m256d r1 = _mm256_load_pd(m + 4);
    __m256d r2 = _mm256_load_pd(m + 8);
    __m256d r3 = _mm256_load_pd(m + 12);

    vtranspose(&r0, &r1, &r2, &r3);

    double __attribute__((aligned(32))) t[16];

    _mm256_store_pd(t, r0);
    _mm256_store_pd(t + 4, r1);
    _mm256_store_pd(t + 8, r2);
    _mm256_store_pd(t + 12, r3);

    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            EXPECT_EQUAL(m[i * 4 + j], t[j * 4 + i]);
        }
    }
#endif
}

DEF_TEST(test_momentum_Dxx_rhs,
         ArenaAllocator *arena,
         int depth, int height, int width);

DEF_TEST(test_momentum_Dxx_solver,
         ArenaAllocator *arena,
         int depth, int height, int width);

DEF_TEST(test_momentum_Dyy_solver,
         ArenaAllocator *arena,
         int depth, int height, int width);

DEF_TEST(test_momentum_Dzz_solver,
         ArenaAllocator *arena,
         int depth, int height, int width);

DEF_TEST(test_pressure_Dxx_solver,
         ArenaAllocator *arena,
         int depth, int height, int width);

DEF_TEST(test_pressure_Dyy_solver,
         ArenaAllocator *arena,
         int depth, int height, int width);

DEF_TEST(test_pressure_Dzz_solver,
         ArenaAllocator *arena,
         int depth, int height, int width);

int main(void)
{
    ArenaAllocator arena;
    arena_init(&arena, 1u << 30);

    RUN_TEST(test_vtranspose);

    RUN_TEST(test_momentum_Dxx_rhs, &arena, 16, 32, 64);
    RUN_TEST(test_momentum_Dxx_rhs, &arena, 32, 64, 128);
    RUN_TEST(test_momentum_Dxx_rhs, &arena, 128, 128, 64);

    RUN_TEST(test_momentum_Dxx_solver, &arena, 1, 16, 16);
    RUN_TEST(test_momentum_Dxx_solver, &arena, 128, 128, 64);
    RUN_TEST(test_momentum_Dxx_solver, &arena, 512, 32, 256);

    RUN_TEST(test_momentum_Dyy_solver, &arena, 32, 64, 64);
    RUN_TEST(test_momentum_Dyy_solver, &arena, 128, 128, 64);
    RUN_TEST(test_momentum_Dyy_solver, &arena, 512, 32, 256);

    RUN_TEST(test_momentum_Dzz_solver, &arena, 32, 64, 64);
    RUN_TEST(test_momentum_Dzz_solver, &arena, 128, 128, 64);
    RUN_TEST(test_momentum_Dzz_solver, &arena, 32, 32, 256);

    RUN_TEST(test_pressure_Dxx_solver, &arena, 1, 32, 32);
    RUN_TEST(test_pressure_Dxx_solver, &arena, 32, 64, 128);
    RUN_TEST(test_pressure_Dxx_solver, &arena, 128, 64, 512);

    RUN_TEST(test_pressure_Dyy_solver, &arena, 1, 32, 32);
    RUN_TEST(test_pressure_Dyy_solver, &arena, 32, 64, 128);
    RUN_TEST(test_pressure_Dyy_solver, &arena, 128, 64, 512);

    RUN_TEST(test_pressure_Dzz_solver, &arena, 64, 32, 32);
    RUN_TEST(test_pressure_Dzz_solver, &arena, 256, 32, 128);
    RUN_TEST(test_pressure_Dzz_solver, &arena, 128, 64, 64);

    arena_destroy(&arena);

    return 0;
}
