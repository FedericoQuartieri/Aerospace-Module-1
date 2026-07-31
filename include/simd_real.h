#ifndef SIMD_REAL_H
#define SIMD_REAL_H

#include "types.h"

/*
 * SIMD backend for NEON and AVX2 architectures.
 *
 * SIMD_LANES is the number of Real values contained in one native vector.
 * It depends on the register width and on the scalar precision: NEON uses
 * 128-bit vectors (4 floats or 2 doubles), while AVX2 uses 256-bit vectors
 * (8 floats or 4 doubles). The intrinsic names also encode that precision,
 * so each backend defines SimdReal, SIMD_LANES and its operations together.
 * The momentum kernels use SIMD_LANES only for indexing and block sizing.
 */
#if defined(USE_SIMD) && defined(__aarch64__)

#include <arm_neon.h>

#define SIMD_AVAILABLE 1

#ifdef USE_FLOAT

#define SIMD_LANES 4
typedef float32x4_t SimdReal;

static inline SimdReal simd_set1(Real value) {
    return vdupq_n_f32(value);
}

static inline SimdReal simd_loadu(const Real *values) {
    return vld1q_f32(values);
}

static inline void simd_storeu(Real *values, SimdReal vector) {
    vst1q_f32(values, vector);
}

static inline SimdReal simd_add(SimdReal left, SimdReal right) {
    return vaddq_f32(left, right);
}

static inline SimdReal simd_sub(SimdReal left, SimdReal right) {
    return vsubq_f32(left, right);
}

static inline SimdReal simd_mul(SimdReal left, SimdReal right) {
    return vmulq_f32(left, right);
}

static inline SimdReal simd_div(SimdReal left, SimdReal right) {
    return vdivq_f32(left, right);
}

#else

#define SIMD_LANES 2
typedef float64x2_t SimdReal;

static inline SimdReal simd_set1(Real value) {
    return vdupq_n_f64(value);
}

static inline SimdReal simd_loadu(const Real *values) {
    return vld1q_f64(values);
}

static inline void simd_storeu(Real *values, SimdReal vector) {
    vst1q_f64(values, vector);
}

static inline SimdReal simd_add(SimdReal left, SimdReal right) {
    return vaddq_f64(left, right);
}

static inline SimdReal simd_sub(SimdReal left, SimdReal right) {
    return vsubq_f64(left, right);
}

static inline SimdReal simd_mul(SimdReal left, SimdReal right) {
    return vmulq_f64(left, right);
}

static inline SimdReal simd_div(SimdReal left, SimdReal right) {
    return vdivq_f64(left, right);
}

#endif

#elif defined(USE_SIMD) && defined(__AVX2__)

#include <immintrin.h>

#define SIMD_AVAILABLE 1

#ifdef USE_FLOAT

#define SIMD_LANES 8
typedef __m256 SimdReal;

static inline SimdReal simd_set1(Real value) {
    return _mm256_set1_ps(value);
}

static inline SimdReal simd_loadu(const Real *values) {
    return _mm256_loadu_ps(values);
}

static inline void simd_storeu(Real *values, SimdReal vector) {
    _mm256_storeu_ps(values, vector);
}

static inline SimdReal simd_add(SimdReal left, SimdReal right) {
    return _mm256_add_ps(left, right);
}

static inline SimdReal simd_sub(SimdReal left, SimdReal right) {
    return _mm256_sub_ps(left, right);
}

static inline SimdReal simd_mul(SimdReal left, SimdReal right) {
    return _mm256_mul_ps(left, right);
}

static inline SimdReal simd_div(SimdReal left, SimdReal right) {
    return _mm256_div_ps(left, right);
}

#else

#define SIMD_LANES 4
typedef __m256d SimdReal;

static inline SimdReal simd_set1(Real value) {
    return _mm256_set1_pd(value);
}

static inline SimdReal simd_loadu(const Real *values) {
    return _mm256_loadu_pd(values);
}

static inline void simd_storeu(Real *values, SimdReal vector) {
    _mm256_storeu_pd(values, vector);
}

static inline SimdReal simd_add(SimdReal left, SimdReal right) {
    return _mm256_add_pd(left, right);
}

static inline SimdReal simd_sub(SimdReal left, SimdReal right) {
    return _mm256_sub_pd(left, right);
}

static inline SimdReal simd_mul(SimdReal left, SimdReal right) {
    return _mm256_mul_pd(left, right);
}

static inline SimdReal simd_div(SimdReal left, SimdReal right) {
    return _mm256_div_pd(left, right);
}

#endif

#else

#define SIMD_AVAILABLE 0
#define SIMD_LANES 1

#endif


#endif
