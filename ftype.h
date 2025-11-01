#ifndef FTYPE_H
#define FTYPE_H

#include <immintrin.h>
#include <stdint.h>

#ifdef FLOAT
    typedef float ftype;
    typedef __m256 vftype;
    typedef __m256i vitype;
    #define vload(...) _mm256_load_ps(__VA_ARGS__)
    #define vloadu(...) _mm256_loadu_ps(__VA_ARGS__)
    #define vstore(...) _mm256_store_ps(__VA_ARGS__)
    #define vadd(...) _mm256_add_ps(__VA_ARGS__)
    #define vsub(...) _mm256_sub_ps(__VA_ARGS__)
    #define vmul(...) _mm256_mul_ps(__VA_ARGS__)
    #define vdiv(...) _mm256_div_ps(__VA_ARGS__)
    #define vbroadcast(...) _mm256_set1_ps(__VA_ARGS__)
    #define vfmadd(...) _mm256_fmadd_ps(__VA_ARGS__)
    #define vxor(...) _mm256_xor_ps(__VA_ARGS__)
#else
    typedef double ftype;
    typedef __m256d vftype;
    typedef __m128i vitype;
    #define vload(...) _mm256_load_pd(__VA_ARGS__)
    #define vloadu(...) _mm256_loadu_pd(__VA_ARGS__)
    #define vstore(...) _mm256_store_pd(__VA_ARGS__)
    #define vadd(...) _mm256_add_pd(__VA_ARGS__)
    #define vsub(...) _mm256_sub_pd(__VA_ARGS__)
    #define vmul(...) _mm256_mul_pd(__VA_ARGS__)
    #define vdiv(...) _mm256_div_pd(__VA_ARGS__)
    #define vbroadcast(...) _mm256_set1_pd(__VA_ARGS__)
    #define vfmadd(...) _mm256_fmadd_pd(__VA_ARGS__)
    #define vxor(...) _mm256_xor_pd(__VA_ARGS__)
#endif

#define VSIZE 32
#define VLEN (VSIZE / sizeof(ftype))

inline __attribute__((always_inline)) void vscatter(vftype src,
                                                    ftype *dst,
                                                    uint64_t stride)
{
    ftype __attribute__((aligned(32))) buff[VLEN];
    vstore(buff, src);
    for (int i = 0; i < VLEN; ++i) {
        dst[i * stride] = buff[i];
    }
}

#ifdef FLOAT
inline __attribute__((always_inline))
void vtranspose(__m256 *r0, __m256 *r1, __m256 *r2, __m256 *r3,
                __m256 *r4, __m256 *r5, __m256 *r6, __m256 *r7)
{
    /* TODO: you could try using blend or insertf128. */

    __m256 t0 = _mm256_unpacklo_ps(*r0, *r1);
    __m256 t1 = _mm256_unpackhi_ps(*r0, *r1);
    __m256 t2 = _mm256_unpacklo_ps(*r2, *r3);
    __m256 t3 = _mm256_unpackhi_ps(*r2, *r3);
    __m256 t4 = _mm256_unpacklo_ps(*r4, *r5);
    __m256 t5 = _mm256_unpackhi_ps(*r4, *r5);
    __m256 t6 = _mm256_unpacklo_ps(*r6, *r7);
    __m256 t7 = _mm256_unpackhi_ps(*r6, *r7);

    __m256 p0 = _mm256_permute2f128_ps(t0, t4, 0x20);
    __m256 p1 = _mm256_permute2f128_ps(t2, t6, 0x20);
    __m256 p2 = _mm256_permute2f128_ps(t1, t5, 0x20);
    __m256 p3 = _mm256_permute2f128_ps(t3, t7, 0x20);
    __m256 p4 = _mm256_permute2f128_ps(t0, t4, 0x31);
    __m256 p5 = _mm256_permute2f128_ps(t2, t6, 0x31);
    __m256 p6 = _mm256_permute2f128_ps(t1, t5, 0x31);
    __m256 p7 = _mm256_permute2f128_ps(t3, t7, 0x31);

    *r0 = _mm256_shuffle_ps(p0, p1, 0x44);
    *r1 = _mm256_shuffle_ps(p0, p1, 0xee);
    *r2 = _mm256_shuffle_ps(p2, p3, 0x44);
    *r3 = _mm256_shuffle_ps(p2, p3, 0xee);
    *r4 = _mm256_shuffle_ps(p4, p5, 0x44);
    *r5 = _mm256_shuffle_ps(p4, p5, 0xee);
    *r6 = _mm256_shuffle_ps(p6, p7, 0x44);
    *r7 = _mm256_shuffle_ps(p6, p7, 0xee);
}
#else
inline __attribute__((always_inline))
void vtranspose(__m256d *r0, __m256d *r1, __m256d *r2, __m256d *r3)
{
    __m256d t0 = _mm256_unpacklo_pd(*r0, *r1);
    __m256d t1 = _mm256_unpackhi_pd(*r0, *r1);
    __m256d t2 = _mm256_unpacklo_pd(*r2, *r3);
    __m256d t3 = _mm256_unpackhi_pd(*r2, *r3);

    *r0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    *r1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    *r2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    *r3 = _mm256_permute2f128_pd(t1, t3, 0x31);
}
#endif

#endif

