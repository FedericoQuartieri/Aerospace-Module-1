#ifndef FTYPE_H
#define FTYPE_H

#include <immintrin.h>
#include <stdint.h>

#ifdef FLOAT
    typedef float ftype;
    typedef __m256 vftype;
#else
    typedef double ftype;
    typedef __m256d vftype;
#endif

#ifdef AUTO_VEC
    #define VLEN 1
    #define vftype ftype
    #define vadd(x, y) ((x) + (y))
    #define vsub(x, y) ((x) - (y))
    #define vmul(x, y) ((x) * (y))
    #define vdiv(x, y) ((x) / (y))
    #define vfmadd(x, y, z) ((x) * (y) + (z))
    #define vload(addr) (*(addr))
    #define vloadu(addr) vload(addr)
    #define vstore(addr, x) *(addr) = (x)
    #define vbroadcast(x) (x)
#else
    #define VLEN (32 / sizeof(ftype))
#ifdef FLOAT
    #define vload(...) _mm256_load_ps(__VA_ARGS__)
    #define vloadu(...) _mm256_loadu_ps(__VA_ARGS__)
    #define vstore(...) _mm256_store_ps(__VA_ARGS__)
    #define vntstore(...) _mm256_stream_ps(__VA_ARGS__)
    #define vadd(...) _mm256_add_ps(__VA_ARGS__)
    #define vsub(...) _mm256_sub_ps(__VA_ARGS__)
    #define vmul(...) _mm256_mul_ps(__VA_ARGS__)
    #define vdiv(...) _mm256_div_ps(__VA_ARGS__)
    #define vbroadcast(...) _mm256_set1_ps(__VA_ARGS__)
    #define vfmadd(...) _mm256_fmadd_ps(__VA_ARGS__)
    #define vxor(...) _mm256_xor_ps(__VA_ARGS__)
#else
    #define vload(...) _mm256_load_pd(__VA_ARGS__)
    #define vloadu(...) _mm256_loadu_pd(__VA_ARGS__)
    #define vstore(...) _mm256_store_pd(__VA_ARGS__)
    #define vntstore(...) _mm256_stream_pd(__VA_ARGS__)
    #define vadd(...) _mm256_add_pd(__VA_ARGS__)
    #define vsub(...) _mm256_sub_pd(__VA_ARGS__)
    #define vmul(...) _mm256_mul_pd(__VA_ARGS__)
    #define vdiv(...) _mm256_div_pd(__VA_ARGS__)
    #define vbroadcast(...) _mm256_set1_pd(__VA_ARGS__)
    #define vfmadd(...) _mm256_fmadd_pd(__VA_ARGS__)
    #define vxor(...) _mm256_xor_pd(__VA_ARGS__)
#endif
#endif

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

#ifndef AUTO_VEC
static inline __attribute__((always_inline))
void transpose_vtile(const ftype *restrict src,
                     uint32_t src_stride,
                     uint32_t dst_stride,
                     ftype *restrict dst)
{
    /* TODO: faster version if you transpose in memory using insert2f128? */
#ifdef FLOAT
    vftype r0 = vload(src + 0 * src_stride);
    vftype r1 = vload(src + 1 * src_stride);
    vftype r2 = vload(src + 2 * src_stride);
    vftype r3 = vload(src + 3 * src_stride);
    vftype r4 = vload(src + 4 * src_stride);
    vftype r5 = vload(src + 5 * src_stride);
    vftype r6 = vload(src + 6 * src_stride);
    vftype r7 = vload(src + 7 * src_stride);
    vtranspose(&r0, &r1, &r2, &r3, &r4, &r5, &r6, &r7);
    vstore(dst + 0 * dst_stride, r0);
    vstore(dst + 1 * dst_stride, r1);
    vstore(dst + 2 * dst_stride, r2);
    vstore(dst + 3 * dst_stride, r3);
    vstore(dst + 4 * dst_stride, r4);
    vstore(dst + 5 * dst_stride, r5);
    vstore(dst + 6 * dst_stride, r6);
    vstore(dst + 7 * dst_stride, r7);
#else
    vftype r0 = vload(src + 0 * src_stride);
    vftype r1 = vload(src + 1 * src_stride);
    vftype r2 = vload(src + 2 * src_stride);
    vftype r3 = vload(src + 3 * src_stride);
    vtranspose(&r0, &r1, &r2, &r3);
    vstore(dst + 0 * dst_stride, r0);
    vstore(dst + 1 * dst_stride, r1);
    vstore(dst + 2 * dst_stride, r2);
    vstore(dst + 3 * dst_stride, r3);
#endif
}
#endif

#endif


