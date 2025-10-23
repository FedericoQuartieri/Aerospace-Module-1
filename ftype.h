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
    #define vgather(...) _mm256_i32gather_ps(__VA_ARGS__)
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
    #define vgather(...) _mm256_i32gather_pd(__VA_ARGS__)
#endif

#define VSIZE 32
#define VLEN (VSIZE / sizeof(ftype))

inline void vscatter(vftype src,
                     ftype *dst,
                     uint64_t stride)
{
    ftype __attribute__((aligned(32))) buff[VLEN];
    vstore(buff, src);
    for (int i = 0; i < VLEN; ++i) {
        dst[i * stride] = buff[i];
    }
}

#endif

