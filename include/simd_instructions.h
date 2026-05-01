#ifndef SIMD_INSTRUCTIONS_H
#define SIMD_INSTRUCTIONS_H

#include "constants.h"

/* Portable SIMD abstraction.
 * Choose architecture with defines:
 * -DUSE_NEON   -> use ARM NEON (falls back to include/neon_instructions.h)
 * -DUSE_SSE    -> use x86 SSE/AVX intrinsics
 * If no arch macro is set, the header will try to pick a sensible default
 * based on predefined compiler macros; otherwise falls back to scalar ops.
 */

#if defined(USE_NEON)
#include "neon_instructions.h"

#elif defined(USE_SSE) || defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)

#include <immintrin.h>

/* x86 SSE/AVX path */
#if defined(USE_FLOAT)
    #define VTYPE       __m128
    #define VLEN        4
    #define VLOAD(ptr)  _mm_loadu_ps((const float*)(ptr))
    #define VSTORE(ptr, vec) _mm_storeu_ps((float*)(ptr), (vec))
    #define VSET1(x)    _mm_set1_ps((float)(x))
    #define VADD(a,b)   _mm_add_ps((a),(b))
    #define VSUB(a,b)   _mm_sub_ps((a),(b))
    #define VMUL(a,b)   _mm_mul_ps((a),(b))
    #define VDIV(a,b)   _mm_div_ps((a),(b))
    #if defined(__FMA__)
        #define VFMA(a,b,c) _mm_fmadd_ps((b),(c),(a)) /* a + b*c */
    #else
        #define VFMA(a,b,c) _mm_add_ps((a), _mm_mul_ps((b),(c)))
    #endif

#elif defined(USE_DOUBLE)
    #define VTYPE       __m128d
    #define VLEN        2
    #define VLOAD(ptr)  _mm_loadu_pd((const double*)(ptr))
    #define VSTORE(ptr, vec) _mm_storeu_pd((double*)(ptr), (vec))
    #define VSET1(x)    _mm_set1_pd((double)(x))
    #define VADD(a,b)   _mm_add_pd((a),(b))
    #define VSUB(a,b)   _mm_sub_pd((a),(b))
    #define VMUL(a,b)   _mm_mul_pd((a),(b))
    #define VDIV(a,b)   _mm_div_pd((a),(b))
    #if defined(__FMA__)
        #define VFMA(a,b,c) _mm_fmadd_pd((b),(c),(a)) /* a + b*c */
    #else
        #define VFMA(a,b,c) _mm_add_pd((a), _mm_mul_pd((b),(c)))
    #endif

#else
    /* default to float scalar if precision not specified */
    #define VTYPE       float
    #define VLEN        1
    #define VLOAD(ptr)  (*(const float*)(ptr))
    #    define VSTORE(ptr, vec) (*(float*)(ptr) = (vec))
    #    define VSET1(x)    ((float)(x))
    #    define VADD(a,b)   ((a)+(b))
    #    define VSUB(a,b)   ((a)-(b))
    #    define VMUL(a,b)   ((a)*(b))
    #    define VDIV(a,b)   ((a)/(b))
    #    define VFMA(a,b,c) ((a) + (b)*(c))
#endif /* float/double selection for x86 */

#else
/* Fallback: if we couldn't detect either neon or x86, try neon header then scalar fallback */
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include "neon_instructions.h"
#else
/* Scalar fallback with precision awareness */
#if defined(USE_FLOAT)
    #define VTYPE       float
    #define VLEN        1
    #define VLOAD(ptr)  (*(const float*)(ptr))
    #    define VSTORE(ptr, vec) (*(float*)(ptr) = (vec))
    #    define VSET1(x)    ((float)(x))
    #    define VADD(a,b)   ((a)+(b))
    #    define VSUB(a,b)   ((a)-(b))
    #    define VMUL(a,b)   ((a)*(b))
    #    define VDIV(a,b)   ((a)/(b))
    #    define VFMA(a,b,c) ((a) + (b)*(c))
#elif defined(USE_DOUBLE)
    #define VTYPE       double
    #    define VLEN        1
    #    define VLOAD(ptr)  (*(const double*)(ptr))
    #    define VSTORE(ptr, vec) (*(double*)(ptr) = (vec))
    #    define VSET1(x)    ((double)(x))
    #    define VADD(a,b)   ((a)+(b))
    #    define VSUB(a,b)   ((a)-(b))
    #    define VMUL(a,b)   ((a)*(b))
    #    define VDIV(a,b)   ((a)/(b))
    #    define VFMA(a,b,c) ((a) + (b)*(c))
#else
    #error "No precision selected: define USE_FLOAT or USE_DOUBLE"
#endif
#endif
#endif /* architecture selection */

#endif // SIMD_INSTRUCTIONS_H
