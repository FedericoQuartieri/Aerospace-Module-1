#ifndef VECTOR_NEON_H
#define VECTOR_NEON_H

#include "constants.h"
#include <arm_neon.h>

/* NEON abstraction layer for float/double precision */

#if defined(USE_FLOAT)
    #define VTYPE       float32x4_t
    #define VLEN        4
    #define VLOAD(ptr)  vld1q_f32(ptr)
    #define VSTORE(ptr, vec) vst1q_f32(ptr, vec)
    #define VSET1(x)    vdupq_n_f32(x)
    #define VADD(a,b)   vaddq_f32(a,b)
    #define VSUB(a,b)   vsubq_f32(a,b)
    #define VMUL(a,b)   vmulq_f32(a,b)
    #define VDIV(a,b)   vdivq_f32(a,b)
    #define VFMA(a,b,c) vfmaq_f32(a,b,c)  /* a + b*c */

#elif defined(USE_DOUBLE)
    #define VTYPE       float64x2_t
    #define VLEN        2
    #define VLOAD(ptr)  vld1q_f64(ptr)
    #define VSTORE(ptr, vec) vst1q_f64(ptr, vec)
    #define VSET1(x)    vdupq_n_f64(x)
    #define VADD(a,b)   vaddq_f64(a,b)
    #define VSUB(a,b)   vsubq_f64(a,b)
    #define VMUL(a,b)   vmulq_f64(a,b)
    #define VDIV(a,b)   vdivq_f64(a,b)
    #define VFMA(a,b,c) vfmaq_f64(a,b,c)  
#endif

#endif // VECTOR_NEON_H
