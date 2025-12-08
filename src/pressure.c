#include "pressure.h"

#include "ftype.h"
#include "field.h"

#ifndef FLOAT

#define load_vtile(src, stride,      \
                   r1, r2, r3, r4)   \
do {                                 \
    r1 = vload(src + 0 * stride);    \
    r2 = vload(src + 1 * stride);    \
    r3 = vload(src + 2 * stride);    \
    r4 = vload(src + 3 * stride);    \
} while (0)

#define fin_diff(r0, r1, r2, r3, r4) \
do {                                 \
    r0 = r1 - r0;                    \
    r1 = r2 - r1;                    \
    r2 = r3 - r2;                    \
    r3 = r4 - r3;                    \
} while (0)

#else

#define load_vtile(src, stride,      \
                   r1, r2, r3, r4,   \
                   r5, r6, r7, r8)   \
do {                                 \
    r1 = vload(src + 0 * stride);    \
    r2 = vload(src + 1 * stride);    \
    r3 = vload(src + 2 * stride);    \
    r4 = vload(src + 3 * stride);    \
    r5 = vload(src + 4 * stride);    \
    r6 = vload(src + 5 * stride);    \
    r7 = vload(src + 6 * stride);    \
    r8 = vload(src + 7 * stride);    \
} while (0)

#define fin_diff(r0, r1, r2, r3, r4, \
                 r5, r6, r7, r8)     \
do {                                 \
    r0 = r1 - r0;                    \
    r1 = r2 - r1;                    \
    r2 = r3 - r2;                    \
    r3 = r4 - r3;                    \
    r4 = r5 - r4;                    \
    r5 = r6 - r5;                    \
    r6 = r7 - r6;                    \
    r7 = r8 - r7;                    \
} while (0)

#endif

#define vtile_ddz(src, idx, width, stride) \
    (vload(src + width * idx) - vload(src + width * idx - stride))

static inline __attribute__((always_inline))
vftype compute_div_vtile(const ftype *restrict src_x,
                         const ftype *restrict src_y,
                         const ftype *restrict src_z,
                         uint32_t height,
                         uint32_t width,
                         vftype rx0_prev,
                         int is_first_tile,
                         int is_first_row,
                         uint32_t dst_stride,
                         ftype *restrict dst)
{
    /* WARNING: Scale by -1/dxdt */

#ifndef FLOAT
    /* Loads u_y tile and computes du_y/dy. */
    vftype ry0, ry1, ry2, ry3, ry4;
    if (!is_first_row) {
        ry0 = vload(src_y - width);
    }
    load_vtile(src_y, width, ry1, ry2, ry3, ry4);

    fin_diff(ry0, ry1, ry2, ry3, ry4);
    /* ry* now contains ddy*. */

    uint64_t face_stride = height * width;
    vftype div0 = vtile_ddz(src_z, 0, width, face_stride) + ry0;
    vftype div1 = vtile_ddz(src_z, 1, width, face_stride) + ry1;
    vftype div2 = vtile_ddz(src_z, 2, width, face_stride) + ry2;
    vftype div3 = vtile_ddz(src_z, 3, width, face_stride) + ry3;

    if (is_first_row) {
        div0 = vbroadcast(0);
    }

    vtranspose(&div0, &div1, &div2, &div3);

    if (is_first_tile) {
        div0 = vbroadcast(0);
    }

    vftype rx0, rx1, rx2, rx3, rx4;
    load_vtile(src_x, width, rx1, rx2, rx3, rx4); 

    if (is_first_row) {
        rx1 = vbroadcast(0);
    }

    vtranspose(&rx1, &rx2, &rx3, &rx4);
    if (is_first_tile) {
        /* So that ddx0 goes to zero. */
        rx0 = rx1;
    } else {
        rx0 = rx0_prev;
    }

    fin_diff(rx0, rx1, rx2, rx3, rx4);

    vstore(dst + 0 * dst_stride, div0 + rx0);
    vstore(dst + 1 * dst_stride, div1 + rx1);
    vstore(dst + 2 * dst_stride, div2 + rx2);
    vstore(dst + 3 * dst_stride, div3 + rx3);

    return rx4;

#else
    /* Loads u_y tile and computes du_y/dy. */
    vftype ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7, ry8;
    if (!is_first_row) {
        ry0 = vload(src_y - width);
    }
    load_vtile(src_y, width, ry1, ry2, ry3, ry4, ry5, ry6, ry7, ry8);

    fin_diff(ry0, ry1, ry2, ry3, ry4, ry5, ry6, ry7, ry8);
    /* ry* now contains ddy*. */

    uint64_t face_stride = height * width;
    vftype div0 = vtile_ddz(src_z, 0, width, face_stride) + ry0;
    vftype div1 = vtile_ddz(src_z, 1, width, face_stride) + ry1;
    vftype div2 = vtile_ddz(src_z, 2, width, face_stride) + ry2;
    vftype div3 = vtile_ddz(src_z, 3, width, face_stride) + ry3;
    vftype div4 = vtile_ddz(src_z, 4, width, face_stride) + ry4;
    vftype div5 = vtile_ddz(src_z, 5, width, face_stride) + ry5;
    vftype div6 = vtile_ddz(src_z, 6, width, face_stride) + ry6;
    vftype div7 = vtile_ddz(src_z, 7, width, face_stride) + ry7;

    if (is_first_row) {
        div0 = vbroadcast(0);
    }

    vtranspose(&div0, &div1, &div2, &div3, &div4, &div5, &div6, &div7);

    if (is_first_tile) {
        div0 = vbroadcast(0);
    }

    vftype rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7, rx8;
    load_vtile(src_x, width, rx1, rx2, rx3, rx4, rx5, rx6, rx7, rx8); 

    if (is_first_row) {
        rx1 = vbroadcast(0);
    }

    vtranspose(&rx1, &rx2, &rx3, &rx4, &rx5, &rx6, &rx7, &rx8);
    if (is_first_tile) {
        /* So that ddx0 goes to zero. */
        rx0 = rx1;
    } else {
        rx0 = rx0_prev;
    }

    fin_diff(rx0, rx1, rx2, rx3, rx4, rx5, rx6, rx7, rx8);

    vstore(dst + 0 * dst_stride, div0 + rx0);
    vstore(dst + 1 * dst_stride, div1 + rx1);
    vstore(dst + 2 * dst_stride, div2 + rx2);
    vstore(dst + 3 * dst_stride, div3 + rx3);
    vstore(dst + 4 * dst_stride, div4 + rx4);
    vstore(dst + 5 * dst_stride, div5 + rx5);
    vstore(dst + 6 * dst_stride, div6 + rx6);
    vstore(dst + 7 * dst_stride, div7 + rx7);

    return rx8;
#endif
}

static inline __attribute__((always_inline))
void gauss_reduce_vcol(const ftype *restrict f_src,
                       ftype upper,
                       vftype *restrict f_prev,
                       ftype *restrict f_dst)
{
    vftype f = vload(f_src);
     /* WARNING: Is this a problem for precision? */
    vftype norm_coeff_inv = vbroadcast(-upper);//3 + *upper_prev;

    //*upper_prev = 1 / norm_coeff;
    //*f_prev = (f + *f_prev) / norm_coeff;
    *f_prev = (f + *f_prev) * norm_coeff_inv;

    //vstore(upper, *upper_prev);
    vstore(f_dst, *f_prev);
}

static inline __attribute__((always_inline))
void backward_sub_vcol(const ftype *restrict f,
                       ftype upper,
                       vftype *restrict p_prev,
                       ftype *restrict p)
{
    vftype f_ = vload(f);
    vftype upp = vbroadcast(upper);
    *p_prev = f_ - *p_prev * upp;
    vstore(p, *p_prev);
}

static inline __attribute__((always_inline))
void solve_vtiles_row(const ftype *restrict u_x,
                      const ftype *restrict u_y,
                      const ftype *restrict u_z,
                      uint32_t height,
                      uint32_t width,
                      int is_first_row,
                      ftype *restrict tmp,
                      ftype *restrict p)
{
    ftype *restrict tmp_upp = tmp;
    /* WARNING: I should skip max(width, height, depth)
     * elements when using this function with the fused version,
     * atm just skip an entire face. */
    ftype *restrict tmp_f = tmp + height * width;

    ftype __attribute__((aligned(32))) div_u_t[VLEN * VLEN];

    vftype last_u_x =
        compute_div_vtile(u_x, u_y, u_z,
                          height, width,
                          /* Not used. */
                          vbroadcast(0),
                          1, is_first_row,
                          VLEN, div_u_t);

    vftype f_prev = vload(div_u_t) / 3.0; /* Left BCs applied. */
    vstore(tmp_f, f_prev);

    for (int k = 1; k < VLEN; ++k) {
        gauss_reduce_vcol(div_u_t + VLEN * k,
                          tmp_upp[k],
                          &f_prev,
                          tmp_f + VLEN * k);
    }

    for (uint32_t tk = VLEN; tk < width; tk += VLEN) {
        last_u_x = compute_div_vtile(u_x + tk, u_y + tk, u_z + tk,
                                     height, width,
                                     last_u_x,
                                     0, is_first_row,
                                     VLEN, div_u_t);

        for (uint32_t k = 0; k < VLEN; ++k) {
            gauss_reduce_vcol(div_u_t + VLEN * k,
                              tmp_upp[tk + k],
                              &f_prev,
                              tmp_f + VLEN * (tk + k));
        }
    }

    ftype __attribute__((aligned(32))) p_t[VLEN * VLEN];

    vftype p_prev = vbroadcast(0);
    for (uint32_t tk = 0; tk < width; tk += VLEN) {
        for (uint32_t k = 0; k < VLEN; ++k) {
            backward_sub_vcol(tmp_f + VLEN * (width - 1 - (tk + k)),
                              tmp_upp[width - 1 - (tk + k)],
                              &p_prev,
                              p_t + VLEN * (VLEN - 1 - k));
        }
        transpose_vtile(p_t, VLEN, width, p + width - VLEN - tk);
    }
}

static void solve_Dxx_blocks(const ftype *restrict u_x,
                             const ftype *restrict u_y,
                             const ftype *restrict u_z,
                             uint32_t depth,
                             uint32_t height,
                             uint32_t width,
                             ftype *restrict tmp,
                             ftype *restrict p)
{
    ftype upp = -2.0 / 3; /* Left BC. */
    tmp[0] = upp;
    /* We can reduce once, each row of the first face
     * is the same system. */
    for (uint32_t k = 1; k < width - 1; ++k) {
        /* Gauss reduce tile column, rhs = 0 here. */
        upp = -1.0 / (3.0 + upp);
        /* Store only once, then broadcast later. */
        tmp[k] = upp;
    }
    tmp[width - 1] = -1.0 / (2 + tmp[width - 2]); /* Right BC. */

    /* Solve the first face, where div(u) = 0. */
    /* WARNING: There's no need to solve the first face again,
     * it's going to be same (zero) across all timesteps. */
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t tk = 0; tk < width; tk += VLEN) {
            vstore(p + width * j + tk, vbroadcast(0));
        }
    }

    /* Solve remaining faces. */
    for (uint32_t i = 1; i < depth; ++i) {
        /* Solve first tile row of the face. */
        solve_vtiles_row(u_x + height * width * i,
                         u_y + height * width * i,
                         u_z + height * width * i,
                         height, width,
                         1, tmp,
                         p + height * width * i);
        /* Solve remaining tile rows of the face. */
        for (uint32_t j = VLEN; j < height; j += VLEN) {
            uint64_t offset = height * width * i + width * j;
            solve_vtiles_row(u_x + offset,
                             u_y + offset,
                             u_z + offset,
                             height, width,
                             0, tmp,
                             p + offset);
        }
    }
}

static inline __attribute__((always_inline))
void gauss_reduce_scalar(uint32_t row_stride,
                         ftype upper,
                         ftype *restrict f)
{
    vftype f_prev = vload(f - row_stride);
    vftype norm_coeff_inv = vbroadcast(-upper);
    //vftype norm_coeff = 3 + upp_prev;

    //vstore(upper, -1.0f / norm_coeff);
    vstore(f, (vload(f) + f_prev) * norm_coeff_inv);
}

static inline __attribute__((always_inline))
void backward_sub_scalar(const ftype *restrict f,
                         ftype upper,
                         uint32_t row_stride,
                         ftype *restrict p)
{
    vftype f_ = vload(f);
    vftype p_prev = vload(p + row_stride);
    vstore(p, f_ - p_prev * vbroadcast(upper));
}

static inline __attribute__((always_inline))
void backward_sub_scalar_ip(const ftype *restrict p_prev,
                            ftype upper,
                            ftype *restrict p)
{
    vftype f_ = vload(p);
    vftype p_prev_ = vload(p_prev);
    vstore(p, f_ - p_prev_ * vbroadcast(upper));
}

static void solve_Dyy_blocks(uint32_t depth,
                             uint32_t height,
                             uint32_t width,
                             /* tmp of size height * width + height */
                             ftype *restrict tmp,
                             ftype *restrict f,
                             ftype *restrict p)
{
    ftype upp = -2.0 / 3; /* Left BC. */
    tmp[0] = upp;
    for (uint32_t j = 1; j < height - 1; ++j) {
        upp = -1.0 / (3.0 + upp);
        tmp[j] = upp;
    }
    tmp[height - 1] = -1.0 / (2 + tmp[height - 2]); /* Right BC. */

    for (uint32_t i = 0; i < depth; ++i) {
        uint64_t face_offset = height * width * i;

        /* Neumann BCs for the first row. */
        for (uint32_t k = 0; k < width; k += VLEN) {
            vstore(f + face_offset + k, vload(f + face_offset + k) / 3.0);
        }

        for (uint32_t j = 1; j < height; ++j) {
            ftype upp = tmp[j];
            for (uint32_t k = 0; k < width; k += VLEN) {
                gauss_reduce_scalar(width,
                                    upp,
                                    f + face_offset + width * j + k);
            }
        }

        /* BCs for the last row already applied. */
        for (uint32_t k = 0; k < width; k += VLEN) {
            vstore(p + face_offset + width * (height - 1) + k,
                   vload(f + face_offset + width * (height - 1) + k));
        }

        for (uint32_t j = 1; j < height; ++j) {
            ftype upp = tmp[height - 1 - j];
            for (uint32_t k = 0; k < width; k += VLEN) {
                /* TODO: Write solution directly to f. */
                backward_sub_scalar(f + face_offset +
                                        width * (height - 1 - j) + k,
                                    upp,
                                    width,
                                    p + face_offset +
                                        width * (height - 1 - j) + k);
            }
        }
    }
}

void solve_Dzz_blocks(uint32_t depth,
                      uint32_t height,
                      uint32_t width,
                      ftype *restrict tmp,
                      ftype *restrict f,
                      ftype *restrict p)
{
    ftype upp = -2.0 / 3; /* Left BC. */
    tmp[0] = upp;
    for (uint32_t i = 1; i < depth - 1; ++i) {
        upp = -1.0 / (3.0 + upp);
        /* Store only once, then broadcast later. */
        tmp[i] = upp;
    }
    tmp[depth - 1] = -1.0 / (2 + tmp[depth - 2]); /* Right BC. */

    /* Apply BCs to first face. */
    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t k = 0; k < width; k += VLEN) {
            vstore(f + width * j + k, vload(f + width * j + k) / 3.0);
        }
    }

    for (uint32_t i = 1; i < depth; ++i) {
        ftype upp = tmp[i];
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t k = 0; k < width; k += VLEN) {
                gauss_reduce_scalar(height * width,
                                    upp,
                                    f + height * width * i + width * j + k);
            }
        }
    }

    for (uint32_t j = 0; j < height; ++j) {
        for (uint32_t k = 0; k < width; k += VLEN) {
            vstore(p + height * width * (depth - 1) + width * j + k,
                   vload(f + height * width * (depth - 1) + width * j + k));
        }
    }

    /* Backward substitute. */
    for (uint32_t i = 1; i < depth; ++i) {
        ftype upp = tmp[depth - 1 - i];
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t k = 0; k < width; k += VLEN) {

                uint64_t offset = height * width * (depth - 1 - i) +
                                  width * j + k;

                backward_sub_scalar(f + offset,
                                    upp,
                                    height * width,
                                    p + offset);
            }
        }
    }
}

#define max(x, y) ((x) > (y) ? (x) : (y))

void solve_pressure_fused(uint32_t depth,
                          uint32_t height,
                          uint32_t width,
                          ftype *restrict tmp,
                          ftype *restrict u_x,
                          ftype *restrict u_y,
                          ftype *restrict u_z,
                          ftype *restrict p)
{
    uint32_t max_dim = max(width, max(height, depth));

    ftype upp = 0.0;
    for (uint32_t i = 0; i < max_dim; ++i) {
        upp = -1.0 / (3.0 + upp);
        /* Store only once, then broadcast later. */
        tmp[i] = upp;
    }

    /* Store intermediate solutions inside p?
     * perhaps use a separate tmp storage of size,
     * so you don't pay extra TLB misses. */

    /* Using p seems fine though, you use less space
     * and the number of TLB misses shouldn't be more. */

    ftype __attribute__((aligned(32))) p_t[VLEN * VLEN];

    /* Solve Dxx the first tile row, where div(u) = 0. */
    for (uint32_t tk = 0; tk < width; tk += VLEN) {
        ftype p_ = tmp[width - 1];
        for (uint32_t k = 0; k < VLEN; ++k) {
            p_ = -p_ * tmp[width - 1 - (tk + k)];
            vstore(p_t + VLEN * (VLEN - 1 - k), vbroadcast(p_));
        }
        transpose_vtile(p_t, VLEN, width, p + width - VLEN - tk);
    }
    /* Advance Dyy reduction for the first tile row. */
    /* TODO: apply_left_bc_hom_neumann(); */
    for (uint32_t j = 1; j < VLEN; ++j) {
        ftype upp = tmp[j - 1];
        for (uint32_t k = 0; k < width; k += VLEN) {
            gauss_reduce_scalar(width,
                                upp,
                                p + width * j + k);
        }
    }

    /* Solve Dxx the remaining tile rows, where div(u) = 0. */
    for (uint32_t tj = VLEN; tj < height - VLEN; tj += VLEN) {
        for (uint32_t tk = 0; tk < width; tk += VLEN) {
            ftype p_ = tmp[width - 1];
            for (uint32_t k = 0; k < VLEN; ++k) {
                p_ = -p_ * tmp[width - 1 - (tk + k)];
                vstore(p_t + VLEN * (VLEN - 1 - k), vbroadcast(p_));
            }
            transpose_vtile(p_t, VLEN, width,
                            p + width * tj + width - VLEN - tk);
        }
        /* Advance Dyy reduction. */
        for (uint32_t j = 0; j < VLEN; ++j) {
            ftype upp = tmp[tj + j - 1];
            for (uint32_t k = 0; k < width; k += VLEN) {
                gauss_reduce_scalar(width,
                                    upp,
                                    p + width * (tj + j) + k);
            }
        }
    }

    /* Solve Dxx for the last tile row, where div(u) = 0. */
    for (uint32_t tk = 0; tk < width; tk += VLEN) {
        ftype p_ = tmp[width - 1];
        for (uint32_t k = 0; k < VLEN; ++k) {
            p_ = -p_ * tmp[width - 1 - (tk + k)];
            vstore(p_t + VLEN * (VLEN - 1 - k), vbroadcast(p_));
        }
        transpose_vtile(p_t, VLEN, width,
                        p + width * (height - VLEN) + width - VLEN - tk);
    }
    /* Advance Dyy reduction for the first tile row. */
    for (uint32_t j = 0; j < VLEN - 1; ++j) {
        ftype upp = tmp[height - VLEN + j - 1];
        for (uint32_t k = 0; k < width; k += VLEN) {
            gauss_reduce_scalar(width,
                                upp,
                                p + width * (height - VLEN + j) + k);
        }
    }
    /* TODO: apply_right_bc_hom_neumann(); */

    /* Backward substitute first face. */
    for (uint32_t j = 1; j < height; ++j) {
        ftype upp = tmp[height - 1 - j];
        for (uint32_t k = 0; k < width; k += VLEN) {
            backward_sub_scalar_ip(p + width * (height - j) + k,
                                   upp,
                                   p + width * (height - 1 - j) + k);
        }
    }

    /* TODO: apply_left_bc_hom_neumann(); for Dzz. */

    for (uint32_t i = 1; i < depth - 1; ++i) {

        uint64_t face_offset = height * width * i;

        /* Solve first tile row of the face. */
        solve_vtiles_row(u_x + face_offset,
                         u_y + face_offset,
                         u_z + face_offset,
                         height, width,
                         1, tmp,
                         p + face_offset);

        /* TODO: apply_left_bc_hom_neumann(); */

        for (uint32_t j = 1; j < VLEN; ++j) {
            ftype upp = tmp[j - 1];
            for (uint32_t k = 0; k < width; k += VLEN) {
                gauss_reduce_scalar(width,
                                    upp,
                                    p + face_offset + width * j + k);
            }
        }

        /* Solve remaining tile rows of the face. */
        for (uint32_t tj = VLEN; tj < height - VLEN; tj += VLEN) {
            solve_vtiles_row(u_x + face_offset + width * tj,
                             u_y + face_offset + width * tj,
                             u_z + face_offset + width * tj,
                             height, width,
                             0, tmp,
                             p + face_offset + width * tj);

            /* TODO: apply_left_bc_hom_neumann(); */

            /* Advance Dyy. */
            for (uint32_t j = 1; j < VLEN; ++j) {
                ftype upp = tmp[j - 1];
                for (uint32_t k = 0; k < width; k += VLEN) {
                    gauss_reduce_scalar(width,
                                        upp,
                                        p + face_offset +
                                            width * (tj + j) + k);
                }
            }
        }

        solve_vtiles_row(u_x + face_offset + width * (height - VLEN),
                         u_y + face_offset + width * (height - VLEN),
                         u_z + face_offset + width * (height - VLEN),
                         height, width,
                         0, tmp,
                         p + face_offset + width * (height - VLEN));

        /* TODO: apply_left_bc_hom_neumann(); */

        /* Advance Dyy. */
        for (uint32_t j = 1; j < VLEN; ++j) {
            ftype upp = tmp[height - VLEN + j - 1];
            for (uint32_t k = 0; k < width; k += VLEN) {
                gauss_reduce_scalar(width,
                                    upp,
                                    p + face_offset +
                                        width * (height - VLEN + j) + k);
            }
        }

        /* TODO: apply_right_bc_hom_neumann() */

        /* Backward substitute Dyy. */
        for (uint32_t j = 1; j < height; ++j) {
            ftype upp = tmp[height - 1 - j];
            for (uint32_t k = 0; k < width; k += VLEN) {
                backward_sub_scalar_ip(p + face_offset +
                                           width * (height - j) + k,
                                       upp,
                                       p + face_offset +
                                           width * (height - 1 - j) + k);
            }
        }

        /* Advance Dzz. */
    }

    /* TODO: Loop for last face, with Dzz BCs. */

    /* Backward substitute Dzz. */
    for (uint32_t i = 1; i < depth; ++i) {
        ftype upp = tmp[depth - 1 - i];
        for (uint32_t j = 0; j < height; ++j) {
            for (uint32_t k = 0; k < width; k += VLEN) {

                backward_sub_scalar_ip(p + height * width * (depth - i) +
                                           width * j + k,
                                       upp,
                                       p + height * width * (depth - 1 - i) +
                                           width * j + k);
            }
        }
    }
}

void pressure_init(field_size size, field field)
{
    uint64_t num_points = size.depth * size.height *size.width;
    memset(field, 0, num_points * sizeof(ftype));
}

void pressure_solve(const_field3 velocity,
                    field_size size,
                    field pressure,
                    field pressure_delta,
                    ArenaAllocator *arena)
{
    arena_enter(arena);

    field tmp = field_alloc(size, arena);
    field sol = field_alloc(size, arena);

    solve_Dxx_blocks(velocity.x, velocity.y, velocity.z,
                     size.depth, size.height, size.width,
                     tmp, pressure_delta);

    solve_Dyy_blocks(size.depth, size.height, size.width, tmp,
                     pressure_delta, sol);

    solve_Dzz_blocks(size.depth, size.height, size.width, tmp,
                     sol, pressure_delta);

    /* Update pressure. */
    uint64_t num_points = field_num_points(size);
    for (uint64_t i = 0; i < num_points; ++i) {
        pressure[i] += pressure_delta[i];
    }

    arena_exit(arena);
}
