#ifndef LIN_SOLVER_H
#define LIN_SOLVER_H

void gauss_reduce_tridiag_w(const ftype *__restrict__ w,
                            unsigned int size,
                            ftype *__restrict__ norm_coefs,
                            ftype *__restrict__ upper_diag);

void solve_tridiag_w(const ftype *__restrict__ w,
                     const ftype *__restrict__ upper_diag,
                     const ftype *__restrict__ norm_coefs,
                     ftype *__restrict__ b,
                     unsigned int size,
                     ftype *__restrict__ x);

#endif
