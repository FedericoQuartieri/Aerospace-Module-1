#ifndef CONVERGENCE_TEST_H
#define CONVERGENCE_TEST_H

static inline void estimate_convergence_order(const double *errors,
                                              const double *dxs,
                                              int num_samples,
                                              double *orders)
{
    orders[0] = 0;
    for (int i = 1; i < num_samples; ++i) {
        orders[i] = log(errors[i - 1] / errors[i]) /
                    log(dxs[i - 1] / dxs[i]);
    }
}

static inline void print_convergence_table(const double *errors,
                                           const double *dxs,
                                           const double *orders,
                                           int num_samples)
{
    printf("%.6f  %e   --\n", dxs[0], errors[0]);
    for (int i = 1; i < num_samples; ++i) {
        printf("%.6f  %e %5.2f\n", dxs[i], errors[i], orders[i]);
    }
}



#endif
