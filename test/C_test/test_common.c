#include "test_common.h"

DTYPE compute_total_l2_error(ErrorNorms err_x, ErrorNorms err_y, ErrorNorms err_z) {
    return sqrt(pow(err_x.L2, 2) + pow(err_y.L2, 2) + pow(err_z.L2, 2));
}

ErrorNorms compute_error_norms(DTYPE *numerical, DTYPE *exact, size_t size) {
    ErrorNorms err = {0.0, 0.0, 0.0};
    
    // Volume element dV
    DTYPE dV = DX * DY * DZ;

    for (size_t i = 0; i < size; i++) {
        DTYPE diff = fabs(numerical[i] - exact[i]);
        err.L1 += diff;
        err.L2 += diff * diff;
        if (diff > err.Linf) {
            err.Linf = diff;
        }
    }
    
    // Scale L1 and L2 by the physical volume elements
    err.L1 *= dV;
    err.L2 = sqrt(err.L2 * dV);
    
    return err;
}


void compute_velocity_error(VelocityField *numerical, VelocityField *exact,
                           ErrorNorms *err_x, ErrorNorms *err_y, ErrorNorms *err_z) {
    size_t n = WIDTH * HEIGHT * DEPTH;
    *err_x = compute_error_norms(numerical->v_x, exact->v_x, n);
    *err_y = compute_error_norms(numerical->v_y, exact->v_y, n);
    *err_z = compute_error_norms(numerical->v_z, exact->v_z, n);
}

ErrorNorms compute_pressure_error(Pressure *numerical, Pressure *exact) {
    size_t n = WIDTH * HEIGHT * DEPTH;
    return compute_error_norms(numerical->p, exact->p, n);
}

void fill_exact_velocity(VelocityField *v, ExactSolution *exact, DTYPE t) {
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                DTYPE x = i * DX;
                DTYPE y = j * DY;
                DTYPE z = k * DZ;
                DTYPE vel_x = x + DX/2;
                DTYPE vel_y = y + DY/2;
                DTYPE vel_z = z + DZ/2;
                
                v->v_x[idx] = exact->velocity(vel_x, y, z, t, 0);
                v->v_y[idx] = exact->velocity(x, vel_y, z, t, 1);
                v->v_z[idx] = exact->velocity(x, y, vel_z, t, 2);
            }
        }
    }
}

void fill_exact_pressure(Pressure *p, ExactSolution *exact, DTYPE t) {
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                DTYPE x = i * DX;
                DTYPE y = j * DY;
                DTYPE z = k * DZ;
                
                p->p[idx] = exact->pressure(x, y, z, t);
            }
        }
    }
}

DTYPE compute_convergence_rate(DTYPE error_coarse, DTYPE error_fine,
                               DTYPE h_coarse, DTYPE h_fine) {
    if (error_fine < 1e-15 || error_coarse < 1e-15) return 0.0;
    return log(error_coarse / error_fine) / log(h_coarse / h_fine);
}

void print_test_result(TestResult *result, const char *test_name) {
    printf("\n========== %s ==========\n", test_name);
    printf("Grid size: %d, dx = %e,\ndt = %.1e, total time = %.1e", result->grid_size, result->dx, DT, TOTAL_TIME);
    printf("\nVelocity errors:\n");
    printf("  v_x: L1=%e, L2=%e, Linf=%e\n", 
           result->velocity_err[0].L1, result->velocity_err[0].L2, result->velocity_err[0].Linf);
    printf("  v_y: L1=%e, L2=%e, Linf=%e\n",
           result->velocity_err[1].L1, result->velocity_err[1].L2, result->velocity_err[1].Linf);
    printf("  v_z: L1=%e, L2=%e, Linf=%e\n",
           result->velocity_err[2].L1, result->velocity_err[2].L2, result->velocity_err[2].Linf);
    printf("\nPressure error:\n");
    printf("  p:   L1=%e, L2=%e, Linf=%e\n",
           result->pressure_err.L1, result->pressure_err.L2, result->pressure_err.Linf);
    if (result->convergence_rate > 0) {
        printf("\nConvergence rate: %.2f\n", result->convergence_rate);
    }
    printf("=====================================\n");
}