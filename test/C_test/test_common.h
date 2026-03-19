#ifndef TEST_COMMON_H
#define TEST_COMMON_H #include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>
#include "../include/constants.h"
#include "../include/velocity_field.h"
#include "../include/pressure.h"
#include "../include/utils.h"

/* ==================== Error Norms ==================== */

typedef struct {
    DTYPE L1;
    DTYPE L2;
    DTYPE Linf;
} ErrorNorms;

DTYPE compute_total_l2_error(ErrorNorms err_x, ErrorNorms err_y, ErrorNorms err_z);

/* Compute error norms between numerical and exact solution */
ErrorNorms compute_error_norms(DTYPE *numerical, DTYPE *exact, size_t size);

/* Compute velocity field error norms (all components) */
void compute_velocity_error(VelocityField *numerical, VelocityField *exact, 
                           ErrorNorms *err_x, ErrorNorms *err_y, ErrorNorms *err_z);

/* Compute pressure error norms */
ErrorNorms compute_pressure_error(Pressure *numerical, Pressure *exact);

/* ==================== Exact Solution Interface ==================== */

typedef struct {
    /* Exact velocity at point (x,y,z,t) for component (0=x, 1=y, 2=z) */
    DTYPE (*velocity)(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component);
    
    /* Exact pressure at point (x,y,z,t) */
    DTYPE (*pressure)(DTYPE x, DTYPE y, DTYPE z, DTYPE t);
    
    /* Forcing term at point (x,y,z,t) for component (0=x, 1=y, 2=z) */
    DTYPE (*forcing)(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component);
    
    /* Boundary condition (same as velocity for Dirichlet) */
    DTYPE (*boundary)(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component);
} ExactSolution;

/* Fill a VelocityField with exact solution at time t */
void fill_exact_velocity(VelocityField *v, ExactSolution *exact, DTYPE t);

/* Fill a Pressure field with exact solution at time t */
void fill_exact_pressure(Pressure *p, ExactSolution *exact, DTYPE t);

/* Translate a Pressure field such that it has zero mean value. */
void translate_pressure_to_origin(Pressure *p);

/* ==================== Test Results ==================== */

typedef struct {
    int grid_size;
    DTYPE dx;
    ErrorNorms velocity_err[3];  /* x, y, z components */
    ErrorNorms pressure_err;
    DTYPE convergence_rate;
} TestResult;

/* Print test results */
void print_test_result(TestResult *result, const char *test_name);

/* Compute convergence rate from two results */
DTYPE compute_convergence_rate(DTYPE error_coarse, DTYPE error_fine, 
                               DTYPE h_coarse, DTYPE h_fine);

/* ==================== Test Runner ==================== */

#define TEST_PASS 0
#define TEST_FAIL 1

#define ASSERT_NEAR(val, expected, tol, msg) \
    do { \
        if (fabs((val) - (expected)) > (tol)) { \
            fprintf(stderr, "FAIL: %s\n  Expected: %e, Got: %e, Tol: %e\n", \
                    (msg), (expected), (val), (tol)); \
            return TEST_FAIL; \
        } \
    } while(0)

#define ASSERT_TRUE(cond, msg) \
    do { \
        if (!(cond)) { \
            fprintf(stderr, "FAIL: %s\n", (msg)); \
            return TEST_FAIL; \
        } \
    } while(0)

#endif /* TEST_COMMON_H */
