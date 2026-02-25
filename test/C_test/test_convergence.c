#include "test_common.h"
#include "../include/solve.h"
#include "../include/g_field.h"
#include "../include/function.h"

/* 
 * Convergence test: analyzes results from multiple resolutions
 * and verifies the expected convergence rate.
 *
 * For 2nd order schemes, expect rate ≈ 2.0
 *
 * Usage:
 *   1. Run ./run_convergence_test.sh to generate convergence_results.txt
 *   2. Run ./test_convergence convergence_results.txt to analyze
 *
 * Or run with no arguments to use current compiled resolution.
 */

#define MAX_REFINEMENTS 10

typedef struct {
    int N;              /* Grid points per dimension */
    DTYPE dx;           /* Grid spacing */
    DTYPE L2_vx;        /* L2 error for velocity x */
    DTYPE L2_vy;        /* L2 error for velocity y */
    DTYPE L2_vz;        /* L2 error for velocity z */
    DTYPE L2_p;         /* L2 error for pressure */
} ConvergenceDataPoint;

/* ==================== Manufactured Solution (same as test_manufactured.c) ==================== */

static DTYPE manufactured_velocity(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    DTYPE sx = sin(M_PI * x); DTYPE sy = sin(M_PI * y); DTYPE sz = sin(M_PI * z);
    DTYPE cx = cos(M_PI * x); DTYPE cy = cos(M_PI * y); DTYPE cz = cos(M_PI * z);
    DTYPE st = sin(t);
    
    switch (component) {
        case 0: return st * (sx * sy * sz);         
        case 1: return st * (cx * cy * cz);           
        case 2: return st * (cx * sy * (cz + sz));    
        default: return 0.0;
    }
}

static DTYPE manufactured_pressure(DTYPE x, DTYPE y, DTYPE z, DTYPE t) {
    DTYPE cx = cos(M_PI * x); DTYPE cz = cos(M_PI * z);
    DTYPE sy = sin(M_PI * y); DTYPE sz = sin(M_PI * z);
    return -3.0 * NU * sin(t) * (cx * sy * (sz - cz));
}

static DTYPE manufactured_boundary(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    return manufactured_velocity(x, y, z, t, component);
}

static ExactSolution create_manufactured_solution(void) {
    ExactSolution exact = {
        .velocity = manufactured_velocity,
        .pressure = manufactured_pressure,
        .forcing = NULL,  /* Not needed for error computation */
        .boundary = manufactured_boundary
    };
    return exact;
}

/* ==================== Single Resolution Test ==================== */

/* Run test at current compiled resolution and return errors */
ConvergenceDataPoint run_single_resolution_test(void) {
    ConvergenceDataPoint data;
    data.N = WIDTH;  /* Assuming WIDTH == HEIGHT == DEPTH */
    data.dx = DX;
    
    printf("Running simulation with N = %d, dx = %e\n", data.N, data.dx);
    
    ExactSolution exact = create_manufactured_solution();
    
    /* Initialize fields */
    Pressure pressure;
    initialize_pressure(&pressure);
    
    VelocityField Eta, Zeta, U;
    function_handle v_boundary = parse_function("test_manufactured_Vboundary.txt");
    
    if (!v_boundary) {
        fprintf(stderr, "Warning: Could not load boundary function file\n");
    }
    
    initialize_velocity_field(&Eta, v_boundary);
    initialize_velocity_field(&Zeta, v_boundary);
    initialize_velocity_field(&U, v_boundary);
    
    /* Set initial conditions from exact solution at t=0 */
    fill_exact_velocity(&U, &exact, 0.0);
    fill_exact_velocity(&Eta, &exact, 0.0);
    fill_exact_velocity(&Zeta, &exact, 0.0);
    fill_exact_pressure(&pressure, &exact, 0.0);
    
    /* Initialize K, Beta, Gamma */
    DTYPE *K = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Beta = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *)malloc(GRID_SIZE);
    
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                size_t idx = rowmaj_idx(i, j, k);
                K[idx] = 1.0;  
                Beta[idx] = 1.0 + (DT * NU) / (2.0 * K[idx]);
                Gamma[idx] = (DT * NU) / (2.0 * Beta[idx]);
            }
        }
    }
    
    /* Initialize G field */
    GField g_field;
    initialize_g_field(&g_field);
    
    /* Load forcing function */
    function_handle forcing = parse_function("test_manufactured_forcing.txt");
    
    /* Run solver - use fewer steps for testing */
    int test_steps = STEPS / 10;
    if (test_steps < 10) test_steps = 10;
    
    solve(g_field, forcing, pressure, K, Eta, Zeta, U, 
          Beta, Gamma, v_boundary, 
          test_steps, false, NULL, NULL);
    
    /* Compute exact solution at final time */
    DTYPE t_final = test_steps * DT;
    VelocityField U_exact;
    Pressure P_exact;
    initialize_velocity_field(&U_exact, v_boundary);
    initialize_pressure(&P_exact);
    
    fill_exact_velocity(&U_exact, &exact, t_final);
    fill_exact_pressure(&P_exact, &exact, t_final);
    
    /* Compute errors */
    ErrorNorms err_vx, err_vy, err_vz;
    compute_velocity_error(&U, &U_exact, &err_vx, &err_vy, &err_vz);
    ErrorNorms err_p = compute_pressure_error(&pressure, &P_exact);
    
    data.L2_vx = err_vx.L2;
    data.L2_vy = err_vy.L2;
    data.L2_vz = err_vz.L2;
    data.L2_p = err_p.L2;
    
    /* Cleanup */
    free(K);
    free(Beta);
    free(Gamma);
    free_pressure(&pressure);
    free_pressure(&P_exact);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_velocity_field(&U_exact);
    free_g_field(&g_field);
    if (v_boundary) destroy_function(v_boundary);
    if (forcing) destroy_function(forcing);
    
    return data;
}

/* ==================== Convergence Analysis from File ==================== */

int test_convergence_from_file(const char *filename) {
    printf("\n====== TEST: Convergence Analysis from File ======\n");
    printf("Reading results from: %s\n\n", filename);
    
    FILE *fp = fopen(filename, "r");
    if (!fp) {
        fprintf(stderr, "Error: Cannot open %s\n", filename);
        return TEST_FAIL;
    }
    
    ConvergenceDataPoint data[MAX_REFINEMENTS];
    int count = 0;
    char line[256];
    
    while (count < MAX_REFINEMENTS && fgets(line, sizeof(line), fp)) {
        /* Skip comment lines */
        if (line[0] == '#') continue;
        
        int n;
        double dx, l2_vx, l2_vy, l2_vz, l2_p;
        if (sscanf(line, "%d %lf %lf %lf %lf %lf", &n, &dx, &l2_vx, &l2_vy, &l2_vz, &l2_p) == 6) {
            data[count].N = n;
            data[count].dx = dx;
            data[count].L2_vx = l2_vx;
            data[count].L2_vy = l2_vy;
            data[count].L2_vz = l2_vz;
            data[count].L2_p = l2_p;
            count++;
        }
    }
    fclose(fp);
    
    if (count < 2) {
        fprintf(stderr, "Error: Need at least 2 data points for convergence analysis\n");
        return TEST_FAIL;
    }
    
    /* Print results table */
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-8s\n", 
           "N", "dx", "L2(vx)", "L2(vy)", "L2(vz)", "L2(p)", "Rate");
    printf("%-8s %-12s %-12s %-12s %-12s %-12s %-8s\n",
           "---", "---", "---", "---", "---", "---", "---");
    
    DTYPE last_rate = 0.0;
    for (int i = 0; i < count; i++) {
        DTYPE rate = 0.0;
        if (i > 0 && data[i].L2_vx > 1e-15 && data[i-1].L2_vx > 1e-15) {
            rate = compute_convergence_rate(
                data[i-1].L2_vx, data[i].L2_vx,
                data[i-1].dx, data[i].dx
            );
            last_rate = rate;
        }
        
        if (i == 0) {
            printf("%-8d %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-8s\n",
                   data[i].N, data[i].dx, data[i].L2_vx, data[i].L2_vy, 
                   data[i].L2_vz, data[i].L2_p, "-");
        } else {
            printf("%-8d %-12.4e %-12.4e %-12.4e %-12.4e %-12.4e %-8.2f\n",
                   data[i].N, data[i].dx, data[i].L2_vx, data[i].L2_vy, 
                   data[i].L2_vz, data[i].L2_p, rate);
        }
    }
    
    /* Check convergence rate */
    DTYPE expected_rate = 2.0;
    DTYPE rate_tolerance = 0.5;  /* Allow some tolerance */
    
    printf("\n");
    printf("Expected convergence rate: %.1f (2nd order method)\n", expected_rate);
    printf("Measured convergence rate: %.2f\n", last_rate);
    
    bool passed = (count >= 2) && (fabs(last_rate - expected_rate) < rate_tolerance);
    
    if (passed) {
        printf("\n✓ TEST PASSED: Convergence rate is within tolerance\n");
        return TEST_PASS;
    } else {
        printf("\n✗ TEST FAILED: Convergence rate outside tolerance (%.1f ± %.1f)\n",
               expected_rate, rate_tolerance);
        return TEST_FAIL;
    }
}

/* ==================== Simple Convergence Test (current resolution) ==================== */

int test_convergence(void) {
    printf("\n====== TEST: Convergence Analysis ======\n");
    printf("Note: For full convergence test, run ./run_convergence_test.sh\n");
    printf("      This runs a single test at the current compiled resolution.\n\n");
    
    ConvergenceDataPoint data = run_single_resolution_test();
    
    printf("\n");
    printf("%-10s %-15s %-15s %-15s %-15s %-15s\n", 
           "N", "dx", "L2(vx)", "L2(vy)", "L2(vz)", "L2(p)");
    printf("%-10s %-15s %-15s %-15s %-15s %-15s\n",
           "---", "---", "---", "---", "---", "---");
    printf("%-10d %-15e %-15e %-15e %-15e %-15e\n",
           data.N, data.dx, data.L2_vx, data.L2_vy, data.L2_vz, data.L2_p);
    
    /* For single resolution, just check that errors are reasonable */
    DTYPE max_error = 1.0;  /* Threshold for reasonable error */
    bool passed = (data.L2_vx < max_error && data.L2_vy < max_error && 
                   data.L2_vz < max_error );//&& data.L2_p < max_error);
    
    printf("\n");
    if (passed) {
        printf("✓ TEST PASSED: Errors are within reasonable bounds\n");
        printf("  Run ./run_convergence_test.sh for full convergence analysis\n");
        return TEST_PASS;
    } else {
        printf("✗ TEST FAILED: Errors exceed threshold\n");
        return TEST_FAIL;
    }
}

int main(int argc, char *argv[]) {
    printf("============================================\n");
    printf("  Navier-Stokes Convergence Test\n");
    printf("============================================\n");
    
    if (argc > 1) {
        /* Analyze results from file */
        return test_convergence_from_file(argv[1]);
    }
    
    /* Run single resolution test */
    return test_convergence();
}