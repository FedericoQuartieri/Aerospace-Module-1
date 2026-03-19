/**
 * test_convergence.c
 * 
 * Test for spatial and temporal convergence using the paper manufactured solution.
 * Outputs results in a parsable format (JSON line) for automated convergence studies.
 * 
 * Usage: ./test_convergence [output_file]
 *   If output_file is provided, appends results to it; otherwise prints to stdout.
 */

 /* 
    Warning: check that the manufactured solution is the one that you want to use,
            check then that the forcing and boundary file corresponds,
            then check the parameters in constant.h, especially the lenght of the pyhisical domain (i.e. [0, 1] or [0, pi])
 */
#include "test_common.h"
#include "../include/solve.h"
#include "../include/g_field.h"
#include "../include/function.h"
#include <sys/stat.h>

/* ==================== Manufactured Solution ==================== */


static DTYPE manufactured_velocity(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    /* x ∈ [0,π] */
    switch (component) {
        case 0: return sin(x) * cos(t + y) * sin(z);         
        case 1: return cos(x) * sin(t + y) * sin(z);           
        case 2: return 2.0 * cos(x) * cos(t + y) * cos(z);    
        default: return 0.0;
    }
}

static DTYPE manufactured_pressure(DTYPE x, DTYPE y, DTYPE z, DTYPE t) {
    /* x ∈ [0,π] */
    return - 3.0 * NU * cos(x) * cos(t + y) * cos(z);
}

/* 
 * Forcing term f = ∂u/∂t - (NU)∇²u + ∇p + (NU/k)u
 * where k = 1
 */
static DTYPE manufactured_forcing(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    /* x ∈ [0,π] */
    
    /* U */
    DTYPE u_x = manufactured_velocity(x , y, z, t, 0);
    DTYPE u_y = manufactured_velocity(x , y, z, t, 1);
    DTYPE u_z = manufactured_velocity(x , y, z, t, 2);

    /* Time derivative du/dt */
    DTYPE dudt_x =   - sin(x) * sin(t + y) * sin(z);
    DTYPE dudt_y =   cos(x) * cos(t + y) * sin(z);
    DTYPE dudt_z =   - 2.0 * cos(x) * sin(t + y) * cos(z);

    /* Laplacian: ∇²u */
    DTYPE lapU_x = -3.0 * u_x;
    DTYPE lapU_y = -3.0 * u_y;
    DTYPE lapU_z = -3.0 * u_z;

    // Simple test with k = 1 constant in all the domain
    DTYPE k = 1;

    /* Pressure gradient: ∇p  */
    DTYPE dpdx = 3.0 * NU * sin(x) * cos(t + y) * cos(z);
    DTYPE dpdy = 3.0 * NU * cos(x) * sin(t + y) * cos(z);
    DTYPE dpdz = 3.0 * NU * cos(x) * cos(t + y) * sin(z);

    /* f = ∂u/∂t - (NU)∇²u + (NU/k)u + ∇p */
    switch (component) {
        case 0: 
            return dudt_x - NU * lapU_x + (NU/k) * u_x + dpdx;
        case 1:
            return dudt_y - NU * lapU_y + (NU/k) * u_y + dpdy;
        case 2:
            return dudt_z - NU * lapU_z + (NU/k) * u_z + dpdz;    
        default: 
            return 0.0;
    }
}

static DTYPE manufactured_boundary(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    return manufactured_velocity(x, y, z, t, component);
}

/* ==================== Test Implementation ==================== */

static ExactSolution create_manufactured_solution(void) {
    ExactSolution exact = {
        .velocity = manufactured_velocity,
        .pressure = manufactured_pressure,
        .forcing = manufactured_forcing,
        .boundary = manufactured_boundary
    };
    return exact;
}

typedef struct {
    int width, height, depth;
    DTYPE dx, dy, dz, dt;
    int steps;
    DTYPE total_time;
    ErrorNorms vel_err[3];
    ErrorNorms pres_err;
} ConvergenceResult;

static void print_result_json(FILE *fp, ConvergenceResult *res) {
    fprintf(fp, "{\"width\":%d,\"height\":%d,\"depth\":%d,"
                "\"dx\":%.15e,\"dy\":%.15e,\"dz\":%.15e,\"dt\":%.15e,"
                "\"steps\":%d,\"total_time\":%.15e,"
                "\"vel_x_L2\":%.15e,\"vel_y_L2\":%.15e,\"vel_z_L2\":%.15e,"
                "\"vel_x_Linf\":%.15e,\"vel_y_Linf\":%.15e,\"vel_z_Linf\":%.15e,"
                "\"pres_L2\":%.15e,\"pres_Linf\":%.15e}\n",
            res->width, res->height, res->depth,
            res->dx, res->dy, res->dz, res->dt,
            res->steps, res->total_time,
            res->vel_err[0].L2, res->vel_err[1].L2, res->vel_err[2].L2,
            res->vel_err[0].Linf, res->vel_err[1].Linf, res->vel_err[2].Linf,
            res->pres_err.L2, res->pres_err.Linf);
}

int run_convergence_test(FILE *output_fp) {
    ExactSolution exact = create_manufactured_solution();
    
    /* Initialize fields */
    Pressure pressure;
    initialize_pressure(&pressure);
    
    VelocityField Eta, Zeta, U;
    function_handle v_boundary = parse_function("../../function_files/test_paper_manufactured_Vboundary.txt");
    
    if (!v_boundary) {
        fprintf(stderr, "Error: Could not load boundary function file\n");
        return TEST_FAIL;
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
    function_handle forcing = parse_function("../../function_files/test_paper_manufactured_forcing.txt");
    
    if (!forcing) {
        fprintf(stderr, "Error: Could not load forcing function file\n");
        return TEST_FAIL;
    }

    /* Run solver - no file output for convergence tests */
    solve(g_field, forcing, pressure, K, Eta, Zeta, U, 
          Beta, Gamma, v_boundary, 
          STEPS, false, NULL, NULL);  /* write_frequency > STEPS means no output */

    /* Compute exact solution at final time */
    DTYPE t_final = STEPS * DT;
    VelocityField U_exact;
    Pressure P_exact;
    initialize_velocity_field(&U_exact, v_boundary);
    initialize_pressure(&P_exact);
    
    fill_exact_velocity(&U_exact, &exact, t_final);
    /* p is solved for at timestep n+1/2 */
    fill_exact_pressure(&P_exact, &exact, t_final - DT / 2);

    translate_pressure_to_origin(&pressure);
    translate_pressure_to_origin(&P_exact);
    
    /* Compute errors and store results */
    ConvergenceResult result;
    result.width = WIDTH;
    result.height = HEIGHT;
    result.depth = DEPTH;
    result.dx = DX;
    result.dy = DY;
    result.dz = DZ;
    result.dt = DT;
    result.steps = STEPS;
    result.total_time = TOTAL_TIME;

    compute_velocity_error(&U, &U_exact, 
                          &result.vel_err[0],
                          &result.vel_err[1],
                          &result.vel_err[2]);
    result.pres_err = compute_pressure_error(&pressure, &P_exact);
    
    /* Output result */
    print_result_json(output_fp, &result);
    
    /* Also print summary to stderr for visibility */
    fprintf(stderr, "Grid: %dx%dx%d, DX=%.4e, DT=%.4e\n", WIDTH, HEIGHT, DEPTH, DX, DT);
    fprintf(stderr, "  Vel L2 errors: [%.4e, %.4e, %.4e]\n", 
            result.vel_err[0].L2, result.vel_err[1].L2, result.vel_err[2].L2);
    fprintf(stderr, "  Pressure L2 error: %.4e\n", result.pres_err.L2);
    
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
    destroy_function(v_boundary);
    destroy_function(forcing);
    
    return TEST_PASS;
}

int main(int argc, char *argv[]) {
    FILE *output_fp = stdout;
    
    if (argc > 1) {
        output_fp = fopen(argv[1], "a");  /* Append mode */
        if (!output_fp) {
            fprintf(stderr, "Error: Could not open output file %s\n", argv[1]);
            return TEST_FAIL;
        }
    }
    
    int result = run_convergence_test(output_fp);
    
    if (output_fp != stdout) {
        fclose(output_fp);
    }
    
    return result;
}
