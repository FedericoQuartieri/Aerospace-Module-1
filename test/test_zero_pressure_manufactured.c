#include "test_common.h"
#include "../include/solve.h"
#include "../include/g_field.h"

/* 
    This test is supposed to be done with a grid normalized between [0, PI]
*/
/* ==================== Manufactured Solution Definition ==================== */

static DTYPE manufactured_zero_velocity(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    /* x ∈ [0,π] */
    switch (component) {
        case 0: return sin(x) * cos(t + y) * sin(z);         
        case 1: return cos(x) * sin(t + y) * sin(z);           
        case 2: return 2.0 * cos(x) * cos(t + y) * cos(z);    
        default: return 0.0;
    }
}

static DTYPE manufactured_zero_pressure(DTYPE x, DTYPE y, DTYPE z, DTYPE t) {
    /* x ∈ [0,π] */
    return 0.0;
}

/* 
 * Forcing term f = ∂u/∂t - (NU)∇²u + ∇p + (NU/k)u
 * where ∇p = 0 and k = 1
 */
static DTYPE manufactured_zero_forcing(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    /* x ∈ [0,π] */
    
    /* U */
    DTYPE u_x = manufactured_zero_velocity(x , y, z, t, 0);
    DTYPE u_y = manufactured_zero_velocity(x , y, z, t, 1);
    DTYPE u_z = manufactured_zero_velocity(x , y, z, t, 2);

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
    DTYPE dpdx = 0.0;
    DTYPE dpdy = 0.0;
    DTYPE dpdz = 0.0;

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

static DTYPE manufactured_zero_boundary(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    /* For Dirichlet BC, boundary = exact solution */
    return manufactured_zero_velocity(x, y, z, t, component);
}

/* ==================== Exact manufactured solution ==================== */
static void write_exact_solution_vti(const char *filename, ExactSolution *exact, DTYPE t) {
    VelocityField U_exact;
    Pressure P_exact;
    
    U_exact.v_x = (DTYPE*) malloc(GRID_SIZE);
    U_exact.v_y = (DTYPE*) malloc(GRID_SIZE);
    U_exact.v_z = (DTYPE*) malloc(GRID_SIZE);
    P_exact.p = (DTYPE*) malloc(GRID_SIZE);
    
    /* Fill with exact solution */
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
                
                U_exact.v_x[idx] = exact->velocity(vel_x, y, z, t, 0);
                U_exact.v_y[idx] = exact->velocity(x, vel_y, z, t, 1);
                U_exact.v_z[idx] = exact->velocity(x, y, vel_z, t, 2);
                P_exact.p[idx] = exact->pressure(x, y, z, t);
            }
        }
    }
    
    write_vti_file(filename, &U_exact, &P_exact);
    
    free(U_exact.v_x);
    free(U_exact.v_y);
    free(U_exact.v_z);
    free(P_exact.p);
}

/* ==================== Test Implementation ==================== */

static ExactSolution create_manufactured_solution(void) {
    ExactSolution exact = {
        .velocity = manufactured_zero_velocity,
        .pressure = manufactured_zero_pressure,
        .forcing = manufactured_zero_forcing,
        .boundary = manufactured_zero_boundary
    };
    return exact;
}

static const Data TEST_ZERO_PRESSURE_DATA = {
    .name = "Zero Pressure Manufactured Solution",
    .bc_velocity = manufactured_zero_boundary,
    .forcing = manufactured_zero_forcing
};


int test_manufactured_solution(void) {
    printf("\n====== TEST: Zero Pressure Paper Manufactured Solution ======\n");
    
    ExactSolution exact = create_manufactured_solution();

    printf("Writing exact solutions to VTI files...\n");
    mkdir("output_exact", 0755);
    /* Write exact solution for each timestep */
    for(int t = 0; t <= STEPS; t++) {
        int write_frequency = WRITE_FREQUENCY;
        if((t % write_frequency) == 0) {
            double time = t*DT;
            char filename[256];
            sprintf(filename, "output_exact/exact_solution_%06d.vti", t);
            write_exact_solution_vti(filename, &exact, time);
        }
    }
    
    /* Initialize fields */
    Pressure pressure;
    initialize_pressure(&pressure);
    
    VelocityField Eta, Zeta, U;
    
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);
    
    /* 
        Set initial conditions from exact solution at t=0,
        in the manufactured solution method we need to start form it and verify convergence
    */
    fill_exact_velocity(&U, &exact, 0.0);
    fill_exact_velocity(&Eta, &exact, 0.0);
    fill_exact_velocity(&Zeta, &exact, 0.0);
    fill_exact_pressure(&pressure, &exact, 0.0);


    /* Initialize K, Beta, Gamma */
    DTYPE *K = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Beta = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *)malloc(GRID_SIZE);
    
    /* K = 1 constant for now, then we might test K = sinx *siny *sinz */
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

    /* Run solver */
    solve(g_field, &TEST_ZERO_PRESSURE_DATA, pressure, K, Eta, Zeta, U, 
          Beta, Gamma, 
          WRITE_FREQUENCY, false, NULL, NULL);  
    
    /* Compute exact solution at final time */
    DTYPE t_final = STEPS * DT;
    VelocityField U_exact;
    Pressure P_exact;
    initialize_velocity_field(&U_exact);
    initialize_pressure(&P_exact);
    
    fill_exact_velocity(&U_exact, &exact, t_final);

    /* p is solved for timestep n+1/2 */
    fill_exact_pressure(&P_exact, &exact, t_final - DT / 2);

    translate_pressure_to_origin(&pressure);
    translate_pressure_to_origin(&P_exact);
    
    /* Compute errors */
    TestResult result;
    result.grid_size = WIDTH;
    result.dx = DX;
    
    compute_velocity_error(&U, &U_exact, 
                          &result.velocity_err[0],
                          &result.velocity_err[1],
                          &result.velocity_err[2]);
    result.pressure_err = compute_pressure_error(&pressure, &P_exact);
    result.convergence_rate = 0.0;
    
    print_test_result(&result, "Zero Pressure Paper Manufactured Solution Test");
    
    /* Check if errors are within acceptable tolerance */
    DTYPE tol = 1e-3;  /* Adjust based on expected accuracy */
    bool passed = (result.velocity_err[0].L2 < tol &&
                   result.velocity_err[1].L2 < tol &&
                   result.velocity_err[2].L2 < tol);
    
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
    
    if (passed) {
        printf("TEST PASSED\n");
        return TEST_PASS;
    } else {
        printf("TEST FAILED\n");
        return TEST_FAIL;
    }
}

int main(void) {
    printf("Running Zero Pressure Paper Manufactured Solution Test...\n");
    return test_manufactured_solution();
}
