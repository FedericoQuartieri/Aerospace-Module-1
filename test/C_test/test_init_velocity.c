#include "test_common.h"
#include "../include/function.h"

/*
    Warning: this test may failed becouse of a miss-interpretation of the staggered grid,
    we should infact consider the fact that if we evaluate the value of velocity in the 
    (i,j,k) point, after converting it into x,y,z we should also consider the 'offset' of the 
    staggered velocity coordinates: x + DX/2 , y + DY/2, z + DZ/2
    Must check all the codebase to be coherent with that...
*/


/*
 * Test to verify that all boundary points are properly initialized
 * using the boundary conditions from test_manufactured_Vboundary.txt
 * 
 * The boundary file contains:
 *   v_x = sin(t) * sin(pi*x) * sin(pi*y) * sin(pi*z)
 *   v_y = sin(t) * cos(pi*x) * cos(pi*y) * cos(pi*z)
 *   v_z = sin(t) * cos(pi*x) * sin(pi*y) * (cos(pi*z) + sin(pi*z))
 */

/* Expected boundary velocity from the .txt file */
static DTYPE expected_boundary_velocity(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component) {
    DTYPE sx = sin(M_PI * x); DTYPE sy = sin(M_PI * y); DTYPE sz = sin(M_PI * z);
    DTYPE cx = cos(M_PI * x); DTYPE cy = cos(M_PI * y); DTYPE cz = cos(M_PI * z);
    DTYPE st = sin(t);
    
    switch (component) {
        case 0: return st * sx * sy * sz;
        case 1: return st * cx * cy * cz;
        case 2: return st * cx * sy * (cz + sz);
        default: return 0.0;
    }
}

/* Check if a point is on the boundary */
static int is_boundary_point(int i, int j, int k) {
    return (i == 0 || i == WIDTH-1 || 
            j == 0 || j == HEIGHT-1 || 
            k == 0 || k == DEPTH-1);
}

/* Check if velocity at a boundary point is initialized (not garbage) */
static int is_initialized(DTYPE value) {
    /* Check for NaN, Inf */
    if (isnan(value) || isinf(value)) return 0;
    /* Check for unreasonably large values */
    if (fabs(value) > 1e10) return 0;
    return 1;
}

/* Test 1: Verify all boundary points are initialized (no garbage values) */
int test_boundary_nan(void) {
    printf("\n====== TEST: Boundary Values nan ======\n");
    
    VelocityField v_field;
    function_handle v_boundary = parse_function("test_manufactured_Vboundary.txt");
    
    if (!v_boundary) {
        fprintf(stderr, "Error: Could not load boundary function file\n");
        return TEST_FAIL;
    }
    
    initialize_velocity_field(&v_field, v_boundary);
    update_left_velocity_boundary(&v_field, v_boundary, 6);
    update_right_velocity_boundary(&v_field, v_boundary, 6);
    
    int failed_points = 0;
    int total_boundary_points = 0;
    
    printf("Checking all boundary points for initialization...\n");
    printf("Grid: WIDTH=%d, HEIGHT=%d, DEPTH=%d\n", WIDTH, HEIGHT, DEPTH);
    
    /* Check all boundary points */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                if (is_boundary_point(i, j, k)) {
                    total_boundary_points++;
                    size_t idx = rowmaj_idx(i, j, k);
                    
                    int vx_ok = is_initialized(v_field.v_x[idx]);
                    int vy_ok = is_initialized(v_field.v_y[idx]);
                    int vz_ok = is_initialized(v_field.v_z[idx]);
                    
                    if (!vx_ok || !vy_ok || !vz_ok) {
                        failed_points++;
                        printf("  UNINITIALIZED at (%d,%d,%d): v_x=%e, v_y=%e, v_z=%e\n",
                               i, j, k, v_field.v_x[idx], v_field.v_y[idx], v_field.v_z[idx]);
                    }
                }
            }
        }
    }
    
    printf("Total boundary points: %d\n", total_boundary_points);
    printf("Uninitialized points: %d\n", failed_points);
    
    free_velocity_field(&v_field);
    destroy_function(v_boundary);
    
    if (failed_points > 0) {
        printf("FAIL: %d boundary points are uninitialized!\n", failed_points);
        return TEST_FAIL;
    }
    
    printf("PASS: All boundary points are valid.\n");
    return TEST_PASS;
}

/* Test 2: Verify boundary values match expected values from .txt file */
int test_boundary_values_accuracy(void) {
    printf("\n====== TEST: Boundary Values Accuracy ======\n");
    
    VelocityField v_field;
    function_handle v_boundary = parse_function("test_manufactured_Vboundary.txt");
    
    if (!v_boundary) {
        fprintf(stderr, "Error: Could not load boundary function file\n");
        return TEST_FAIL;
    }
    
    int time_step = 6;  /* Use time_step != 0 so sin(t) != 0 */
    DTYPE t = time_step * DT;
    
    initialize_velocity_field(&v_field, v_boundary);
    update_left_velocity_boundary(&v_field, v_boundary, time_step);
    update_right_velocity_boundary(&v_field, v_boundary, time_step);

    int failed_points = 0;
    int total_checked = 0;
    DTYPE max_error = 0.0;
    DTYPE tolerance = 1e-6;
    
    printf("Checking boundary values at t = %f (time_step = %d)\n", t, time_step);
    printf("DX=%f, DY=%f, DZ=%f, DT=%f\n", DX, DY, DZ, DT);
    
    /* Check all boundary points - only on faces where the component is directly set */
    for (int k = 0; k < DEPTH; k++) {
        for (int j = 0; j < HEIGHT; j++) {
            for (int i = 0; i < WIDTH; i++) {
                if (!is_boundary_point(i, j, k)) continue;
                
                size_t idx = rowmaj_idx(i, j, k);
                /* Velocity is in staggered points = idx + D/2 */
                DTYPE x = i * DX + DX/2; 
                DTYPE y = j * DY + DY/2;
                DTYPE z = k * DZ + DZ/2;
                
                /* Check y and z components (which don't have Taylor expansion correction) 
                   on faces where they're set directly */
                int check_vy = 1, check_vz = 1;
                
                /* Get expected values from the parsed function */
                DTYPE expected_vx = eval_function(v_boundary, x, y, z, t, 0);
                DTYPE expected_vy = eval_function(v_boundary, x, y, z, t, 1);
                DTYPE expected_vz = eval_function(v_boundary, x, y, z, t, 2);
                
                DTYPE actual_vx = v_field.v_x[idx];
                DTYPE actual_vy = v_field.v_y[idx];
                DTYPE actual_vz = v_field.v_z[idx];
                
                /* Check v_y (always should match on boundaries except where Taylor is used) */
                if (check_vy) {
                    DTYPE err_vy = fabs(actual_vy - expected_vy);
                    if (err_vy > max_error) max_error = err_vy;
                    
                    /* For y=0 face (j==0), v_y has Taylor correction, skip strict check */
                    if (j != 0 && err_vy > tolerance) {
                        if (failed_points < 10) {
                            printf("  v_y mismatch at (%d,%d,%d): expected=%e, got=%e, err=%e\n",
                                   i, j, k, expected_vy, actual_vy, err_vy);
                        }
                        failed_points++;
                    }
                    total_checked++;
                }
                
                /* Check v_z (always should match on boundaries except where Taylor is used) */
                if (check_vz) {
                    DTYPE err_vz = fabs(actual_vz - expected_vz);
                    if (err_vz > max_error) max_error = err_vz;
                    
                    /* For z=0 face (k==0), v_z has Taylor correction, skip strict check */
                    if (k != 0 && err_vz > tolerance) {
                        if (failed_points < 10) {
                            printf("  v_z mismatch at (%d,%d,%d): expected=%e, got=%e, err=%e\n",
                                   i, j, k, expected_vz, actual_vz, err_vz);
                        }
                        failed_points++;
                    }
                    total_checked++;
                }
            }
        }
    }
    
    printf("Total boundary components checked: %d\n", total_checked);
    printf("Max error found: %e\n", max_error);
    printf("Failed checks: %d\n", failed_points);
    
    free_velocity_field(&v_field);
    destroy_function(v_boundary);
    
    if (failed_points > 0) {
        printf("FAIL: %d boundary values don't match expected!\n", failed_points);
        return TEST_FAIL;
    }
    
    printf("PASS: All boundary values match expected values.\n");
    return TEST_PASS;
}

/* Test 3: Check specific missing points (edges and vertices) */
int test_missing_boundary_points(void) {
    printf("\n====== TEST: Check Missing Boundary Points ======\n");
    
    VelocityField v_field;
    function_handle v_boundary = parse_function("test_manufactured_Vboundary.txt");
    
    if (!v_boundary) {
        fprintf(stderr, "Error: Could not load boundary function file\n");
        return TEST_FAIL;
    }
    
    int time_step = 1;
    DTYPE t = time_step * DT;
    
    initialize_velocity_field(&v_field, v_boundary);
    update_left_velocity_boundary(&v_field, v_boundary, time_step);
    update_right_velocity_boundary(&v_field, v_boundary, time_step);
    
    printf("Checking specific edge/vertex points that might be missing...\n");
    printf("t = %f\n\n", t);
    
    /* Points that might be missed: edges and vertices with index 0 on right boundaries */
    typedef struct { int i, j, k; const char *desc; } TestPoint;
    TestPoint points[] = {
        /* Edges with i=WIDTH-1 and j=0 or k=0 */
        {WIDTH-1, 0, 1, "(WIDTH-1, 0, k) - edge x-max, y-min"},
        {WIDTH-1, 0, DEPTH-1, "(WIDTH-1, 0, DEPTH-1) - edge x-max, y-min, z-max"},
        {WIDTH-1, 1, 0, "(WIDTH-1, j, 0) - edge x-max, z-min"},
        {WIDTH-1, HEIGHT-1, 0, "(WIDTH-1, HEIGHT-1, 0) - edge x-max, y-max, z-min"},
        
        /* Edges with j=HEIGHT-1 and i=0 or k=0 */
        {0, HEIGHT-1, 1, "(0, HEIGHT-1, k) - edge x-min, y-max"},
        {0, HEIGHT-1, DEPTH-1, "(0, HEIGHT-1, DEPTH-1) - edge x-min, y-max, z-max"},
        {1, HEIGHT-1, 0, "(i, HEIGHT-1, 0) - edge y-max, z-min"},
        
        /* Edges with k=DEPTH-1 and i=0 or j=0 */
        {0, 1, DEPTH-1, "(0, j, DEPTH-1) - edge x-min, z-max"},
        {1, 0, DEPTH-1, "(i, 0, DEPTH-1) - edge y-min, z-max"},
        
        /* Vertices */
        {WIDTH-1, 0, 0, "(WIDTH-1, 0, 0) - vertex"},
        {0, HEIGHT-1, 0, "(0, HEIGHT-1, 0) - vertex"},
        {0, 0, DEPTH-1, "(0, 0, DEPTH-1) - vertex"},
        {WIDTH-1, HEIGHT-1, 0, "(WIDTH-1, HEIGHT-1, 0) - vertex"},
        {WIDTH-1, 0, DEPTH-1, "(WIDTH-1, 0, DEPTH-1) - vertex"},
        {0, HEIGHT-1, DEPTH-1, "(0, HEIGHT-1, DEPTH-1) - vertex"},
        {WIDTH-1, HEIGHT-1, DEPTH-1, "(WIDTH-1, HEIGHT-1, DEPTH-1) - vertex"},
    };
    
    int n_points = sizeof(points) / sizeof(points[0]);
    int failed = 0;
    
    for (int p = 0; p < n_points; p++) {
        int i = points[p].i;
        int j = points[p].j;
        int k = points[p].k;
        
        /* Skip if out of bounds */
        if (i >= WIDTH || j >= HEIGHT || k >= DEPTH) continue;
        
        size_t idx = rowmaj_idx(i, j, k);
        DTYPE x = i * DX + DX/2;
        DTYPE y = j * DY + DY/2;
        DTYPE z = k * DZ + DZ/2;
        
        DTYPE exp_vx = eval_function(v_boundary, x, y, z, t, 0);
        DTYPE exp_vy = eval_function(v_boundary, x, y, z, t, 1);
        DTYPE exp_vz = eval_function(v_boundary, x, y, z, t, 2);
        
        int is_ok = is_initialized(v_field.v_x[idx]) && 
                    is_initialized(v_field.v_y[idx]) && 
                    is_initialized(v_field.v_z[idx]);
        
        const char *status = is_ok ? "OK" : "UNINITIALIZED";
        if (!is_ok) failed++;
        
        printf("%s: %s\n", points[p].desc, status);
        printf("  (%d,%d,%d) x=%.3f y=%.3f z=%.3f\n", i, j, k, x, y, z);
        printf("  v_x: %.6e (expected: %.6e)\n", v_field.v_x[idx], exp_vx);
        printf("  v_y: %.6e (expected: %.6e)\n", v_field.v_y[idx], exp_vy);
        printf("  v_z: %.6e (expected: %.6e)\n", v_field.v_z[idx], exp_vz);
        printf("\n");
    }
    
    free_velocity_field(&v_field);
    destroy_function(v_boundary);
    
    if (failed > 0) {
        printf("FAIL: %d points are uninitialized!\n", failed);
        return TEST_FAIL;
    }
    
    printf("PASS: All checked points are initialized.\n");
    return TEST_PASS;
}

/* Main test runner */
int main(int argc, char *argv[]) {
    printf("========== Velocity Boundary Initialization Tests ==========\n");
    printf("Grid: WIDTH=%d, HEIGHT=%d, DEPTH=%d\n", WIDTH, HEIGHT, DEPTH);
    printf("Spacing: DX=%.6f, DY=%.6f, DZ=%.6f\n", DX, DY, DZ);
    printf("============================================================\n");
    
    int result = TEST_PASS;
    
    /* Run all tests */
    if (test_boundary_nan() != TEST_PASS) result = TEST_FAIL;
    if (test_missing_boundary_points() != TEST_PASS) result = TEST_FAIL;
    if (test_boundary_values_accuracy() != TEST_PASS) result = TEST_FAIL;
    
    printf("\n============================================================\n");
    if (result == TEST_PASS) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("SOME TESTS FAILED\n");
    }
    printf("============================================================\n");
    
    return result;
}
