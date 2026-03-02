#include "velocity_field.h"
#include <stdlib.h>
#include <string.h>

void initialize_velocity_field(VelocityField *v_field, function_handle v_boundary) {
    v_field->v_x = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_y = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_z = (DTYPE*) malloc(GRID_SIZE);


    //update_left_velocity_boundary(v_field, v_boundary, 0);
    //update_right_velocity_boundary(v_field, v_boundary, 0);
}

void rand_fill_velocity_field(VelocityField *v_field) {
    rand_fill(v_field->v_x);
    rand_fill(v_field->v_y);
    rand_fill(v_field->v_z);
}

/* 
    The divergence of the Velocity is computed in the pressure points,
    with div(U_n) = (Ux_n - Ux_n-1) / DX + (Uy_n - Uy_n-1) / DY + (Uz_n - Uz_n-1) / DZ

    note: in the left boundaries points, we assume the div(U) = 0 
*/

DTYPE compute_velocity_x_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    uint64_t idx = rowmaj_idx(i, j, k);
    uint64_t left = rowmaj_idx(i-1, j, k);

    return (v_component[idx] - v_component[left]) * DX_INVERSE;
}

DTYPE compute_velocity_y_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    uint64_t idx = rowmaj_idx(i, j, k);
    uint64_t left = rowmaj_idx(i, j-1, k);

    return (v_component[idx] - v_component[left]) * DY_INVERSE;
}

DTYPE compute_velocity_z_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    uint64_t idx = rowmaj_idx(i,j,k);
    uint64_t left = rowmaj_idx(i, j, k-1);

    return (v_component[idx] - v_component[left]) * DZ_INVERSE;
}

DTYPE compute_velocity_xx_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the x-direction -> (i-1,i,i+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i-1,j,k);
    size_t right_idx = rowmaj_idx(i+1,j,k);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DX_INVERSE_SQUARE;
}

DTYPE compute_velocity_yy_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the y-direction -> (j-1,j,j+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i,j-1,k);
    size_t right_idx = rowmaj_idx(i,j+1,k);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DY_INVERSE_SQUARE;
}

DTYPE compute_velocity_zz_grad(DTYPE *v_component, size_t i, size_t j, size_t k){
    // Gradient is done along the z-direction -> (z-1,z,z+1)
    size_t idx = rowmaj_idx(i,j,k);
    size_t left_idx = rowmaj_idx(i,j,k-1);
    size_t right_idx = rowmaj_idx(i,j,k+1);

    return (v_component[left_idx] - 2*v_component[idx] + v_component[right_idx]) * DZ_INVERSE_SQUARE;
}

void free_velocity_field(VelocityField *v_field) {
    free(v_field->v_x);
    free(v_field->v_y);
    free(v_field->v_z);
}

/* 
    Update by Ema:

    We might want to pass to eval_delta_function() the physical coordinates (i.e. i*DX, j*DY, k*DZ) instead of 
    the index in the grid.
    This is done since we have a physical domain [0, L]^3 and a grid built as:
    DX = L/(WIDTH-1), DY = L/(HEIGHT-1), DZ = L/(DEPTH-1), DT = Total_time/Steps

    So to get the physical coordinates from the index of the grid:
    x = i * DX , y = j * DY, z = k * DZ, t = time_step * DT

    -> After that, we have the physical coordinate of the pressure point, but since we are using
    a staggered grid, where velocity point are shifted of D/2, we then set:
    v_x (coordinate) = (x + DX/2, y, z)
    v_y (coordinate) = (x, y + DY/2, z)
    v_z (coordinate) = (x, y, z + DZ/2)

*/
void update_delta_left_velocity_boundary(VelocityField *v_field, function_handle v_boundary, int time_step) {
    size_t idx;
    DTYPE t = time_step * DT;  

    /* 
        Warning: currently I'm computing the delta of the function here supposing that the eval_delta_function 
        took the timestep, while instead it use the physical time.
        -> FIXED !
    */

    // (i,j,k) = (0,j,k) 
    for(int k = 1; k < DEPTH; k++){
        for(int j = 1; j < HEIGHT; j++){
            idx = rowmaj_idx(0, j, k);
            DTYPE y = j * DY;
            DTYPE z = k * DZ;
            DTYPE vel_y = y + DY/2; 
            DTYPE vel_z = z + DZ/2;
            
            v_field->v_x[idx] = eval_delta_function(v_boundary, 0.0, y, z, t, 0) 
                + DX/2 * (- ((eval_delta_function(v_boundary, 0.0, vel_y, z, t, 1) 
                            - eval_delta_function(v_boundary, 0.0, vel_y - DY, z, t, 1)) * DY_INVERSE) 
                         - ((eval_delta_function(v_boundary, 0.0, y, vel_z, t, 2) 
                            - eval_delta_function(v_boundary, 0.0, y, vel_z - DZ, t, 2)) * DZ_INVERSE));
            v_field->v_y[idx] = eval_delta_function(v_boundary, 0.0, vel_y, z, t, 1);
            v_field->v_z[idx] = eval_delta_function(v_boundary, 0.0, y, vel_z, t, 2); 
        }
    }

    // (i,j,k) = (i,0,k) 
    for(int k = 1; k < DEPTH; k++){
        for(int i = 1; i < WIDTH; i++){
            idx = rowmaj_idx(i, 0, k);
            DTYPE x = i * DX;
            DTYPE z = k * DZ;
            DTYPE vel_x = x + DX/2; 
            DTYPE vel_z = z + DZ/2;
            
            v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, 0.0, z, t, 0);
            v_field->v_y[idx] = eval_delta_function(v_boundary, x, 0.0, z, t, 1)
                + DY/2 * (- ((eval_delta_function(v_boundary, vel_x, 0.0, z, t, 0) 
                            - eval_delta_function(v_boundary, vel_x - DX, 0.0, z, t, 0)) * DX_INVERSE) 
                         - ((eval_delta_function(v_boundary, x, 0.0, vel_z, t, 2) 
                            - eval_delta_function(v_boundary, x, 0.0, vel_z - DZ, t, 2)) * DZ_INVERSE));
            v_field->v_z[idx] = eval_delta_function(v_boundary, x, 0.0, vel_z, t, 2);
        }
    }

    // (i,j,k) = (i,j,0) 
    for(int j = 1; j < HEIGHT; j++){
        for(int i = 1; i < WIDTH; i++){
            idx = rowmaj_idx(i, j, 0);
            DTYPE x = i * DX;
            DTYPE y = j * DY;
            DTYPE vel_x = x + DX/2;
            DTYPE vel_y = y + DY/2;
            
            v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, y, 0.0, t, 0);
            v_field->v_y[idx] = eval_delta_function(v_boundary, x, vel_y, 0.0, t, 1);
            v_field->v_z[idx] = eval_delta_function(v_boundary, x, y, 0.0, t, 2)
                + DZ/2 * (- ((eval_delta_function(v_boundary, vel_x, y, 0.0, t, 0) 
                            - eval_delta_function(v_boundary, vel_x - DX, y, 0.0, t, 0)) * DX_INVERSE) 
                         - ((eval_delta_function(v_boundary, x, vel_y, 0.0, t, 1) 
                            - eval_delta_function(v_boundary, x, vel_y - DY, 0.0, t, 1)) * DY_INVERSE));
        }
    }
    
    // (i,j,k) = (0,0,k)
    for(int k = 1; k < DEPTH; k++){
        idx = rowmaj_idx(0, 0, k);
        DTYPE z = k * DZ;
        DTYPE vel_x = DX/2;
        DTYPE vel_y = DY/2;
        DTYPE vel_z = z + DZ/2;

        v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, 0.0, z, t, 0);
        v_field->v_y[idx] = eval_delta_function(v_boundary, 0.0, vel_y, z, t, 1); 
        v_field->v_z[idx] = eval_delta_function(v_boundary, 0.0, 0.0, vel_z, t, 2);
    }

    // (i,j,k) = (0,j,0)
    for(int j = 1; j < HEIGHT; j++){
        idx = rowmaj_idx(0, j, 0);
        DTYPE y = j * DY;
        DTYPE vel_x = DX/2;
        DTYPE vel_y = y + DY/2;
        DTYPE vel_z = DZ/2;

        v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, y, 0.0, t, 0); 
        v_field->v_y[idx] = eval_delta_function(v_boundary, 0.0, vel_y, 0.0, t, 1);
        v_field->v_z[idx] = eval_delta_function(v_boundary, 0.0, y, vel_z, t, 2); 
    }

    // (i,j,k) = (i,0,0)
    for(int i = 1; i < WIDTH; i++){
        idx = rowmaj_idx(i, 0, 0);
        DTYPE x = i * DX;
        DTYPE vel_x = x + DX/2;
        DTYPE vel_y = DY/2;
        DTYPE vel_z = DZ/2;

        v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, 0.0, 0.0, t, 0);
        v_field->v_y[idx] = eval_delta_function(v_boundary, x, vel_y, 0.0, t, 1);
        v_field->v_z[idx] = eval_delta_function(v_boundary, x, 0.0, vel_z, t, 2);
    }

    // (i,j,k) = (0,0,0)
    idx = rowmaj_idx(0, 0, 0);
    v_field->v_x[idx] = eval_delta_function(v_boundary, DX/2, 0.0, 0.0, t, 0);
    v_field->v_y[idx] = eval_delta_function(v_boundary, 0.0, DY/2, 0.0, t, 1);
    v_field->v_z[idx] = eval_delta_function(v_boundary, 0.0, 0.0, DZ/2, t, 2);

}

// !! Warning: this functions puts all the components as if they were on the boundary walls
// and this is exactly what Thomas functions expects (in the current implementation) !!
// (Thomas expects U_ex on the boundary velocity value in the array *u passed as parameter)
void update_delta_right_velocity_boundary(VelocityField *v_field, function_handle v_boundary, int time_step) {
    size_t idx;
    DTYPE t = time_step * DT;

    // (WIDTH-1, j, k): right x boundary face,
    for(int k = 1; k < DEPTH; k++) {
        for(int j = 1; j < HEIGHT; j++) { 
            idx = rowmaj_idx(WIDTH-1, j, k);
            DTYPE x = (WIDTH-1)*DX; 
            DTYPE y = j*DY; 
            DTYPE z = k*DZ;
            DTYPE vel_x = x + DX/2;
            DTYPE vel_y = y + DY/2;
            DTYPE vel_z = z + DZ/2;

            v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, y, z, t, 0);
            v_field->v_y[idx] = eval_delta_function(v_boundary, vel_x, vel_y, z, t, 1);
            v_field->v_z[idx] = eval_delta_function(v_boundary, vel_x, y, vel_z, t, 2);
        }
    }

    // (i, HEIGHT-1, k): right y boundary face
    for(int k = 1; k < DEPTH; k++) {
        for(int i = 1; i < WIDTH; i++) { 
            idx = rowmaj_idx(i, HEIGHT-1, k);
            DTYPE x = i*DX; 
            DTYPE y = (HEIGHT-1)*DY; 
            DTYPE z = k*DZ;
            DTYPE vel_x = x + DX/2;
            DTYPE vel_y = y + DY/2;
            DTYPE vel_z = z + DZ/2;

            v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, vel_y, z, t, 0);
            v_field->v_y[idx] = eval_delta_function(v_boundary, x, vel_y, z, t, 1);
            v_field->v_z[idx] = eval_delta_function(v_boundary, x, vel_y, vel_z, t, 2);
        }
    }

    // (i, j, DEPTH-1): right z boundary face
    for(int j = 1; j < HEIGHT; j++) {
        for(int i = 1; i < WIDTH; i++) { 
            idx = rowmaj_idx(i, j, DEPTH-1);
            DTYPE x = i*DX; 
            DTYPE y = j*DY; 
            DTYPE z = (DEPTH-1)*DZ;
            DTYPE vel_x = x + DX/2;
            DTYPE vel_y = y + DY/2;
            DTYPE vel_z = z + DZ/2;

            v_field->v_x[idx] = eval_delta_function(v_boundary, vel_x, y, vel_z, t, 0);
            v_field->v_y[idx] = eval_delta_function(v_boundary, x, vel_y, vel_z, t, 1);
            v_field->v_z[idx] = eval_delta_function(v_boundary, x, y, vel_z, t, 2);
        }
    }
}