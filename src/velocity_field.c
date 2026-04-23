#include "velocity_field.h"
#include <stdlib.h>
#include <string.h>

void initialize_velocity_field(VelocityField *v_field) {
    v_field->v_x = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_y = (DTYPE*) malloc(GRID_SIZE);
    v_field->v_z = (DTYPE*) malloc(GRID_SIZE);

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
    uint64_t idx = rowmaj_idx(i, j, k);
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
void update_delta_left_velocity_boundary(VelocityField *v_field, int time_step, const Data *data) {
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
            
            v_field->v_x[idx] = delta_bc_velocity(data, 0.0, y, z, t, 0) 
                + DX/2 * (- ((delta_bc_velocity(data, 0.0, vel_y, z, t, 1) 
                            - delta_bc_velocity(data, 0.0, vel_y - DY, z, t, 1)) * DY_INVERSE) 
                         - ((delta_bc_velocity(data, 0.0, y, vel_z, t, 2) 
                            - delta_bc_velocity(data, 0.0, y, vel_z - DZ, t, 2)) * DZ_INVERSE));
            v_field->v_y[idx] = delta_bc_velocity(data, 0.0, vel_y, z, t, 1);
            v_field->v_z[idx] = delta_bc_velocity(data, 0.0, y, vel_z, t, 2); 
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
            
            v_field->v_x[idx] = delta_bc_velocity(data, vel_x, 0.0, z, t, 0);
            v_field->v_y[idx] = delta_bc_velocity(data, x, 0.0, z, t, 1)
                + DY/2 * (- ((delta_bc_velocity(data, vel_x, 0.0, z, t, 0) 
                            - delta_bc_velocity(data, vel_x - DX, 0.0, z, t, 0)) * DX_INVERSE) 
                         - ((delta_bc_velocity(data, x, 0.0, vel_z, t, 2) 
                            - delta_bc_velocity(data, x, 0.0, vel_z - DZ, t, 2)) * DZ_INVERSE));
            v_field->v_z[idx] = delta_bc_velocity(data, x, 0.0, vel_z, t, 2);
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
            
            v_field->v_x[idx] = delta_bc_velocity(data, vel_x, y, 0.0, t, 0);
            v_field->v_y[idx] = delta_bc_velocity(data, x, vel_y, 0.0, t, 1);
            v_field->v_z[idx] = delta_bc_velocity(data, x, y, 0.0, t, 2)
                + DZ/2 * (- ((delta_bc_velocity(data, vel_x, y, 0.0, t, 0) 
                            - delta_bc_velocity(data, vel_x - DX, y, 0.0, t, 0)) * DX_INVERSE) 
                         - ((delta_bc_velocity(data, x, vel_y, 0.0, t, 1) 
                            - delta_bc_velocity(data, x, vel_y - DY, 0.0, t, 1)) * DY_INVERSE));
        }
    }
    
    // (i,j,k) = (0,0,k)
    for(int k = 1; k < DEPTH; k++){
        idx = rowmaj_idx(0, 0, k);
        DTYPE z = k * DZ;
        DTYPE vel_x = DX/2;
        DTYPE vel_y = DY/2;
        DTYPE vel_z = z + DZ/2;

        v_field->v_x[idx] = delta_bc_velocity(data, vel_x, 0.0, z, t, 0);
        v_field->v_y[idx] = delta_bc_velocity(data, 0.0, vel_y, z, t, 1); 
        v_field->v_z[idx] = delta_bc_velocity(data, 0.0, 0.0, vel_z, t, 2);
    }

    // (i,j,k) = (0,j,0)
    for(int j = 1; j < HEIGHT; j++){
        idx = rowmaj_idx(0, j, 0);
        DTYPE y = j * DY;
        DTYPE vel_x = DX/2;
        DTYPE vel_y = y + DY/2;
        DTYPE vel_z = DZ/2;

        v_field->v_x[idx] = delta_bc_velocity(data, vel_x, y, 0.0, t, 0); 
        v_field->v_y[idx] = delta_bc_velocity(data, 0.0, vel_y, 0.0, t, 1);
        v_field->v_z[idx] = delta_bc_velocity(data, 0.0, y, vel_z, t, 2); 
    }

    // (i,j,k) = (i,0,0)
    for(int i = 1; i < WIDTH; i++){
        idx = rowmaj_idx(i, 0, 0);
        DTYPE x = i * DX;
        DTYPE vel_x = x + DX/2;
        DTYPE vel_y = DY/2;
        DTYPE vel_z = DZ/2;

        v_field->v_x[idx] = delta_bc_velocity(data, vel_x, 0.0, 0.0, t, 0);
        v_field->v_y[idx] = delta_bc_velocity(data, x, vel_y, 0.0, t, 1);
        v_field->v_z[idx] = delta_bc_velocity(data, x, 0.0, vel_z, t, 2);
    }

    // (i,j,k) = (0,0,0)
    idx = rowmaj_idx(0, 0, 0);
    v_field->v_x[idx] = delta_bc_velocity(data, DX/2, 0.0, 0.0, t, 0);
    v_field->v_y[idx] = delta_bc_velocity(data, 0.0, DY/2, 0.0, t, 1);
    v_field->v_z[idx] = delta_bc_velocity(data, 0.0, 0.0, DZ/2, t, 2);

}

// !! Warning: this functions puts all the components as if they were on the boundary walls
// and this is exactly what Thomas functions expects (in the current implementation) !!
// (Thomas expects U_ex on the boundary velocity value in the array *u passed as parameter)
void update_delta_right_velocity_boundary(VelocityField *v_field, int time_step, const Data *data) {
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

            v_field->v_x[idx] = delta_bc_velocity(data, vel_x, y, z, t, 0);
            v_field->v_y[idx] = delta_bc_velocity(data, vel_x, vel_y, z, t, 1);
            v_field->v_z[idx] = delta_bc_velocity(data, vel_x, y, vel_z, t, 2);
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

            v_field->v_x[idx] = delta_bc_velocity(data, vel_x, vel_y, z, t, 0);
            v_field->v_y[idx] = delta_bc_velocity(data, x, vel_y, z, t, 1);
            v_field->v_z[idx] = delta_bc_velocity(data, x, vel_y, vel_z, t, 2);
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

            v_field->v_x[idx] = delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
            v_field->v_y[idx] = delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
            v_field->v_z[idx] = delta_bc_velocity(data, x, y, vel_z, t, 2);
        }
    }
}



/*
    Returns the boundary velocity (vx, vy, vz) at grid point (i,j,k) and physical time t,
    covering all cases handled by update_delta_left_velocity_boundary and
    update_delta_right_velocity_boundary. Left boundaries take priority over right ones.
    For interior (non-boundary) points, all components are set to 0.
*/
/* WARNING: CURRENTLY T IS ALREADY PHYSICAL WHILE INSTEAD THE SPATIAL COORDINATE ARE NOT*/
/* void get_boundary_velocity(size_t i, size_t j, size_t k, DTYPE t, const Data *data,
                           DTYPE *vx, DTYPE *vy, DTYPE *vz) {
    DTYPE x = i * DX, y = j * DY, z = k * DZ;
    DTYPE vel_x = x + DX/2, vel_y = y + DY/2, vel_z = z + DZ/2;

    if (i == 0 && j == 0 && k == 0) {
        *vx = delta_bc_velocity(data, DX/2, 0.0, 0.0, t, 0);
        *vy = delta_bc_velocity(data, 0.0, DY/2, 0.0, t, 1);
        *vz = delta_bc_velocity(data, 0.0, 0.0, DZ/2, t, 2);
    } else if (i == 0 && j == 0) {
        // (0, 0, k)
        *vx = delta_bc_velocity(data, DX/2, 0.0, z, t, 0);
        *vy = delta_bc_velocity(data, 0.0, DY/2, z, t, 1);
        *vz = delta_bc_velocity(data, 0.0, 0.0, vel_z, t, 2);
    } else if (i == 0 && k == 0) {
        // (0, j, 0)
        *vx = delta_bc_velocity(data, DX/2, y, 0.0, t, 0);
        *vy = delta_bc_velocity(data, 0.0, vel_y, 0.0, t, 1);
        *vz = delta_bc_velocity(data, 0.0, y, DZ/2, t, 2);
    } else if (j == 0 && k == 0) {
        // (i, 0, 0)
        *vx = delta_bc_velocity(data, vel_x, 0.0, 0.0, t, 0);
        *vy = delta_bc_velocity(data, x, DY/2, 0.0, t, 1);
        *vz = delta_bc_velocity(data, x, 0.0, DZ/2, t, 2);
    } else if (i == 0) {
        // (0, j, k)
        *vx = delta_bc_velocity(data, 0.0, y, z, t, 0)
            + DX/2 * (- ((delta_bc_velocity(data, 0.0, vel_y, z, t, 1)
                        - delta_bc_velocity(data, 0.0, vel_y - DY, z, t, 1)) * DY_INVERSE)
                     - ((delta_bc_velocity(data, 0.0, y, vel_z, t, 2)
                        - delta_bc_velocity(data, 0.0, y, vel_z - DZ, t, 2)) * DZ_INVERSE));
        *vy = delta_bc_velocity(data, 0.0, vel_y, z, t, 1);
        *vz = delta_bc_velocity(data, 0.0, y, vel_z, t, 2);
    } else if (j == 0) {
        // (i, 0, k)
        *vx = delta_bc_velocity(data, vel_x, 0.0, z, t, 0);
        *vy = delta_bc_velocity(data, x, 0.0, z, t, 1)
            + DY/2 * (- ((delta_bc_velocity(data, vel_x, 0.0, z, t, 0)
                        - delta_bc_velocity(data, vel_x - DX, 0.0, z, t, 0)) * DX_INVERSE)
                     - ((delta_bc_velocity(data, x, 0.0, vel_z, t, 2)
                        - delta_bc_velocity(data, x, 0.0, vel_z - DZ, t, 2)) * DZ_INVERSE));
        *vz = delta_bc_velocity(data, x, 0.0, vel_z, t, 2);
    } else if (k == 0) {
        // (i, j, 0)
        *vx = delta_bc_velocity(data, vel_x, y, 0.0, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, 0.0, t, 1);
        *vz = delta_bc_velocity(data, x, y, 0.0, t, 2)
            + DZ/2 * (- ((delta_bc_velocity(data, vel_x, y, 0.0, t, 0)
                        - delta_bc_velocity(data, vel_x - DX, y, 0.0, t, 0)) * DX_INVERSE)
                     - ((delta_bc_velocity(data, x, vel_y, 0.0, t, 1)
                        - delta_bc_velocity(data, x, vel_y - DY, 0.0, t, 1)) * DY_INVERSE));
    } else if (i == WIDTH-1 && j == HEIGHT-1 && k == DEPTH-1) {
        // right corner: z-face loop runs last
        *vx = delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        *vz = delta_bc_velocity(data, x, y, vel_z, t, 2);
    } else if (i == WIDTH-1 && j == HEIGHT-1) {
        // right edge x-y: y-face loop runs last
        *vx = delta_bc_velocity(data, vel_x, vel_y, z, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, z, t, 1);
        *vz = delta_bc_velocity(data, x, vel_y, vel_z, t, 2);
    } else if (i == WIDTH-1 && k == DEPTH-1) {
        // right edge x-z: z-face loop runs last
        *vx = delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        *vz = delta_bc_velocity(data, x, y, vel_z, t, 2);
    } else if (j == HEIGHT-1 && k == DEPTH-1) {
        // right edge y-z: z-face loop runs last
        *vx = delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        *vz = delta_bc_velocity(data, x, y, vel_z, t, 2);
    } else if (i == WIDTH-1) {
        // (WIDTH-1, j, k)
        *vx = delta_bc_velocity(data, vel_x, y, z, t, 0);
        *vy = delta_bc_velocity(data, vel_x, vel_y, z, t, 1);
        *vz = delta_bc_velocity(data, vel_x, y, vel_z, t, 2);
    } else if (j == HEIGHT-1) {
        // (i, HEIGHT-1, k)
        *vx = delta_bc_velocity(data, vel_x, vel_y, z, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, z, t, 1);
        *vz = delta_bc_velocity(data, x, vel_y, vel_z, t, 2);
    } else if (k == DEPTH-1) {
        // (i, j, DEPTH-1)
        *vx = delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        *vy = delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        *vz = delta_bc_velocity(data, x, y, vel_z, t, 2);
    } else {
        *vx = *vy = *vz = 0.0;
    }
} */

DTYPE get_boundary_velocity(int i, int j, int k, int time_step, const Data *data,
                            int v_component)
{
    DTYPE t = time_step * DT;
    DTYPE x = i * DX, y = j * DY, z = k * DZ;
    DTYPE vel_x = x + DX/2, vel_y = y + DY/2, vel_z = z + DZ/2;

    if (i == 0 && j == 0 && k == 0) {
        if (v_component == 0) return delta_bc_velocity(data, DX/2, 0.0, 0.0, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, 0.0, DY/2, 0.0, t, 1);
        return delta_bc_velocity(data, 0.0, 0.0, DZ/2, t, 2);

    } else if (i == 0 && j == 0) {
        // (0, 0, k)
        if (v_component == 0) return delta_bc_velocity(data, DX/2, 0.0, z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, 0.0, DY/2, z, t, 1);
        return delta_bc_velocity(data, 0.0, 0.0, vel_z, t, 2);

    } else if (i == 0 && k == 0) {
        // (0, j, 0)
        if (v_component == 0) return delta_bc_velocity(data, DX/2, y, 0.0, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, 0.0, vel_y, 0.0, t, 1);
        return delta_bc_velocity(data, 0.0, y, DZ/2, t, 2);

    } else if (j == 0 && k == 0) {
        // (i, 0, 0)
        if (v_component == 0) return delta_bc_velocity(data, vel_x, 0.0, 0.0, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, DY/2, 0.0, t, 1);
        return delta_bc_velocity(data, x, 0.0, DZ/2, t, 2);

    } else if (i == 0) {
        // (0, j, k)
        if (v_component == 0) {
            return delta_bc_velocity(data, 0.0, y, z, t, 0)
                 + DX/2 * (- ((delta_bc_velocity(data, 0.0, vel_y, z, t, 1)
                             - delta_bc_velocity(data, 0.0, vel_y - DY, z, t, 1)) * DY_INVERSE)
                          - ((delta_bc_velocity(data, 0.0, y, vel_z, t, 2)
                             - delta_bc_velocity(data, 0.0, y, vel_z - DZ, t, 2)) * DZ_INVERSE));
        }
        if (v_component == 1) return delta_bc_velocity(data, 0.0, vel_y, z, t, 1);
        return delta_bc_velocity(data, 0.0, y, vel_z, t, 2);

    } else if (j == 0) {
        // (i, 0, k)
        if (v_component == 0) return delta_bc_velocity(data, vel_x, 0.0, z, t, 0);
        if (v_component == 1) {
            return delta_bc_velocity(data, x, 0.0, z, t, 1)
                 + DY/2 * (- ((delta_bc_velocity(data, vel_x, 0.0, z, t, 0)
                             - delta_bc_velocity(data, vel_x - DX, 0.0, z, t, 0)) * DX_INVERSE)
                          - ((delta_bc_velocity(data, x, 0.0, vel_z, t, 2)
                             - delta_bc_velocity(data, x, 0.0, vel_z - DZ, t, 2)) * DZ_INVERSE));
        }
        return delta_bc_velocity(data, x, 0.0, vel_z, t, 2);

    } else if (k == 0) {
        // (i, j, 0)
        if (v_component == 0) return delta_bc_velocity(data, vel_x, y, 0.0, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, 0.0, t, 1);
        return delta_bc_velocity(data, x, y, 0.0, t, 2)
             + DZ/2 * (- ((delta_bc_velocity(data, vel_x, y, 0.0, t, 0)
                         - delta_bc_velocity(data, vel_x - DX, y, 0.0, t, 0)) * DX_INVERSE)
                      - ((delta_bc_velocity(data, x, vel_y, 0.0, t, 1)
                         - delta_bc_velocity(data, x, vel_y - DY, 0.0, t, 1)) * DY_INVERSE));

    } else if (i == WIDTH-1 && j == HEIGHT-1 && k == DEPTH-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        return delta_bc_velocity(data, x, y, vel_z, t, 2);

    } else if (i == WIDTH-1 && j == HEIGHT-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, vel_y, z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, z, t, 1);
        return delta_bc_velocity(data, x, vel_y, vel_z, t, 2);

    } else if (i == WIDTH-1 && k == DEPTH-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        return delta_bc_velocity(data, x, y, vel_z, t, 2);

    } else if (j == HEIGHT-1 && k == DEPTH-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        return delta_bc_velocity(data, x, y, vel_z, t, 2);

    } else if (i == WIDTH-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, y, z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, vel_x, vel_y, z, t, 1);
        return delta_bc_velocity(data, vel_x, y, vel_z, t, 2);

    } else if (j == HEIGHT-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, vel_y, z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, z, t, 1);
        return delta_bc_velocity(data, x, vel_y, vel_z, t, 2);

    } else if (k == DEPTH-1) {
        if (v_component == 0) return delta_bc_velocity(data, vel_x, y, vel_z, t, 0);
        if (v_component == 1) return delta_bc_velocity(data, x, vel_y, vel_z, t, 1);
        return delta_bc_velocity(data, x, y, vel_z, t, 2);
    }

    return 0.0;
}