#include <stdio.h>
#include <stdlib.h>


/** 
 * We want to calculate 
 *                 [dx]    f_x   - Grad_x(P) - c * U_x + c[ Grad_xx(N_x) + Grad_yy(Z_x) + Grad_zz(U_x)]
 *            G:   [dy] =  f_y   - Grad_y(P) - c * U_y + c[ Grad_xx(N_y) + Grad_yy(Z_y) + Grad_zz(U_y)]
 *                 [dz]    f_z   - Grad_z(P) - c * U_z + c[ Grad_xx(N_z) + Grad_yy(Z_z) + Grad_zz(U_z)] 
 * */   



#define WIDTH 10
#define HEIGHT 10
#define DEPTH 10

#define DTYPE double
#define FIELD_SIZE (WIDTH+2)* (HEIGHT+2) * (DEPTH+2)


typedef struct {
    DTYPE *U_x;
    DTYPE *U_y;
    DTYPE *U_z;
} VelocityField;



/**             k=0    
 *              *   *   *   *   
 *          j=1  *   *   *   *
 *              *   *   *   *
 *              *   *   *   *   
 *                                i         k
 *                                      *   *   *   *   
 *                                  j   *   *   *   *
 *                                      *   *   *   *
 *                                      *   *   *   *   
 * 
 *  */ 
static inline __attribute__((always_inline)) size_t rowmaj_idx(size_t i, size_t j, size_t k)
{
    size_t face_size = WIDTH * HEIGHT;
    return i * face_size + j * WIDTH + k;
}

static void rand_fill(DTYPE *v_component){
    for(size_t i = 0; i < FIELD_SIZE; i++){
        v_component[i] = ((DTYPE) rand()) / RAND_MAX;
    }
}

void initialize_velocity_field(VelocityField *v_field){
    
    v_field->U_x = (DTYPE*) malloc(FIELD_SIZE );
    v_field->U_y = (DTYPE*) malloc(FIELD_SIZE);
    v_field->U_z = (DTYPE*) malloc(FIELD_SIZE);

    memset(v_field->U_x, 0, FIELD_SIZE);
    memset(v_field->U_y, 0, FIELD_SIZE);
    memset(v_field->U_z, 0, FIELD_SIZE);

    rand_fill(v_field->U_x);
    rand_fill(v_field->U_y);
    rand_fill(v_field->U_z);
}

/* Compute laplacian for the three components on the x direction */
void comp_laplacian_XX(VelocityField *v_field){

    // First component U_x
    for(int i = 0; i < DEPTH; i++){
        for(int j = 0; j < HEIGHT; j++){
            for(int k = 0; k < WIDTH; k++){
                size_t idx = rowmaj_idx(i, j, k);

                DTYPE value = v_field->U_x[idx];

                // It doesn't check boundaries point,


                


            }
        }
    }


}

/* Free allocated memory for the three components of velocity */
void free_velocity_field(VelocityField *v_field){
    free(v_field->U_x);
    free(v_field->U_y);
    free(v_field->U_z);
}

int main(){

    VelocityField *U;
    VelocityField *Eta;
    VelocityField *Zeta;
    initialize_velocity_field(U);
    initialize_velocity_field(Eta);
    initialize_velocity_field(Zeta);






    free_velocity_field(U);
    free_velocity_field(Eta);
    free_velocity_field(Zeta);
    return 0;
}












