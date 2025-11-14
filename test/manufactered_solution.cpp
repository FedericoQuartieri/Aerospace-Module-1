#include <gtest/gtest.h>
#include "constants.h"
#include "utils.h"

// Include C headers in extern "C" block for C++ compatibility
extern "C"
{
#include "g_field.h"
#include "force_field.h"
#include "velocity_field.h"
#include "pressure.h"
}

TEST(GFieldComputationTest, ComplexFullFieldWithViscousTerms)
{

    ForceField f_field;
    initialize_force_field(&f_field);

    // Initilize pressure
    Pressure pressure;
    initialize_pressure(&pressure);
    rand_fill(pressure.p);

    // Inizialize the 3 velocity field
    VelocityField Eta;
    VelocityField Zeta;
    VelocityField U;
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);

    // Set K
    DTYPE *K = (DTYPE *)malloc(GRID_SIZE);
    rand_fill(K);

    DTYPE *Beta = (DTYPE *)malloc(GRID_SIZE);
    DTYPE *Gamma = (DTYPE *)malloc(GRID_SIZE);
    for (int k = 0; k < DEPTH; k++)
    {
        for (int j = 0; j < HEIGHT; j++)
        {
            for (int i = 0; i < WIDTH; i++)
            {
                size_t idx = rowmaj_idx(i, j, k);
                Beta[idx] = 1 + (DT * NU) / (2 * K[idx]);
                Gamma[idx] = (DT * NU) / (2 * Beta[idx]);
            }
        }
    }

    // Inizialize G
    GField g_field;
    initialize_g_field(&g_field);

    /**
     * Compute G as:
     *                 [dx]    f_x   - Grad_x(P) - c * U_x + c[ Grad_xx(N_x) + Grad_yy(Z_x) + Grad_zz(U_x)]
     *            G:   [dy] =  f_y   - Grad_y(P) - c * U_y + c[ Grad_xx(N_y) + Grad_yy(Z_y) + Grad_zz(U_y)]
     *                 [dz]    f_z   - Grad_z(P) - c * U_z + c[ Grad_xx(N_z) + Grad_yy(Z_z) + Grad_zz(U_z)]
     * */
    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);

    solve(g_field, f_field, pressure, K, Eta, Zeta, U, Beta, Gamma, frequency = 1);


    

    

    printf("momentum\n");

    free(K);
    free_force_field(&f_field);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);
}
