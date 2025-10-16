#include <cmath>
#include <gtest/gtest.h>

extern "C" {
#include "g_field.h"
#include "force_field.h"
#include "pressure.h"
#include "velocity_field.h"
#include "constants.h"
#include "utils.h"
}

static double two_pi = 2.0 * M_PI;

// funzioni analitiche (dominio unitario x=i*dx)
static DTYPE ux_analytic(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(sin(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}
static DTYPE uy_analytic(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(cos(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}
static DTYPE uz_analytic(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(sin(two_pi*x)*cos(two_pi*y)*sin(two_pi*z));
}
static DTYPE p_analytic(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(sin(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

TEST(GFieldTest, ManufacturedSolution) {
    // grid spacing: assume uniform unit cube
    DTYPE dx = 0.1;

    // allocate and fill velocity fields and pressure and K,f as required
    VelocityField Eta, Zeta, U;
    initialize_velocity_field(&Eta);
    initialize_velocity_field(&Zeta);
    initialize_velocity_field(&U);

    Pressure pressure;
    initialize_pressure(&pressure);

    ForceField f_field;
    initialize_force_field(&f_field);

    // K choose zero to simplify (so term (NU/2 * K) * U vanishes)
    DTYPE *K = (DTYPE*) malloc(GRID_SIZE);
    for(size_t idx=0; idx<GRID_SIZE/sizeof(DTYPE); ++idx) K[idx] = 0.0;

    // fill fields (including boundaries; compute_g reads interior 1..N-1)
    for(int k=0;k<DEPTH;k++){
      for(int j=0;j<HEIGHT;j++){
        for(int i=0;i<WIDTH;i++){
          size_t idx = rowmaj_idx(i,j,k);
          U.v_x[idx] = ux_analytic(i,j,k,dx);
          U.v_y[idx] = uy_analytic(i,j,k,dx);
          U.v_z[idx] = uz_analytic(i,j,k,dx);

          Eta.v_x[idx] = U.v_x[idx];
          Eta.v_y[idx] = U.v_y[idx];
          Eta.v_z[idx] = U.v_z[idx];

          Zeta.v_x[idx] = U.v_x[idx];
          Zeta.v_y[idx] = U.v_y[idx];
          Zeta.v_z[idx] = U.v_z[idx];

          pressure.p[idx] = p_analytic(i,j,k,dx);

          // set forcing to zero (we test gradient + viscous parts)
          f_field.f_x[idx] = 0.0;
          f_field.f_y[idx] = 0.0;
          f_field.f_z[idx] = 0.0;
        }
      }
    }

    GField g_field;
    initialize_g_field(&g_field);

    compute_g(&g_field, &f_field, &pressure, K, &Eta, &Zeta, &U);

    // DIAGNOSTICA: formato rapido
    {
      size_t center = rowmaj_idx(WIDTH/2, HEIGHT/2, DEPTH/2);
      double gx = (double) g_field.g_x[center];
      double expected_gx_center;
      {
        DTYPE x = (WIDTH/2) * dx, y = (HEIGHT/2) * dx, z = (DEPTH/2) * dx;
        DTYPE dpdx = (DTYPE)(two_pi * cos(two_pi*x) * sin(two_pi*y) * sin(two_pi*z));
        DTYPE lap_x_from_eta = (DTYPE)(-two_pi*two_pi * Eta.v_x[center]);
        DTYPE lap_y_from_zeta = (DTYPE)(-two_pi*two_pi * Zeta.v_x[center]);
        DTYPE lap_z_from_u = (DTYPE)(-two_pi*two_pi * U.v_x[center]);
        expected_gx_center = (double)( - dpdx + (NU/2.0) * (lap_x_from_eta + lap_y_from_zeta + lap_z_from_u) );
      }
      std::cout << "DIAG: sizeof(DTYPE)=" << sizeof(DTYPE)
                << " DBL_EPS=" << DBL_EPSILON << "\n";
      std::cout << "DIAG: center gx=" << gx
                << " expected=" << expected_gx_center
                << " absdiff=" << fabs(gx-expected_gx_center)
                << " reldiff=" << (fabs(expected_gx_center) > 0 ? fabs(gx-expected_gx_center)/fabs(expected_gx_center) : 0.0)
                << "\n";
    }

    // compute analytic g and compare at interior points (same loop indices used in compute_g)
    DTYPE maxerr = 0.0;
    for(int k=1; k<DEPTH; ++k){
      for(int j=1; j<HEIGHT; ++j){
        for(int i=1; i<WIDTH; ++i){
          size_t idx = rowmaj_idx(i,j,k);

          // analytic derivatives:
          // dp/dx = 2π cos(2πx) sin(2πy) sin(2πz)
          DTYPE x = i*dx, y = j*dx, z = k*dx;
          DTYPE dpdx = (DTYPE)(two_pi * cos(two_pi*x) * sin(two_pi*y) * sin(two_pi*z));
          DTYPE dpdy = (DTYPE)(two_pi * sin(two_pi*x) * cos(two_pi*y) * sin(two_pi*z));
          DTYPE dpdz = (DTYPE)(two_pi * sin(two_pi*x) * sin(two_pi*y) * cos(two_pi*z));

          // second derivatives of the chosen U (∂xx of sin(2πx)*sin(2πy)*sin(2πz) = -(2π)^2 * same function)
          DTYPE lap_x_from_eta = (DTYPE)(-two_pi*two_pi * Eta.v_x[idx]); // ∂xx Eta_x
          // but careful: compute_velocity_yy_grad(Zeta->v_x,...) means ∂yy of Zeta.v_x
          DTYPE lap_y_from_zeta = (DTYPE)(-two_pi*two_pi * Zeta.v_x[idx]); // ∂yy of Zeta_x
          DTYPE lap_z_from_u = (DTYPE)(-two_pi*two_pi * U.v_x[idx]); // ∂zz of U_x

          DTYPE expected_gx = /* f_x = 0 */ - dpdx - /* K term zero */ 0.0 + (NU/2.0) * (lap_x_from_eta + lap_y_from_zeta + lap_z_from_u);

          DTYPE err = fabs(g_field.g_x[idx] - expected_gx);
          if(err > maxerr) maxerr = err;

          // similarly can check g_y and g_z (omitted for brevity)
        }
      }
    }
    // tolerance: depends on FD stencil order; for central 2nd order pick tol ~ C * dx^2
    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    EXPECT_LT(maxerr, tol) << "max error = " << maxerr;

    std::cout << "GFieldTest.ManufacturedSolution: max error = " << maxerr << " (tolerance " << tol << ")\n";
    // cleanup
    free_g_field(&g_field);
    free(K);
    free_force_field(&f_field);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
}