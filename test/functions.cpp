#include <cmath>
#include <gtest/gtest.h>
#include <random>

extern "C" {
#include "g_field.h"
#include "force_field.h"
#include "pressure.h"
#include "velocity_field.h"
#include "constants.h"
#include "utils.h"
}

static double two_pi = 2.0 * M_PI;
static double four_pi_square = 4.0 * M_PI * M_PI;

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

static DTYPE p_analytic_gradx(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(two_pi*cos(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE p_analytic_grady(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(two_pi*sin(two_pi*x)*cos(two_pi*y)*sin(two_pi*z));
}

static DTYPE p_analytic_gradz(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(two_pi*sin(two_pi*x)*sin(two_pi*y)*cos(two_pi*z));
}

static DTYPE ux_analytic_gradxx(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*sin(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE ux_analytic_gradyy(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*sin(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE ux_analytic_gradzz(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*sin(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE uy_analytic_gradxx(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*cos(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE uy_analytic_gradyy(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(four_pi_square*cos(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE uy_analytic_gradzz(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(four_pi_square*cos(two_pi*x)*sin(two_pi*y)*sin(two_pi*z));
}

static DTYPE uz_analytic_gradxx(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*sin(two_pi*x)*cos(two_pi*y)*sin(two_pi*z));
}

static DTYPE uz_analytic_gradyy(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*sin(two_pi*x)*cos(two_pi*y)*sin(two_pi*z));
}

static DTYPE uz_analytic_gradzz(int i,int j,int k, DTYPE dx){
    DTYPE x = i * dx; DTYPE y = j * dx; DTYPE z = k * dx;
    return (DTYPE)(-four_pi_square*sin(two_pi*x)*cos(two_pi*y)*sin(two_pi*z));
}



TEST(PressureGradientTest, DirectionX){
    DTYPE dx = 0.001;
    DTYPE computed_result;
    DTYPE analytical_result;

    Pressure pressure;
    initialize_pressure(&pressure);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_int_distribution<> distrib_i(1, WIDTH - 1);
    int i = distrib_i(gen);

    std::uniform_int_distribution<> distrib_j(1, HEIGHT - 1);
    int j = distrib_j(gen);
    
    std::uniform_int_distribution<> distrib_k(1, DEPTH - 1);
    int k = distrib_j(gen);

    size_t idx = rowmaj_idx(i,j,k);
    size_t idx_neighbour = rowmaj_idx(i + 1,j,k);

    pressure.p[idx] = p_analytic(i,j,k,dx);
    pressure.p[idx_neighbour] = p_analytic(i+1,j,k,dx);


    computed_result = compute_pressure_x_grad(pressure.p,i,j,k);

    analytical_result = p_analytic_gradx(i, j, k, dx);

    
    DTYPE err = fabs(computed_result - analytical_result);

    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    std::cout <<computed_result << " " << analytical_result << std::endl;
    EXPECT_LT(err, tol) << "max error = " << err;
    std::cout << "PressureGradientTest.DirectionX: max error = " << err << " (tolerance " << tol << ")\n";
    
    free_pressure(&pressure);
}

TEST(PressureGradientTest, DirectionY){
    DTYPE dy = 0.001;
    DTYPE computed_result;
    DTYPE analytical_result;

    Pressure pressure;
    initialize_pressure(&pressure);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_int_distribution<> distrib_i(1, WIDTH - 1);
    int i = distrib_i(gen);

    std::uniform_int_distribution<> distrib_j(1, HEIGHT - 1);
    int j = distrib_j(gen);
    
    std::uniform_int_distribution<> distrib_k(1, DEPTH - 1);
    int k = distrib_j(gen);

    size_t idx = rowmaj_idx(i,j,k);
    size_t idx_neighbour = rowmaj_idx(i,j+1,k);

    pressure.p[idx] = p_analytic(i,j,k,dy);
    pressure.p[idx_neighbour] = p_analytic(i,j+1,k,dy);


    computed_result = compute_pressure_y_grad(pressure.p,i,j,k);

    analytical_result = p_analytic_grady(i, j, k, dy);

    
    DTYPE err = fabs(computed_result - analytical_result);

    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    std::cout <<computed_result << " " << analytical_result << std::endl;
    EXPECT_LT(err, tol) << "max error = " << err;
    std::cout << "PressureGradientTest.DirectionY: max error = " << err << " (tolerance " << tol << ")\n";
    
    free_pressure(&pressure);
}

TEST(PressureGradientTest, DirectionZ){
    DTYPE dz = 0.001;
    DTYPE computed_result;
    DTYPE analytical_result;

    Pressure pressure;
    initialize_pressure(&pressure);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_int_distribution<> distrib_i(1, WIDTH - 1);
    int i = distrib_i(gen);

    std::uniform_int_distribution<> distrib_j(1, HEIGHT - 1);
    int j = distrib_j(gen);
    
    std::uniform_int_distribution<> distrib_k(1, DEPTH - 1);
    int k = distrib_j(gen);

    size_t idx = rowmaj_idx(i,j,k);
    size_t idx_neighbour = rowmaj_idx(i,j,k+1);

    pressure.p[idx] = p_analytic(i,j,k,dz);
    pressure.p[idx_neighbour] = p_analytic(i,j,k+1,dz);


    computed_result = compute_pressure_z_grad(pressure.p,i,j,k);

    analytical_result = p_analytic_gradz(i, j, k, dz);

    
    DTYPE err = fabs(computed_result - analytical_result);

    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    std::cout <<computed_result << " " << analytical_result << std::endl;
    EXPECT_LT(err, tol) << "max error = " << err;
    std::cout << "PressureGradientTest.DirectionZ: max error = " << err << " (tolerance " << tol << ")\n";
    
    free_pressure(&pressure);
}

TEST(VelocitySecondDerivativeTest, DirectionX){
    DTYPE dx = 0.001;
    DTYPE dy = 0.001;
    DTYPE dz = 0.001;
    DTYPE computed_resultxx;
    DTYPE analytical_resultxx;
    DTYPE computed_resultyy;
    DTYPE analytical_resultyy;
    DTYPE computed_resultzz;
    DTYPE analytical_resultzz;

    VelocityField U;
    initialize_velocity_field(&U);
    rand_fill_velocity_field(&U);

    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::uniform_int_distribution<> distrib_i(1, WIDTH - 1);
    int i = distrib_i(gen);

    std::uniform_int_distribution<> distrib_j(1, HEIGHT - 1);
    int j = distrib_j(gen);
    
    std::uniform_int_distribution<> distrib_k(1, DEPTH - 1);
    int k = distrib_j(gen);

    size_t idx = rowmaj_idx(i,j,k);
    
    size_t idx_neighbour_x_before = rowmaj_idx(i-1,j,k);
    size_t idx_neighbour_x_after = rowmaj_idx(i+1,j,k);

    size_t idx_neighbour_y_before = rowmaj_idx(i,j-1,k);
    size_t idx_neighbour_y_after = rowmaj_idx(i,j+1,k);
    
    size_t idx_neighbour_z_before = rowmaj_idx(i,j,k-1);
    size_t idx_neighbour_z_after = rowmaj_idx(i,j,k+1);


    U.v_x[idx] = ux_analytic(i,j,k,dx);
    U.v_x[idx_neighbour_x_before] = ux_analytic(i-1,j,k,dx);
    U.v_x[idx_neighbour_x_after] = ux_analytic(i+1,j,k,dx);
    U.v_x[idx_neighbour_y_before] = ux_analytic(i,j-1,k,dx);
    U.v_x[idx_neighbour_y_after] = ux_analytic(i,j+1,k,dx);
    U.v_x[idx_neighbour_z_before] = ux_analytic(i,j,k-1,dx);
    U.v_x[idx_neighbour_z_after] = ux_analytic(i,j,k+1,dx);

    



    computed_resultxx = compute_velocity_xx_grad(U.v_x,i,j,k);
    analytical_resultxx = ux_analytic_gradxx(i, j, k, dx);

    computed_resultyy = compute_velocity_yy_grad(U.v_x,i,j,k);
    analytical_resultyy = ux_analytic_gradyy(i, j, k, dx);

    computed_resultzz = compute_velocity_zz_grad(U.v_x,i,j,k);
    analytical_resultzz = ux_analytic_gradzz(i, j, k, dx);

    
    DTYPE errx = fabs(computed_resultxx - analytical_resultxx);
    DTYPE erry = fabs(computed_resultyy - analytical_resultyy);
    DTYPE errz = fabs(computed_resultzz - analytical_resultzz);

    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    EXPECT_LT(errx, tol) << "max error = " << errx;
    std::cout << "VelocitySecondDerivativeTest.DirectionX: max error = " << errx << " (tolerance " << tol << ")\n";
    EXPECT_LT(erry, tol) << "max error = " << erry;
    std::cout << "VelocitySecondDerivativeTest.DirectionX: max error = " << erry << " (tolerance " << tol << ")\n";
    EXPECT_LT(errz, tol) << "max error = " << errz;
    std::cout << "VelocitySecondDerivativeTest.DirectionX: max error = " << errz << " (tolerance " << tol << ")\n";


    free_velocity_field(&U);
}


TEST(PressureGradientTestAllPoints, DirectionX){
    DTYPE dx = 0.001;
    DTYPE computed_result;
    DTYPE analytical_result;

    Pressure pressure;
    initialize_pressure(&pressure);

    DTYPE max_err = 0.0;
    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    DTYPE current_err;

    for(int k=1; k<DEPTH - 1; ++k){
      for(int j=1; j<HEIGHT - 1; ++j){
        for(int i=1; i<WIDTH - 1; ++i){
            
            size_t idx = rowmaj_idx(i,j,k);
            size_t idx_neighbour = rowmaj_idx(i + 1,j,k);
           
            pressure.p[idx] = p_analytic(i,j,k,dx);
            pressure.p[idx_neighbour] = p_analytic(i+1,j,k,dx);
            
            computed_result = compute_pressure_x_grad(pressure.p,i,j,k);
            analytical_result = p_analytic_gradx(i, j, k, dx);
            current_err = fabs(computed_result - analytical_result);

            if(current_err > max_err){
                max_err = current_err;
            }
        }
      }
    }
    EXPECT_LT(max_err, tol) << "max error = " << max_err;
    std::cout << "PressureGradientTest.DirectionX: max error = " << max_err << " (tolerance " << tol << ")\n";
    
    free_pressure(&pressure);
}

TEST(PressureGradientTestAllPoints, DirectionY){
    DTYPE dy = 0.001;
    DTYPE computed_result;
    DTYPE analytical_result;

    Pressure pressure;
    initialize_pressure(&pressure);

    DTYPE max_err = 0.0;
    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    DTYPE current_err;

    for(int k=1; k<DEPTH - 1; ++k){
      for(int j=1; j<HEIGHT - 1; ++j){
        for(int i=1; i<WIDTH - 1; ++i){
            
            size_t idx = rowmaj_idx(i,j,k);
            size_t idx_neighbour = rowmaj_idx(i,j+1,k);
           
            pressure.p[idx] = p_analytic(i,j,k,dy);
            pressure.p[idx_neighbour] = p_analytic(i,j+1,k,dy);
            
            computed_result = compute_pressure_y_grad(pressure.p,i,j,k);
            analytical_result = p_analytic_grady(i, j, k, dy);
            current_err = fabs(computed_result - analytical_result);

            if(current_err > max_err){
                max_err = current_err;
            }
        }
      }
    }
    EXPECT_LT(max_err, tol) << "max error = " << max_err;
    std::cout << "PressureGradientTest.DirectionY: max error = " << max_err << " (tolerance " << tol << ")\n";
    
    free_pressure(&pressure);
}


TEST(PressureGradientTestAllPoints, DirectionZ){
    DTYPE dz = 0.001;
    DTYPE computed_result;
    DTYPE analytical_result;

    Pressure pressure;
    initialize_pressure(&pressure);

    DTYPE max_err = 0.0;
    DTYPE tol = 1e-5; // possibile adattare in base a WIDTH
    DTYPE current_err;

    for(int k=1; k<DEPTH - 1; ++k){
      for(int j=1; j<HEIGHT - 1; ++j){
        for(int i=1; i<WIDTH - 1; ++i){
            
            size_t idx = rowmaj_idx(i,j,k);
            size_t idx_neighbour = rowmaj_idx(i,j,k+1);
           
            pressure.p[idx] = p_analytic(i,j,k,dz);
            pressure.p[idx_neighbour] = p_analytic(i,j,k+1,dz);
            
            computed_result = compute_pressure_z_grad(pressure.p,i,j,k);
            analytical_result = p_analytic_gradz(i, j, k, dz);
            current_err = fabs(computed_result - analytical_result);

            if(current_err > max_err){
                max_err = current_err;
            }
        }
      }
    }
    EXPECT_LT(max_err, tol) << "max error = " << max_err;
    std::cout << "PressureGradientTest.DirectionZ: max error = " << max_err << " (tolerance " << tol << ")\n";
    
    free_pressure(&pressure);
}


TEST(VelocitySecondDerivativeTestAllPoints, DirectionX){
    DTYPE dx = 0.001;
    DTYPE dy = 0.001;
    DTYPE dz = 0.001;
    DTYPE computed_resultxx;
    DTYPE analytical_resultxx;
    DTYPE computed_resultyy;
    DTYPE analytical_resultyy;
    DTYPE computed_resultzz;
    DTYPE analytical_resultzz;

    VelocityField U;
    initialize_velocity_field(&U);
    rand_fill_velocity_field(&U);

    DTYPE tol = 1e-5;
    DTYPE max_errxx = 0.0;
    DTYPE current_errxx;
    DTYPE max_erryy = 0.0;
    DTYPE current_erryy;
    DTYPE max_errzz = 0.0;
    DTYPE current_errzz;

    for(int k=1; k<DEPTH - 1; ++k){
        for(int j=1; j<HEIGHT - 1; ++j){
            for(int i=1; i<WIDTH - 1; ++i){
                
                size_t idx = rowmaj_idx(i,j,k);
            
                size_t idx_neighbour_x_before = rowmaj_idx(i-1,j,k);
                size_t idx_neighbour_x_after = rowmaj_idx(i+1,j,k);

                size_t idx_neighbour_y_before = rowmaj_idx(i,j-1,k);
                size_t idx_neighbour_y_after = rowmaj_idx(i,j+1,k);
                
                size_t idx_neighbour_z_before = rowmaj_idx(i,j,k-1);
                size_t idx_neighbour_z_after = rowmaj_idx(i,j,k+1);
                        
                U.v_x[idx] = ux_analytic(i,j,k,dx);
                U.v_x[idx_neighbour_x_before] = ux_analytic(i-1,j,k,dx);
                U.v_x[idx_neighbour_x_after] = ux_analytic(i+1,j,k,dx);
                U.v_x[idx_neighbour_y_before] = ux_analytic(i,j-1,k,dx);
                U.v_x[idx_neighbour_y_after] = ux_analytic(i,j+1,k,dx);
                U.v_x[idx_neighbour_z_before] = ux_analytic(i,j,k-1,dx);
                U.v_x[idx_neighbour_z_after] = ux_analytic(i,j,k+1,dx);
                
                computed_resultxx = compute_velocity_xx_grad(U.v_x,i,j,k);
                analytical_resultxx = ux_analytic_gradxx(i, j, k, dx);

                computed_resultyy = compute_velocity_yy_grad(U.v_x,i,j,k);
                analytical_resultyy = ux_analytic_gradyy(i, j, k, dx);

                computed_resultzz = compute_velocity_zz_grad(U.v_x,i,j,k);
                analytical_resultzz = ux_analytic_gradzz(i, j, k, dx);
                current_errxx = fabs(computed_resultxx - analytical_resultxx);
                current_erryy = fabs(computed_resultyy - analytical_resultyy);
                current_errzz = fabs(computed_resultzz - analytical_resultzz);

                if(current_errxx > max_errxx){
                    max_errxx = current_errxx;
                }

                if(current_erryy > max_erryy){
                    max_erryy = current_erryy;
                }

                if(current_errzz > max_errzz){
                    max_errzz = current_errzz;
                }
            }
        }
    }





    






    EXPECT_LT(max_errxx, tol) << "max error = " << max_errxx;
    std::cout << "VelocitySecondDerivativeTestAllPoints.DirectionX: max error x = " << max_errxx << " (tolerance " << tol << ")\n";
    EXPECT_LT(max_erryy, tol) << "max error = " << max_erryy;
    std::cout << "VelocitySecondDerivativeTestAllPoints.DirectionX: max error y = " << max_erryy<< " (tolerance " << tol << ")\n";
    EXPECT_LT(max_errzz, tol) << "max error = " << max_errzz;
    std::cout << "VelocitySecondDerivativeTestAllPoints.DirectionX: max error z = " << max_errzz << " (tolerance " << tol << ")\n";


    free_velocity_field(&U);
}
