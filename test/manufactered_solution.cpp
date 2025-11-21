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
#include "solve.h"
#include <math.h>
}
#include <math.h>

/* ----- Parameters (tune as you wish) ----- */

static const double Re       = 100.0;  /* Reynolds number */
static const double k_over_nu = 0.0;   /* k/nu term in the PDE, set if needed */

/* ----- Basic data structures ----- */

typedef struct {
    double x;
    double y;
    double z;
} Vec3;

typedef struct {
    double xx, xy, xz;
    double yx, yy, yz;
    double zx, zy, zz;
} Mat3;

/* ----- Exact solution u(x,y,z,t) ----- */
/* u = sin(t) * [ sin x sin y sin z,
 *                cos x cos y cos z,
 *                cos x sin y (cos z + sin z) ]
 */

Vec3 u_exact(double x, double y, double z, double t)
{
    Vec3 u;

    double sx = sin(x), cx = cos(x);
    double sy = sin(y), cy = cos(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t);

    u.x = st * sx * sy * sz;
    u.y = st * cx * cy * cz;
    u.z = st * cx * sy * (cz + sz);

    return u;
}

/* ----- Time derivative du/dt ----- */
/* du/dt = cos(t) * [ sin x sin y sin z,
 *                    cos x cos y cos z,
 *                    cos x sin y (cos z + sin z) ]
 */

Vec3 du_dt_exact(double x, double y, double z, double t)
{
    Vec3 dudt;

    double sx = sin(x), cx = cos(x);
    double sy = sin(y), cy = cos(y);
    double sz = sin(z), cz = cos(z);
    double ct = cos(t);

    dudt.x = ct * sx * sy * sz;
    dudt.y = ct * cx * cy * cz;
    dudt.z = ct * cx * sy * (cz + sz);

    return dudt;
}

/* ----- Gradient of u: ∇u (3×3 tensor) ----- */
/* (∇u)_{ij} = ∂u_i / ∂x_j with x_j = (x,y,z)
 *
 * u_i = sin(t) * phi_i(x,y,z), so ∂u_i/∂x_j = sin(t) * ∂phi_i/∂x_j
 *
 * phi1 = sin x sin y sin z
 * phi2 = cos x cos y cos z
 * phi3 = cos x sin y (cos z + sin z)
 */

Mat3 grad_u_exact(double x, double y, double z, double t)
{
    Mat3 G;

    double sx = sin(x), cx = cos(x);
    double sy = sin(y), cy = cos(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t);

    /* Derivatives of phi1 = sx * sy * sz */
    double dphi1_dx = cx * sy * sz;
    double dphi1_dy = sx * cy * sz;
    double dphi1_dz = sx * sy * cz;

    /* Derivatives of phi2 = cx * cy * cz */
    double dphi2_dx = -sx * cy * cz;
    double dphi2_dy = -cx * sy * cz;
    double dphi2_dz = -cx * cy * sz;

    /* phi3 = cx * sy * (cz + sz) */
    double h       = cz + sz;
    double dh_dz   = -sz + cz;

    double dphi3_dx = -sx * sy * h;
    double dphi3_dy =  cx * cy * h;
    double dphi3_dz =  cx * sy * dh_dz;

    /* Multiply by sin(t) to get ∂u_i/∂x_j */
    G.xx = st * dphi1_dx;
    G.xy = st * dphi1_dy;
    G.xz = st * dphi1_dz;

    G.yx = st * dphi2_dx;
    G.yy = st * dphi2_dy;
    G.yz = st * dphi2_dz;

    G.zx = st * dphi3_dx;
    G.zy = st * dphi3_dy;
    G.zz = st * dphi3_dz;

    return G;
}

/* ----- Pressure p(x,y,z,t) ----- */
/* p = - (3/Re) * sin(t) * cos x * sin y * (sin z - cos z) */

double p_exact(double x, double y, double z, double t)
{
    double sx = sin(x), cx = cos(x);
    double sy = sin(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t);

    double factor = -3.0 / Re;

    double p = factor * st * cx * sy * (sz - cz);
    return p;
}

/* ----- Gradient of p: ∇p ----- */

Vec3 grad_p_exact(double x, double y, double z, double t)
{
    Vec3 gp;

    double sx = sin(x), cx = cos(x);
    double sy = sin(y), cy = cos(y);
    double sz = sin(z), cz = cos(z);
    double st = sin(t);

    double factor = -3.0 / Re;

    /* g(x,y,z) = cx * sy * (sz - cz)
       p = factor * st * g
       ∂g/∂x = -sx * sy * (sz - cz)
       ∂g/∂y =  cx * cy * (sz - cz)
       ∂g/∂z =  cx * sy * (cz + sz)
    */

    double dpdx = -sx * sy * (sz - cz);
    double dpdy =  cx * cy * (sz - cz);
    double dpdz =  cx * sy * (cz + sz);

    gp.x = factor * st * dpdx;
    gp.y = factor * st * dpdy;
    gp.z = factor * st * dpdz;

    return gp;
}

/* ----- Laplacian of u: ∇² u ----- */
/* For this manufactured solution we have ∇² u = -3 u */

Vec3 laplace_u_exact(double x, double y, double z, double t)
{
    Vec3 u = u_exact(x, y, z, t);
    Vec3 Lu;

    Lu.x = -3.0 * u.x;
    Lu.y = -3.0 * u.y;
    Lu.z = -3.0 * u.z;

    return Lu;
}

/* ----- Forcing term f(x,y,z,t) ----- */
/* PDE: du/dt - (1/Re) ∇²u + (k/nu) u + ∇p = f
 *
 * Using ∇²u = -3u:
 * f = du/dt + (3/Re) u + (k/nu) u + ∇p
 */

Vec3 f_exact(double x, double y, double z, double t)
{
    Vec3 dudt = du_dt_exact(x, y, z, t);
    Vec3 u    = u_exact(x, y, z, t);
    Vec3 gp   = grad_p_exact(x, y, z, t);

    Vec3 f;
    double coeff_u = 3.0 / Re + k_over_nu;

    f.x = dudt.x + coeff_u * u.x + gp.x;
    f.y = dudt.y + coeff_u * u.y + gp.y;
    f.z = dudt.z + coeff_u * u.z + gp.z;

    return f;
}


TEST(ManufacturedSolution, ManufacturedSolutionOnVelocitySystem)
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

    int write_frequency = 10;
    solve(g_field, f_field, pressure, K, Eta, Zeta, U, Beta, Gamma, write_frequency);


    printf("momentum\n");

    free(K);
    free_force_field(&f_field);
    free_pressure(&pressure);
    free_velocity_field(&Eta);
    free_velocity_field(&Zeta);
    free_velocity_field(&U);
    free_g_field(&g_field);
}

