#include "physics.h"
#include "solver.h"

static Real spacing_from_component(int component) {
    switch (component) {
        case 0:
            return (Real)DX;
        case 1:
            return (Real)DY;
        case 2:
            return (Real)DZ;
        default:
            fprintf(stderr, "Invalid vector component: %d\n", component);
            exit(1);
    }
}

Real beta_from_k(Real k) {
    return 1.0 + (DT * NU) / (2.0 * k);
}

Real gamma_from_k(Real k) {
    Real beta = beta_from_k(k);
    return (DT * NU) / (2.0 * beta);
}

Real time_physical_coord(Real t_step) {
    return t_step * (Real)DT;
}

Real centered_physical_coord(int index, int component) {
    return (Real)index * spacing_from_component(component);
}

Real staggered_physical_coord(int index, int component) {
    return ((Real)index + 0.5) * spacing_from_component(component);
}

static inline Real boundary_increment(VectorFunction bc_velocity,
                                      Real x, Real y, Real z,
                                      Real t, int t_step, int component) {
    const Real current = bc_velocity(x, y, z, t, component);
    if (t_step == 0) {
        return current;
    }
    return current - bc_velocity(x, y, z, t - (Real)DT, component);
}

/*
 * Return the boundary increment u(t_step) - u(t_step - 1).
 * On a lower face, the normal component is reconstructed from the
 * divergence-free constraint; tangential components are sampled directly.
 * At lower edges and corners the prescribed staggered value has priority.
 */
Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component) {
    const Real t = time_physical_coord(t_step);
    const Real x = (Real)i * (Real)DX;
    const Real y = (Real)j * (Real)DY;
    const Real z = (Real)k * (Real)DZ;
    const Real vx = x + (Real)DX / 2.0;
    const Real vy = y + (Real)DY / 2.0;
    const Real vz = z + (Real)DZ / 2.0;
    const int lower_face_count = (i == 0) + (j == 0) + (k == 0);

    if ((unsigned int)component > 2U) {
        fprintf(stderr, "Invalid vector component: %d\n", component);
        exit(1);
    }
    if (lower_face_count == 0) {
        return 0.0;
    }

#define BC_INCREMENT(px, py, pz, comp) \
    boundary_increment(bc_velocity, (px), (py), (pz), \
                       t, t_step, (comp))

    if (lower_face_count > 1) {
        switch (component) {
            case 0:
                return BC_INCREMENT(vx, y, z, 0);
            case 1:
                return BC_INCREMENT(x, vy, z, 1);
            default:
                return BC_INCREMENT(x, y, vz, 2);
        }
    }

    if (i == 0) {
        if (component == 0) {
            const Real divergence_y =
                (BC_INCREMENT(0.0, vy, z, 1) -
                 BC_INCREMENT(0.0, vy - (Real)DY, z, 1)) *
                (Real)DY_INVERSE;
            const Real divergence_z =
                (BC_INCREMENT(0.0, y, vz, 2) -
                 BC_INCREMENT(0.0, y, vz - (Real)DZ, 2)) *
                (Real)DZ_INVERSE;
            return BC_INCREMENT(0.0, y, z, 0) -
                   ((Real)DX / 2.0) * (divergence_y + divergence_z);
        }
        if (component == 1) {
            return BC_INCREMENT(0.0, vy, z, 1);
        }
        return BC_INCREMENT(0.0, y, vz, 2);
    }

    if (j == 0) {
        if (component == 0) {
            return BC_INCREMENT(vx, 0.0, z, 0);
        }
        if (component == 1) {
            const Real divergence_x =
                (BC_INCREMENT(vx, 0.0, z, 0) -
                 BC_INCREMENT(vx - (Real)DX, 0.0, z, 0)) *
                (Real)DX_INVERSE;
            const Real divergence_z =
                (BC_INCREMENT(x, 0.0, vz, 2) -
                 BC_INCREMENT(x, 0.0, vz - (Real)DZ, 2)) *
                (Real)DZ_INVERSE;
            return BC_INCREMENT(x, 0.0, z, 1) -
                   ((Real)DY / 2.0) * (divergence_x + divergence_z);
        }
        return BC_INCREMENT(x, 0.0, vz, 2);
    }

    if (component == 0) {
        return BC_INCREMENT(vx, y, 0.0, 0);
    }
    if (component == 1) {
        return BC_INCREMENT(x, vy, 0.0, 1);
    }

    {
        const Real divergence_x =
            (BC_INCREMENT(vx, y, 0.0, 0) -
             BC_INCREMENT(vx - (Real)DX, y, 0.0, 0)) *
            (Real)DX_INVERSE;
        const Real divergence_y =
            (BC_INCREMENT(x, vy, 0.0, 1) -
             BC_INCREMENT(x, vy - (Real)DY, 0.0, 1)) *
            (Real)DY_INVERSE;
        return BC_INCREMENT(x, y, 0.0, 2) -
               ((Real)DZ / 2.0) * (divergence_x + divergence_y);
    }

#undef BC_INCREMENT
}

/*
 * Upper faces use the prescribed value at the physical wall.  When a point
 * belongs to more than one upper face, Z has priority over Y, then X, matching
 * the boundary overwrite order.  Lower faces retain priority at mixed edges.
 */
Real bc_right(VectorFunction bc_velocity,
              int i, int j, int k, int t_step, int component) {
    if (i == 0 || j == 0 || k == 0) {
        return bc_left(bc_velocity, i, j, k, t_step, component);
    }

    if ((unsigned int)component > 2U) {
        fprintf(stderr, "Invalid vector component: %d\n", component);
        exit(1);
    }

    const Real t = time_physical_coord(t_step);
    const Real x = (Real)i * (Real)DX;
    const Real y = (Real)j * (Real)DY;
    const Real z = (Real)k * (Real)DZ;
    const Real vx = x + (Real)DX / 2.0;
    const Real vy = y + (Real)DY / 2.0;
    const Real vz = z + (Real)DZ / 2.0;

#define BC_INCREMENT(px, py, pz, comp) \
    boundary_increment(bc_velocity, (px), (py), (pz), \
                       t, t_step, (comp))

    if (k == DEPTH - 1) {
        switch (component) {
            case 0:
                return BC_INCREMENT(vx, y, vz, 0);
            case 1:
                return BC_INCREMENT(x, vy, vz, 1);
            default:
                return BC_INCREMENT(x, y, vz, 2);
        }
    }

    if (j == HEIGHT - 1) {
        switch (component) {
            case 0:
                return BC_INCREMENT(vx, vy, z, 0);
            case 1:
                return BC_INCREMENT(x, vy, z, 1);
            default:
                return BC_INCREMENT(x, vy, vz, 2);
        }
    }

    if (i == WIDTH - 1) {
        switch (component) {
            case 0:
                return BC_INCREMENT(vx, y, z, 0);
            case 1:
                return BC_INCREMENT(vx, vy, z, 1);
            default:
                return BC_INCREMENT(vx, y, vz, 2);
        }
    }

#undef BC_INCREMENT
    return 0.0;
}


static inline Real interior_second_derivative(const Real *restrict field,
                                              size_t index,
                                              size_t stride,
                                              Real inverse_spacing_square) {
    return (field[index - stride] -
            2.0 * field[index] +
            field[index + stride]) * inverse_spacing_square;
}

static inline Real upper_second_derivative(const Real *restrict field,
                                           size_t index,
                                           size_t stride,
                                           Real boundary_value,
                                           Real inverse_spacing_square) {
    return (field[index - stride] -
            3.0 * field[index] +
            2.0 * boundary_value) * inverse_spacing_square;
}

/*
 * Momentum source at (t_step - 1/2) DT:
 *
 *   g = f - grad(p*) - (NU / K) u
 *       + NU (Dxx(eta) + Dyy(zeta) + Dzz(u)).
 *
 * The support excludes lower faces and the upper face normal to the velocity
 * component.  Tangential upper faces use a Dirichlet ghost value.
 */
Real g_value(int i, int j, int k, int t_step, Real k_i,
             const SolverMemState *solver_mem_state,
             const Data *data, int component) {
    const size_t plane_size = (size_t)WIDTH * HEIGHT;
    const size_t index = (size_t)k * plane_size +
                         (size_t)j * WIDTH +
                         (size_t)i;
    const Real forcing_time = ((Real)t_step - 0.5) * (Real)DT;
    const Real velocity_time = ((Real)t_step - 1.0) * (Real)DT;
    const Real upper_x = ((Real)WIDTH - 0.5) * (Real)DX;
    const Real upper_y = ((Real)HEIGHT - 0.5) * (Real)DY;
    const Real upper_z = ((Real)DEPTH - 0.5) * (Real)DZ;
    Real x = (Real)i * (Real)DX;
    Real y = (Real)j * (Real)DY;
    Real z = (Real)k * (Real)DZ;
    const Real *restrict eta;
    const Real *restrict zeta;
    const Real *restrict velocity;
    const Real *restrict pressure = solver_mem_state->pressure_star.v;
    Real pressure_gradient;
    Real laplacian_x;
    Real laplacian_y;
    Real laplacian_z;

    switch (component) {
        case 0:
            if (i < 1 || i >= WIDTH - 1 ||
                j < 1 || j >= HEIGHT ||
                k < 1 || k >= DEPTH) {
                return 0.0;
            }

            eta = solver_mem_state->eta.v_x;
            zeta = solver_mem_state->zeta.v_x;
            velocity = solver_mem_state->u.v_x;
            x += (Real)DX / 2.0;

            pressure_gradient =
                (pressure[index + 1] - pressure[index]) *
                (Real)DX_INVERSE;
            laplacian_x = interior_second_derivative(
                eta, index, 1, (Real)DX_INVERSE_SQUARE);
            laplacian_y = (j == HEIGHT - 1)
                ? upper_second_derivative(
                      zeta, index, WIDTH,
                      data->bc_velocity(x, upper_y, z,
                                        velocity_time, component),
                      (Real)DY_INVERSE_SQUARE)
                : interior_second_derivative(
                      zeta, index, WIDTH, (Real)DY_INVERSE_SQUARE);
            laplacian_z = (k == DEPTH - 1)
                ? upper_second_derivative(
                      velocity, index, plane_size,
                      data->bc_velocity(x, y, upper_z,
                                        velocity_time, component),
                      (Real)DZ_INVERSE_SQUARE)
                : interior_second_derivative(
                      velocity, index, plane_size,
                      (Real)DZ_INVERSE_SQUARE);
            break;

        case 1:
            if (i < 1 || i >= WIDTH ||
                j < 1 || j >= HEIGHT - 1 ||
                k < 1 || k >= DEPTH) {
                return 0.0;
            }

            eta = solver_mem_state->eta.v_y;
            zeta = solver_mem_state->zeta.v_y;
            velocity = solver_mem_state->u.v_y;
            y += (Real)DY / 2.0;

            pressure_gradient =
                (pressure[index + WIDTH] - pressure[index]) *
                (Real)DY_INVERSE;
            laplacian_x = (i == WIDTH - 1)
                ? upper_second_derivative(
                      eta, index, 1,
                      data->bc_velocity(upper_x, y, z,
                                        velocity_time, component),
                      (Real)DX_INVERSE_SQUARE)
                : interior_second_derivative(
                      eta, index, 1, (Real)DX_INVERSE_SQUARE);
            laplacian_y = interior_second_derivative(
                zeta, index, WIDTH, (Real)DY_INVERSE_SQUARE);
            laplacian_z = (k == DEPTH - 1)
                ? upper_second_derivative(
                      velocity, index, plane_size,
                      data->bc_velocity(x, y, upper_z,
                                        velocity_time, component),
                      (Real)DZ_INVERSE_SQUARE)
                : interior_second_derivative(
                      velocity, index, plane_size,
                      (Real)DZ_INVERSE_SQUARE);
            break;

        case 2:
            if (i < 1 || i >= WIDTH ||
                j < 1 || j >= HEIGHT ||
                k < 1 || k >= DEPTH - 1) {
                return 0.0;
            }

            eta = solver_mem_state->eta.v_z;
            zeta = solver_mem_state->zeta.v_z;
            velocity = solver_mem_state->u.v_z;
            z += (Real)DZ / 2.0;

            pressure_gradient =
                (pressure[index + plane_size] - pressure[index]) *
                (Real)DZ_INVERSE;
            laplacian_x = (i == WIDTH - 1)
                ? upper_second_derivative(
                      eta, index, 1,
                      data->bc_velocity(upper_x, y, z,
                                        velocity_time, component),
                      (Real)DX_INVERSE_SQUARE)
                : interior_second_derivative(
                      eta, index, 1, (Real)DX_INVERSE_SQUARE);
            laplacian_y = (j == HEIGHT - 1)
                ? upper_second_derivative(
                      zeta, index, WIDTH,
                      data->bc_velocity(x, upper_y, z,
                                        velocity_time, component),
                      (Real)DY_INVERSE_SQUARE)
                : interior_second_derivative(
                      zeta, index, WIDTH, (Real)DY_INVERSE_SQUARE);
            laplacian_z = interior_second_derivative(
                velocity, index, plane_size,
                (Real)DZ_INVERSE_SQUARE);
            break;

        default:
            return 0.0;
    }

    return data->forcing_fn(x, y, z, forcing_time, component) -
           pressure_gradient -
           ((Real)NU / k_i) * velocity[index] +
           (Real)NU * (laplacian_x + laplacian_y + laplacian_z);
}
