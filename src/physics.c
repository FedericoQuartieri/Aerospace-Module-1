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

// calcola beta in base alla permeaabilità k, secondo la formula beta = 1 + (DT * NU) / (2 * k)
Real beta_from_k(Real k) {
    return 1.0 + (DT * NU) / (2.0 * k);
}

// calcola gamma in base alla permeaabilità k, secondo la formula gamma = (DT * NU) / (2 * beta)
Real gamma_from_k(Real k) {
    Real beta = beta_from_k(k);
    return (DT * NU) / (2.0 * beta);
}

//ritorna il tempo fisico corrispondente al passo temporale t_step, secondo la formula t = t_step * DT
Real time_physical_coord(Real t_step) {
    return t_step * (Real)DT;
}

//ritorna la coordinata fisica centrata di un punto con indice index e componente component, secondo la formula x = index * spacing
Real centered_physical_coord(int index, int component) {
    return (Real)index * spacing_from_component(component);
}

//ritorna la coordinata fisica sfalsata di 0.5 di un punto con indice index e componente component, secondo la formula x = (index + 0.5) * spacing
Real staggered_physical_coord(int index, int component) {
    return ((Real)index + 0.5) * spacing_from_component(component);
}

//se il passo temporale è 0, ritorna il valore della funzione di velocità al tempo t,
// altrimenti ritorna la differenza tra il valore della funzione di velocità
// al tempo t e il valore della funzione di velocità al tempo t - DT
static inline Real boundary_increment(VectorFunction bc_velocity,
                                      Real x, Real y, Real z,
                                      Real t, int t_step, int component) {
    Real current = bc_velocity(x, y, z, t, component);
    if (t_step == 0) {
        return current;
    }
    return current - bc_velocity(x, y, z, t - (Real)DT, component);
}

/*
 * Restituisce l'incremento al bordo u(t_step) - u(t_step - 1).
 * Su una faccia inferiore, la componente normale viene ricostruita dal
 * vincolo di divergenza nulla; le componenti tangenziali vengono campionate
 * direttamente. Sui bordi inferiori e negli angoli ha la precedenza il
 * valore sfalsato prescritto.
 *
 * i, j, k sono indici globali, quindi i test sulle facce sottostanti
 * selezionano il bordo fisico del dominio anziché il bordo di un blocco di
 * processo.
 */
Real bc_left(VectorFunction bc_velocity,
             int i, int j, int k, int t_step, int component) {
    Real t = time_physical_coord(t_step);
    Real x = (Real)i * (Real)DX;
    Real y = (Real)j * (Real)DY;
    Real z = (Real)k * (Real)DZ;
    Real vx = x + (Real)DX / 2.0;
    Real vy = y + (Real)DY / 2.0;
    Real vz = z + (Real)DZ / 2.0;

    //vale 0 se il punto è interno, 1 se è su una faccia, 
    //2 se è su uno spigolo, 3 se è su un vertice
    int lower_face_count = (i == 0) + (j == 0) + (k == 0);

    if ((unsigned int)component > 2U) {
        fprintf(stderr, "Invalid vector component: %d\n", component);
        exit(1);
    }
    // 0 => non fa nulla
    if (lower_face_count == 0) {
        return 0.0;
    }

#define BC_INCREMENT(px, py, pz, comp) \
    boundary_increment(bc_velocity, (px), (py), (pz), \
                       t, t_step, (comp))

    // lower_face_count > 1 significa che il punto è su uno spigolo o un vertice, quindi
    // si fa un incremento in base alla componente normale della velocità, che ha la precedenza sulle altre
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
            Real divergence_y =
                (BC_INCREMENT(0.0, vy, z, 1) -
                 BC_INCREMENT(0.0, vy - (Real)DY, z, 1)) *
                (Real)DY_INVERSE;
            Real divergence_z =
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
            Real divergence_x =
                (BC_INCREMENT(vx, 0.0, z, 0) -
                 BC_INCREMENT(vx - (Real)DX, 0.0, z, 0)) *
                (Real)DX_INVERSE;
            Real divergence_z =
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
        Real divergence_x =
            (BC_INCREMENT(vx, y, 0.0, 0) -
             BC_INCREMENT(vx - (Real)DX, y, 0.0, 0)) *
            (Real)DX_INVERSE;
        Real divergence_y =
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

    Real t = time_physical_coord(t_step);
    Real x = (Real)i * (Real)DX;
    Real y = (Real)j * (Real)DY;
    Real z = (Real)k * (Real)DZ;
    Real vx = x + (Real)DX / 2.0;
    Real vy = y + (Real)DY / 2.0;
    Real vz = z + (Real)DZ / 2.0;

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

//ritorna la derivata seconda a 3 punti standard, secondo la formula (f(x - h) - 2 * f(x) + f(x + h)) / (h^2)
static inline Real interior_second_derivative(const Real *restrict field,
                                              size_t index,
                                              size_t stride,
                                              Real inverse_spacing_square) {
    return (field[index - stride] -
            2.0 * field[index] +
            field[index + stride]) * inverse_spacing_square;
}

//ritorna la derivata seconda a 3 punti con condizione al contorno, secondo la formula (f(x - h) - 3 * f(x) + 2 * boundary_value) / (h^2)
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
Real g_value(const Decomp *d,
             int i, int j, int k, int t_step, Real k_i,
             const SolverMemState *solver_mem_state,
             const Data *data, int component) {
    size_t stride_x = d->stride[0];
    size_t stride_y = d->stride[1];
    size_t stride_z = d->stride[2];
    size_t index = decomp_index(d, i, j, k);
    /*
     * The support of g and the ghost-value reconstruction below are properties
     * of the physical boundary, so both are decided on the global indices.
     */
    //gi, gj, gk sono gli indici globali del punto (i, j, k)
    int gi = decomp_global(d, i, 0);
    int gj = decomp_global(d, j, 1);
    int gk = decomp_global(d, k, 2);

    //per la forzante è t_step - 1/2, per la velocità è t_step - 1
    Real forcing_time = ((Real)t_step - 0.5) * (Real)DT;
    Real velocity_time = ((Real)t_step - 1.0) * (Real)DT;
    
    //upper_x, upper_y, upper_z sono le coordinate fisiche del bordo superiore del dominio
    Real upper_x = ((Real)d->n_global[0] - 0.5) * (Real)DX;
    Real upper_y = ((Real)d->n_global[1] - 0.5) * (Real)DY;
    Real upper_z = ((Real)d->n_global[2] - 0.5) * (Real)DZ;

    //x, y, z sono le coordinate fisiche del punto (gi, gj, gk)
    Real x = (Real)gi * (Real)DX;
    Real y = (Real)gj * (Real)DY;
    Real z = (Real)gk * (Real)DZ;
    const Real *restrict eta;
    const Real *restrict zeta;
    const Real *restrict velocity;
    const Real *restrict pressure = solver_mem_state->pressure_star.v;
    Real pressure_gradient;
    Real laplacian_x;
    Real laplacian_y;
    Real laplacian_z;
    
    //case 0: v_x, case 1: v_y, case 2: v_z
    switch (component) {
        case 0:

        //se il punto è su una faccia inferiore o su una faccia superiore,
        // ritorna 0 perché il valore della velocità lì è imposto dalle bc,
        // quindi non c'è bisogno di calcolare g, la stessa cosa vale,
        // per il case 1 e 2
            if (gi < 1 || gi >= d->n_global[0] - 1 ||
                gj < 1 || gj >= d->n_global[1] ||
                gk < 1 || gk >= d->n_global[2]) {
                return 0.0;
            }

            eta = solver_mem_state->eta.v_x;
            zeta = solver_mem_state->zeta.v_x;
            velocity = solver_mem_state->u.v_x;
            x += (Real)DX / 2.0;

            pressure_gradient =
                (pressure[index + stride_x] - pressure[index]) *
                (Real)DX_INVERSE;
            laplacian_x = interior_second_derivative(
                eta, index, stride_x, (Real)DX_INVERSE_SQUARE);
            laplacian_y = (gj == d->n_global[1] - 1)
                ? upper_second_derivative(
                      zeta, index, stride_y,
                      data->bc_velocity(x, upper_y, z,
                                        velocity_time, component),
                      (Real)DY_INVERSE_SQUARE)
                : interior_second_derivative(
                      zeta, index, stride_y, (Real)DY_INVERSE_SQUARE);
            laplacian_z = (gk == d->n_global[2] - 1)
                ? upper_second_derivative(
                      velocity, index, stride_z,
                      data->bc_velocity(x, y, upper_z,
                                        velocity_time, component),
                      (Real)DZ_INVERSE_SQUARE)
                : interior_second_derivative(
                      velocity, index, stride_z,
                      (Real)DZ_INVERSE_SQUARE);
            break;

        case 1:
            if (gi < 1 || gi >= d->n_global[0] ||
                gj < 1 || gj >= d->n_global[1] - 1 ||
                gk < 1 || gk >= d->n_global[2]) {
                return 0.0;
            }

            eta = solver_mem_state->eta.v_y;
            zeta = solver_mem_state->zeta.v_y;
            velocity = solver_mem_state->u.v_y;
            y += (Real)DY / 2.0;

            pressure_gradient =
                (pressure[index + stride_y] - pressure[index]) *
                (Real)DY_INVERSE;
            laplacian_x = (gi == d->n_global[0] - 1)
                ? upper_second_derivative(
                      eta, index, stride_x,
                      data->bc_velocity(upper_x, y, z,
                                        velocity_time, component),
                      (Real)DX_INVERSE_SQUARE)
                : interior_second_derivative(
                      eta, index, stride_x, (Real)DX_INVERSE_SQUARE);
            laplacian_y = interior_second_derivative(
                zeta, index, stride_y, (Real)DY_INVERSE_SQUARE);
            laplacian_z = (gk == d->n_global[2] - 1)
                ? upper_second_derivative(
                      velocity, index, stride_z,
                      data->bc_velocity(x, y, upper_z,
                                        velocity_time, component),
                      (Real)DZ_INVERSE_SQUARE)
                : interior_second_derivative(
                      velocity, index, stride_z,
                      (Real)DZ_INVERSE_SQUARE);
            break;

        case 2:
            if (gi < 1 || gi >= d->n_global[0] ||
                gj < 1 || gj >= d->n_global[1] ||
                gk < 1 || gk >= d->n_global[2] - 1) {
                return 0.0;
            }

            eta = solver_mem_state->eta.v_z;
            zeta = solver_mem_state->zeta.v_z;
            velocity = solver_mem_state->u.v_z;
            z += (Real)DZ / 2.0;

            pressure_gradient =
                (pressure[index + stride_z] - pressure[index]) *
                (Real)DZ_INVERSE;
            laplacian_x = (gi == d->n_global[0] - 1)
                ? upper_second_derivative(
                      eta, index, stride_x,
                      data->bc_velocity(upper_x, y, z,
                                        velocity_time, component),
                      (Real)DX_INVERSE_SQUARE)
                : interior_second_derivative(
                      eta, index, stride_x, (Real)DX_INVERSE_SQUARE);
            laplacian_y = (gj == d->n_global[1] - 1)
                ? upper_second_derivative(
                      zeta, index, stride_y,
                      data->bc_velocity(x, upper_y, z,
                                        velocity_time, component),
                      (Real)DY_INVERSE_SQUARE)
                : interior_second_derivative(
                      zeta, index, stride_y, (Real)DY_INVERSE_SQUARE);
            laplacian_z = interior_second_derivative(
                velocity, index, stride_z,
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
