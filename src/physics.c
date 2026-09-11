#include "physics.h"
#include "simd_real.h"
#include "solver.h"

#include <stdbool.h>

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
 * L'ultima riga di g, per conto suo perche' tre percorsi devono arrotondarla
 * allo stesso modo: g_value, il ciclo scalare di g_line e il suo blocco
 * vettoriale.  Spezzarla diversamente -- sommare prima lo stencil e aggiungere
 * la forzante dopo, per dire -- cambierebbe l'ultimo bit del risultato.
 */
static inline Real g_combine(Real forcing, Real gradient, Real drag,
                             Real laplacian_sum) {
    return forcing - gradient - drag + (Real)NU * laplacian_sum;
}

#if defined(USE_SIMD) && SIMD_AVAILABLE

/*
 * Le due qui sopra, un vettore di celle x adiacenti alla volta.  Adiacenti
 * vuol dire contigue (stride[0] == 1), quindi i vicini lungo ogni asse sono
 * letture normali non allineate: niente gather, niente trasposizioni.  Le
 * operazioni sono nello stesso ordine delle versioni scalari, ed e' quello a
 * tenere i due percorsi uguali fino all'ultima cifra.
 */
static inline SimdReal interior_second_derivative_simd(
    const Real *restrict field,
    size_t index,
    size_t stride,
    SimdReal inverse_spacing_square) {
    SimdReal left = simd_loadu(&field[index - stride]);
    SimdReal here = simd_loadu(&field[index]);
    SimdReal right = simd_loadu(&field[index + stride]);

    return simd_mul(simd_add(simd_sub(left, simd_mul(simd_set1(2.0), here)),
                             right),
                    inverse_spacing_square);
}

static inline SimdReal g_combine_simd(SimdReal forcing, SimdReal gradient,
                                      SimdReal drag, SimdReal laplacian_sum,
                                      SimdReal nu) {
    return simd_add(simd_sub(simd_sub(forcing, gradient), drag),
                    simd_mul(nu, laplacian_sum));
}

#endif

/*
 * Le coordinate della cella (i, j, k), con la mezza cella che g_value aggiunge
 * alla componente richiesta.  Sono scritte qui una volta sola perche' i tre
 * modi di arrivare alla forzante -- la linea, la singola cella e g_value --
 * devono valutarla nello stesso punto fino all'ultimo bit.
 */
static void forcing_coords(const Decomp *d, int i, int j, int k,
                           int component, Real *x, Real *y, Real *z) {
    *x = (Real)decomp_global(d, i, 0) * (Real)DX;
    *y = (Real)decomp_global(d, j, 1) * (Real)DY;
    *z = (Real)decomp_global(d, k, 2) * (Real)DZ;

    if (component == 0) {
        *x += (Real)DX / 2.0;
    } else if (component == 1) {
        *y += (Real)DY / 2.0;
    } else if (component == 2) {
        *z += (Real)DZ / 2.0;
    }
}

static Real forcing_time_of(int t_step) {
    return ((Real)t_step - 0.5) * (Real)DT;
}

void forcing_line_coords(const Decomp *d, int component, Real *restrict xs) {
    for (int i = 0; i < d->n[0]; i++) {
        Real x = (Real)decomp_global(d, i, 0) * (Real)DX;

        /* La mezza cella che g_value aggiunge alla componente x. */
        if (component == 0) {
            x += (Real)DX / 2.0;
        }
        xs[i] = x;
    }
}

void forcing_fill_line(const Decomp *d, const Data *data,
                       int j, int k, int t_step, int component,
                       const Real *restrict xs, Real *restrict out) {
    int n = d->n[0];
    Real y = (Real)decomp_global(d, j, 1) * (Real)DY;
    Real z = (Real)decomp_global(d, k, 2) * (Real)DZ;
    Real t = forcing_time_of(t_step);

    /* Le stesse mezze celle di g_value, nello stesso ordine di operazioni. */
    if (component == 1) {
        y += (Real)DY / 2.0;
    } else if (component == 2) {
        z += (Real)DZ / 2.0;
    }

    if (data->forcing_line_fn != NULL) {
        data->forcing_line_fn(out, xs, n, y, z, t, component);
        return;
    }

    /* Nessuna versione a linea: si paga il prezzo di prima, una chiamata per
     * cella, ma il ciclo interno del chiamante resta comunque pulito. */
    for (int i = 0; i < n; i++) {
        out[i] = data->forcing_fn(xs[i], y, z, t, component);
    }
}

Real forcing_at_cell(const Decomp *d, const Data *data,
                     int i, int j, int k, int t_step, int component) {
    Real x;
    Real y;
    Real z;

    forcing_coords(d, i, j, k, component, &x, &y, &z);

    return data->forcing_fn(x, y, z, forcing_time_of(t_step), component);
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
/*
 * Il nucleo di g. `forzante_pronta' dice se usare il valore che arriva da
 * fuori -- chi percorre le linee lo ha gia' calcolato per tutta la linea --
 * oppure valutarlo qui, dove x, y e z sono gia' sfalsate per la componente.
 */
static Real g_core(const Decomp *d,
                   int i, int j, int k, int t_step, Real k_i,
                   const SolverMemState *solver_mem_state,
                   const Data *data, int component,
                   int forzante_pronta, Real forcing) {
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

    //la forzante arriva gia' valutata in (t_step - 1/2) DT; per la velocità
    //serve invece t_step - 1
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

    if (!forzante_pronta) {
        forcing = data->forcing_fn(x, y, z, forcing_time_of(t_step), component);
    }

    return g_combine(forcing,
                     pressure_gradient,
                     ((Real)NU / k_i) * velocity[index],
                     laplacian_x + laplacian_y + laplacian_z);
}

Real g_value(const Decomp *d,
             int i, int j, int k, int t_step, Real k_i,
             const SolverMemState *solver_mem_state,
             const Data *data, int component, Real forcing) {
    return g_core(d, i, j, k, t_step, k_i, solver_mem_state, data, component,
                  1, forcing);
}

Real g_value_here(const Decomp *d,
                  int i, int j, int k, int t_step, Real k_i,
                  const SolverMemState *solver_mem_state,
                  const Data *data, int component) {
    return g_core(d, i, j, k, t_step, k_i, solver_mem_state, data, component,
                  0, (Real)0);
}

/*
 * g di tutta una linea lungo x.
 *
 * Ogni test che g_value fa per cella qui si fa una volta per linea, perche' il
 * supporto di g e la scelta del nodo fantasma dipendono da j e k, che sulla
 * linea non cambiano.  Quello che sopravvive e' un'espressione dritta su celle
 * vicine in memoria, quindi il blocco vettoriale e' la stessa aritmetica del
 * ciclo scalare che lo segue, un vettore di celle alla volta.
 *
 * I pezzi che invece cambiano da cella a cella -- la forzante e i due estremi
 * della linea -- non sono riscritti: la forzante la fa lo scenario, gli
 * estremi tornano a g_value.
 */
void g_line(const Decomp *d, const Data *data,
            const SolverMemState *solver_mem_state,
            const Real *restrict k_porosity,
            int j, int k, int t_step, int component,
            const Real *restrict xs, Real *restrict out) {
    int count = d->n[0];
    int gj = decomp_global(d, j, 1);
    int gk = decomp_global(d, k, 2);
    size_t row = decomp_index(d, 0, j, k);
    const Real *restrict pressure = solver_mem_state->pressure_star.v;
    const Real *restrict eta;
    const Real *restrict zeta;
    const Real *restrict velocity;
    size_t gradient_stride;
    Real gradient_inverse;
    /* Sono proprieta' della linea, non della cella. */
    bool in_support;
    bool leans_on_ghost;
    int first;
    int fast_end;
    int n;

    switch (component) {
        case 0:
            in_support = (gj >= 1 && gj < d->n_global[1] &&
                          gk >= 1 && gk < d->n_global[2]);
            leans_on_ghost = (gj == d->n_global[1] - 1 ||
                              gk == d->n_global[2] - 1);
            eta = solver_mem_state->eta.v_x;
            zeta = solver_mem_state->zeta.v_x;
            velocity = solver_mem_state->u.v_x;
            gradient_stride = d->stride[0];
            gradient_inverse = (Real)DX_INVERSE;
            break;

        case 1:
            in_support = (gj >= 1 && gj < d->n_global[1] - 1 &&
                          gk >= 1 && gk < d->n_global[2]);
            leans_on_ghost = (gk == d->n_global[2] - 1);
            eta = solver_mem_state->eta.v_y;
            zeta = solver_mem_state->zeta.v_y;
            velocity = solver_mem_state->u.v_y;
            gradient_stride = d->stride[1];
            gradient_inverse = (Real)DY_INVERSE;
            break;

        case 2:
            in_support = (gj >= 1 && gj < d->n_global[1] &&
                          gk >= 1 && gk < d->n_global[2] - 1);
            leans_on_ghost = (gj == d->n_global[1] - 1);
            eta = solver_mem_state->eta.v_z;
            zeta = solver_mem_state->zeta.v_z;
            velocity = solver_mem_state->u.v_z;
            gradient_stride = d->stride[2];
            gradient_inverse = (Real)DZ_INVERSE;
            break;

        default:
            for (n = 0; n < count; n++) {
                out[n] = 0.0;
            }
            return;
    }

    /* Fuori dal supporto in j o k la linea e' tutta zero, una cella come
     * l'altra: e' quello che g_value risponde li'. */
    if (!in_support) {
        for (n = 0; n < count; n++) {
            out[n] = 0.0;
        }
        return;
    }

    /* La forzante di tutta la linea prima di tutto: la ritrovano sia il ciclo
     * vettoriale sia le celle che tornano a g_value. */
    forcing_fill_line(d, data, j, k, t_step, component, xs, out);

    /*
     * Una linea che appoggia su un nodo fantasma legge il valore alla parete
     * nell'ascissa della cella, quindi cambia da una cella all'altra e non
     * resta niente da issare fuori.  Quelle linee sono due piani del blocco:
     * tengono il percorso di prima, cella per cella.
     */
    if (leans_on_ghost) {
        for (n = 0; n < count; n++) {
            out[n] = g_value(d, n, j, k, t_step, k_porosity[row + (size_t)n],
                             solver_mem_state, data, component, out[n]);
        }
        return;
    }

    /*
     * Le celle scritte dal percorso veloce sono quelle di indice globale da 1
     * a n_global[0] - 2.  Ne resta al piu' una per estremo: gi == 0, sempre
     * fuori supporto, e gi == n_global[0] - 1, fuori supporto per la
     * componente x e nodo fantasma per le altre due.  Tornano tutt'e due a
     * g_value, cosi' quell'algebra resta scritta una volta sola.
     */
    first = 1 - d->start[0];
    fast_end = d->n_global[0] - 1 - d->start[0];
    if (first < 0) {
        first = 0;
    }
    if (fast_end > count) {
        fast_end = count;
    }
    if (fast_end < first) {
        fast_end = first;
    }

    for (n = 0; n < first; n++) {
        out[n] = g_value(d, n, j, k, t_step, k_porosity[row + (size_t)n],
                         solver_mem_state, data, component, out[n]);
    }
    for (n = fast_end; n < count; n++) {
        out[n] = g_value(d, n, j, k, t_step, k_porosity[row + (size_t)n],
                         solver_mem_state, data, component, out[n]);
    }

    n = first;

#if defined(USE_SIMD) && SIMD_AVAILABLE
    {
        const SimdReal nu = simd_set1((Real)NU);
        const SimdReal gradient_inverse_v = simd_set1(gradient_inverse);
        const SimdReal inverse_x = simd_set1((Real)DX_INVERSE_SQUARE);
        const SimdReal inverse_y = simd_set1((Real)DY_INVERSE_SQUARE);
        const SimdReal inverse_z = simd_set1((Real)DZ_INVERSE_SQUARE);

        for (; n + SIMD_LANES <= fast_end; n += SIMD_LANES) {
            size_t at = row + (size_t)n;
            SimdReal pressure_gradient = simd_mul(
                simd_sub(simd_loadu(&pressure[at + gradient_stride]),
                         simd_loadu(&pressure[at])),
                gradient_inverse_v);
            SimdReal laplacian_x = interior_second_derivative_simd(
                eta, at, d->stride[0], inverse_x);
            SimdReal laplacian_y = interior_second_derivative_simd(
                zeta, at, d->stride[1], inverse_y);
            SimdReal laplacian_z = interior_second_derivative_simd(
                velocity, at, d->stride[2], inverse_z);
            SimdReal drag = simd_mul(simd_div(nu, simd_loadu(&k_porosity[at])),
                                     simd_loadu(&velocity[at]));

            simd_storeu(&out[n],
                        g_combine_simd(simd_loadu(&out[n]),
                                       pressure_gradient, drag,
                                       simd_add(simd_add(laplacian_x,
                                                         laplacian_y),
                                                laplacian_z),
                                       nu));
        }
    }
#endif

    /* Le celle che i vettori non hanno coperto, e tutta la linea quando la
     * build non ha SIMD: stessa espressione, una cella alla volta. */
    for (; n < fast_end; n++) {
        size_t at = row + (size_t)n;
        Real pressure_gradient =
            (pressure[at + gradient_stride] - pressure[at]) * gradient_inverse;
        Real laplacian_x = interior_second_derivative(
            eta, at, d->stride[0], (Real)DX_INVERSE_SQUARE);
        Real laplacian_y = interior_second_derivative(
            zeta, at, d->stride[1], (Real)DY_INVERSE_SQUARE);
        Real laplacian_z = interior_second_derivative(
            velocity, at, d->stride[2], (Real)DZ_INVERSE_SQUARE);
        Real drag = ((Real)NU / k_porosity[at]) * velocity[at];

        out[n] = g_combine(out[n], pressure_gradient, drag,
                           laplacian_x + laplacian_y + laplacian_z);
    }
}
