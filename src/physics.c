#include "physics.h"

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

Real time_physical_coord(int t_step) {
    return (Real)t_step * DT;
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
