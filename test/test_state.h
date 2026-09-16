#ifndef TEST_STATE_H
#define TEST_STATE_H
#include "solver.h"
#include "parallel.h"

static inline void test_state_fields(SolverMemState *s, Real *fields[11]) {
    Real *values[11] = {s->eta.v_x,s->eta.v_y,s->eta.v_z,
        s->zeta.v_x,s->zeta.v_y,s->zeta.v_z,s->u.v_x,s->u.v_y,s->u.v_z,
        s->pressure.v,s->pressure_star.v};
    for (int i = 0; i < 11; i++) fields[i] = values[i];
}
static inline void test_state_free(SolverMemState *s) {
    Real *fields[11]; test_state_fields(s, fields);
    for (int i = 0; i < 11; i++) free(fields[i]);
    free(s->k.v_x); free(s->k.v_y); free(s->k.v_z);
}
static inline Real test_variable_porosity(Real x, Real y, Real z, Real t, int c) {
    return (Real)(1.0 + 0.2 * sin(x + 2*y + z + t + c));
}
#endif
