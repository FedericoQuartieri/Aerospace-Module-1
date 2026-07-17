#ifndef MANUFACTURED_CASES_H
#define MANUFACTURED_CASES_H

#include "solver.h"

typedef struct {
    /* Exact callbacks and the default run used by correctness checks. */
    ProblemDefinition problem;
    SolverConfig base_config;
    VectorFunction exact_velocity;
    ScalarFunction exact_pressure;
    /* Correctness maxima followed by minimum accepted convergence orders. */
    Real max_velocity_l2;
    Real max_velocity_linf;
    Real max_pressure_l2;
    Real max_pressure_linf;
    Real max_divergence_l2;
    Real max_boundary_linf;
    Real min_velocity_space_order;
    Real min_pressure_space_order;
    Real min_velocity_time_order;
    Real min_pressure_time_order;
} ManufacturedCase;

extern const ManufacturedCase MANUFACTURED_CASES[];
extern const size_t MANUFACTURED_CASE_COUNT;

#endif
