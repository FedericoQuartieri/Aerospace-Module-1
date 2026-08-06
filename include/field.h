#ifndef FIELD_H
#define FIELD_H

#include "decomp.h"
#include "types.h"

void scalarField_alloc(const Decomp *d, ScalarField *sf);
void vectorField_alloc(const Decomp *d, VectorField *vf);

void scalarField_fill(const Decomp *restrict d,
                      ScalarField *restrict sf,
                      ScalarFunction scalar_fn,
                      Real t_step);
void vectorField_fill(const Decomp *restrict d,
                      VectorField *restrict vf,
                      VectorFunction vector_fn,
                      Real t_step);

#endif
