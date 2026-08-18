#ifndef FIELD_H
#define FIELD_H

#include "types.h"
#include "parallel.h"

void scalarField_alloc(ScalarField *sf, const Domain *domain);
void vectorField_alloc(VectorField *vf, const Domain *domain);

void scalarField_fill(ScalarField *restrict sf,
                      const Domain *domain,
                      ScalarFunction scalar_fn,
                      Real t_step);
void vectorField_fill(VectorField *restrict vf,
                      const Domain *domain,
                      VectorFunction vector_fn,
                      Real t_step);

void vectorField_free(VectorField *vf);
#endif
