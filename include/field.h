#ifndef FIELD_H
#define FIELD_H

#include "types.h"

void scalarField_alloc(ScalarField *sf);
void vectorField_alloc(VectorField *vf);

void scalarField_fill(ScalarField *restrict sf,
                      ScalarFunction scalar_fn,
                      int t_step);
void vectorField_fill(VectorField *restrict vf,
                      VectorFunction vector_fn,
                      int t_step);

#endif
