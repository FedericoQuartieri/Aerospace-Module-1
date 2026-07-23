#ifndef FIELD_H
#define FIELD_H

#include "types.h"

void scalarField_alloc(ScalarField *sf);
void vectorField_alloc(VectorField *vf);

void scalarField_fill(ScalarField *sf, ScalarFunction scalar_fn, int t_step);
void vectorField_fill(VectorField *vf, VectorFunction vector_fn, int t_step);

#endif
