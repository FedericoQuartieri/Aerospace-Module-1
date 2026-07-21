#ifndef FIELD_H
#define FIELD_H

#include "types.h"

void scalarField_alloc(ScalarField *sf);
void vectorField_alloc(VectorField *vf);

void scalarField_fill(ScalarField *sf, ScalarFunction);
void vectorField_fill(VectorField *vf, VectorFunction);

#endif
