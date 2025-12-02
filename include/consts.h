#ifndef CONSTS_H
#define CONSTS_H

#include "ftype.h"

extern const ftype _NU;
extern const ftype _DT;
extern const ftype _DX;

#define DEFINE_NU(x) const ftype _NU = x;
#define DEFINE_DT(x) const ftype _DT = x;
#define DEFINE_DX(x) const ftype _DX = x;

/* TODO: vector getters? */

#endif
