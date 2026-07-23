#include "physics.h"

Real inline beta_from_k(Real k) {
    return 1.0 + (DT * NU) / (2.0 * k);
}

Real inline gamma_from_beta(Real k) {
    Real beta = beta_from_k(k);
    return (DT * NU) / (2.0 * beta);
}

