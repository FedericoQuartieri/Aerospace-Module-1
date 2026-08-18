#ifndef PARALLEL_H
#define PARALLEL_H

#include <mpi.h>
#include <stddef.h>

#include "types.h"

enum { AXIS_X = 0, AXIS_Y = 1, AXIS_Z = 2, AXIS_COUNT = 3 };

typedef struct Domain {
    MPI_Comm cart;
    int rank;
    int size;
    int dims[AXIS_COUNT];
    int coords[AXIS_COUNT];
    int lower[AXIS_COUNT];
    int upper[AXIS_COUNT];
    int global[AXIS_COUNT];
    int local[AXIS_COUNT];
    int start[AXIS_COUNT];
    size_t stride_y;
    size_t stride_z;
    size_t owned_cells;
    size_t allocated_cells;
    MPI_Datatype send_lower[AXIS_COUNT];
    MPI_Datatype send_upper[AXIS_COUNT];
    MPI_Datatype recv_lower[AXIS_COUNT];
    MPI_Datatype recv_upper[AXIS_COUNT];
} Domain;

void domain_init(Domain *domain, const int global[AXIS_COUNT]);
void domain_destroy(Domain *domain);
void domain_exchange_halo(const Domain *domain, Real *field, int axis);

static inline MPI_Datatype mpi_real_type(void) {
#ifdef USE_FLOAT
    return MPI_FLOAT;
#else
    return MPI_DOUBLE;
#endif
}

/* Local owned coordinates are [0, local[d]); -1 and local[d] address halos. */
static inline size_t domain_index(const Domain *domain,
                                  int i, int j, int k) {
    return (size_t)(k + 1) * domain->stride_z +
           (size_t)(j + 1) * domain->stride_y +
           (size_t)(i + 1);
}

static inline int domain_global_index(const Domain *domain,
                                      int local_index, int axis) {
    return domain->start[axis] + local_index;
}

#endif
