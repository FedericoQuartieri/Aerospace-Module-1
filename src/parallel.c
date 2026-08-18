#include "parallel.h"

#include <stdio.h>
#include <stdlib.h>

static void abort_mpi(const Domain *domain, const char *message) {
    if (domain == NULL || domain->rank == 0) {
        fprintf(stderr, "%s\n", message);
    }
    MPI_Abort(domain != NULL ? domain->cart : MPI_COMM_WORLD, EXIT_FAILURE);
}

static void split_extent(int global, int parts, int coordinate,
                         int *start, int *length) {
    int quotient = global / parts;
    int remainder = global % parts;

    *length = quotient + (coordinate < remainder);
    *start = coordinate * quotient +
             (coordinate < remainder ? coordinate : remainder);
}

static void create_face_type(const int sizes[3], const int subsizes[3],
                             const int starts[3], MPI_Datatype *type) {
    MPI_Type_create_subarray(3, sizes, subsizes, starts, MPI_ORDER_C,
                             mpi_real_type(), type);
    MPI_Type_commit(type);
}

static void create_halo_types(Domain *domain) {
    const int nx = domain->local[AXIS_X];
    const int ny = domain->local[AXIS_Y];
    const int nz = domain->local[AXIS_Z];
    const int sizes[3] = {nz + 2, ny + 2, nx + 2};

    for (int axis = 0; axis < AXIS_COUNT; axis++) {
        int subsizes[3] = {nz, ny, nx};
        int send_low[3] = {1, 1, 1};
        int send_high[3] = {1, 1, 1};
        int recv_low[3] = {1, 1, 1};
        int recv_high[3] = {1, 1, 1};
        int array_axis = 2 - axis;

        subsizes[array_axis] = 1;
        send_high[array_axis] = domain->local[axis];
        recv_low[array_axis] = 0;
        recv_high[array_axis] = domain->local[axis] + 1;

        create_face_type(sizes, subsizes, send_low,
                         &domain->send_lower[axis]);
        create_face_type(sizes, subsizes, send_high,
                         &domain->send_upper[axis]);
        create_face_type(sizes, subsizes, recv_low,
                         &domain->recv_lower[axis]);
        create_face_type(sizes, subsizes, recv_high,
                         &domain->recv_upper[axis]);
    }
}

void domain_init(Domain *domain, const int global[AXIS_COUNT]) {
    int initialized = 0;
    int periods[AXIS_COUNT] = {0, 0, 0};

    MPI_Initialized(&initialized);
    if (!initialized) {
        fprintf(stderr, "MPI_Init must be called before solver_init\n");
        exit(EXIT_FAILURE);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &domain->size);
    domain->dims[0] = 0;
    domain->dims[1] = 0;
    domain->dims[2] = 0;
    MPI_Dims_create(domain->size, AXIS_COUNT, domain->dims);
    MPI_Cart_create(MPI_COMM_WORLD, AXIS_COUNT, domain->dims,
                    periods, 0, &domain->cart);
    MPI_Comm_rank(domain->cart, &domain->rank);
    MPI_Cart_coords(domain->cart, domain->rank, AXIS_COUNT, domain->coords);

    for (int axis = 0; axis < AXIS_COUNT; axis++) {
        domain->global[axis] = global[axis];
        if (domain->dims[axis] > global[axis]) {
            abort_mpi(domain, "Process grid is finer than the numerical grid");
        }
        split_extent(global[axis], domain->dims[axis], domain->coords[axis],
                     &domain->start[axis], &domain->local[axis]);
        MPI_Cart_shift(domain->cart, axis, 1,
                       &domain->lower[axis], &domain->upper[axis]);
    }

    domain->stride_y = (size_t)domain->local[AXIS_X] + 2;
    domain->stride_z = domain->stride_y *
                       ((size_t)domain->local[AXIS_Y] + 2);
    domain->owned_cells = (size_t)domain->local[AXIS_X] *
                          (size_t)domain->local[AXIS_Y] *
                          (size_t)domain->local[AXIS_Z];
    domain->allocated_cells = domain->stride_z *
                              ((size_t)domain->local[AXIS_Z] + 2);
    create_halo_types(domain);
}

void domain_destroy(Domain *domain) {
    for (int axis = 0; axis < AXIS_COUNT; axis++) {
        MPI_Type_free(&domain->send_lower[axis]);
        MPI_Type_free(&domain->send_upper[axis]);
        MPI_Type_free(&domain->recv_lower[axis]);
        MPI_Type_free(&domain->recv_upper[axis]);
    }
    MPI_Comm_free(&domain->cart);
}

void domain_exchange_halo(const Domain *domain, Real *field, int axis) {
    MPI_Request requests[4];
    const int tag_to_lower = 400 + 2 * axis;
    const int tag_to_upper = tag_to_lower + 1;

    MPI_Irecv(field, 1, domain->recv_lower[axis], domain->lower[axis],
              tag_to_upper, domain->cart, &requests[0]);
    MPI_Irecv(field, 1, domain->recv_upper[axis], domain->upper[axis],
              tag_to_lower, domain->cart, &requests[1]);
    MPI_Isend(field, 1, domain->send_lower[axis], domain->lower[axis],
              tag_to_lower, domain->cart, &requests[2]);
    MPI_Isend(field, 1, domain->send_upper[axis], domain->upper[axis],
              tag_to_upper, domain->cart, &requests[3]);
    MPI_Waitall(4, requests, MPI_STATUSES_IGNORE);
}
