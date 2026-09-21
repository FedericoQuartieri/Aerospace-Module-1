/*
 * The two implementations of the functions declared in parallel.h: the real
 * one, which calls MPI, and the fake one, which describes a single process.
 * The compiler chooses based on -DUSE_MPI, that is `make MPI=1`.
 *
 * It is the only file of the project that includes <mpi.h>.
 */
#include "parallel.h"
#include "utils.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI

#include <mpi.h>

#ifdef USE_FLOAT
#define PAR_REAL MPI_FLOAT
#else
#define PAR_REAL MPI_DOUBLE
#endif

/* Time spent inside MPI, for the end-of-simulation report. */
static unsigned long long comm_ns = 0;

unsigned long long par_comm_nanoseconds(void) {
    return comm_ns;
}

/* The process grid, created once by par_topology_init. */
static MPI_Comm cart_comm = MPI_COMM_NULL;
static int cart_dims[3] = {1, 1, 1};
static int cart_coords[3] = {0, 0, 0};

/* One communicator per direction, with only the processes aligned along it: it
 * is the group that exchanges the interfaces in the Schur complement. */
static MPI_Comm line_comm[3];

/* The four packets of the halo exchange (two to send, two to receive), kept
 * from one step to the next: the faces always have the same size, so
 * allocating and freeing them at every call was work repeated for nothing. The
 * exchange happens about ten times per step. */
static Real *halo_packets = NULL;
static size_t halo_capacity = 0;

void par_init(int *argc, char ***argv) {
#ifdef USE_OMP
    /*
     * The threads never call MPI: the collectives of the Schur complement and
     * the halo exchange are outside the parallel regions, and the main thread
     * does them. FUNNELED is exactly this promise, and it is the level that
     * every implementation provides without taking internal locks.
     */
    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            fprintf(stderr,
                    "this MPI only goes up to MPI_THREAD_SINGLE: "
                    "rebuild without OMP=1\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
#else
    MPI_Init(argc, argv);
#endif
}

void par_finalize(void) {
    free(halo_packets);
    halo_packets = NULL;
    halo_capacity = 0;
    MPI_Finalize();
}

void par_abort(int code) {
    MPI_Abort(MPI_COMM_WORLD, code);
}

int par_rank(void) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

int par_size(void) {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

void par_topology_init(const int procs[3]) {
    int size = par_size();
    int dims[3] = {procs[0], procs[1], procs[2]};
    int periods[3] = {0, 0, 0};  /* the walls are not periodic */

    /* Fills the directions left at 0 with a balanced decomposition. */
    MPI_Dims_create(size, 3, dims);

    /* reorder = 0: the ranks of the grid stay those of MPI_COMM_WORLD, so
     * par_rank() and the position in the grid never diverge. */
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_get(cart_comm, 3, cart_dims, periods, cart_coords);

    /* MPI_Cart_sub is the Cartesian form of MPI_Comm_split: it keeps a single
     * direction and groups the processes that share the other two. */
    for (int c = 0; c < 3; c++) {
        int keep[3] = {0, 0, 0};
        keep[c] = 1;
        MPI_Cart_sub(cart_comm, keep, &line_comm[c]);
    }
}

static void require_topology(void) {
    if (cart_comm == MPI_COMM_NULL) {
        fprintf(stderr, "par_topology_init must be called first\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

void par_dims(int dims[3]) {
    require_topology();
    for (int c = 0; c < 3; c++) {
        dims[c] = cart_dims[c];
    }
}

void par_coords(int coords[3]) {
    require_topology();
    for (int c = 0; c < 3; c++) {
        coords[c] = cart_coords[c];
    }
}

void par_coords_of(int rank, int coords[3]) {
    require_topology();
    MPI_Cart_coords(cart_comm, rank, 3, coords);
}

int par_neighbor(int axis, int step) {
    int lower;
    int upper;

    require_topology();
    /* MPI returns MPI_PROC_NULL where the grid ends. */
    MPI_Cart_shift(cart_comm, axis, 1, &lower, &upper);

    int neighbor = (step < 0) ? lower : upper;
    return (neighbor == MPI_PROC_NULL) ? PAR_NO_NEIGHBOR : neighbor;
}

long long par_sum_long(long long value) {
    long long total;
    uint64_t begin = time_ns();
    MPI_Allreduce(&value, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    comm_ns += time_ns() - begin;
    return total;
}

long long par_max_long(long long value) {
    long long largest;
    uint64_t begin = time_ns();
    MPI_Allreduce(&value, &largest, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);
    comm_ns += time_ns() - begin;
    return largest;
}

int par_rank_of_max_long(long long value) {
    struct {
        long value;
        int rank;
    } here, global;

    if (value > LONG_MAX) {
        here.value = LONG_MAX;
    } else if (value < LONG_MIN) {
        here.value = LONG_MIN;
    } else {
        here.value = (long)value;
    }
    here.rank = par_rank();

    uint64_t begin = time_ns();
    MPI_Allreduce(&here, &global, 1, MPI_LONG_INT, MPI_MAXLOC,
                  MPI_COMM_WORLD);
    comm_ns += time_ns() - begin;

    return global.rank;
}

Real par_sum_real(Real value) {
    Real total;
    uint64_t begin = time_ns();
    MPI_Allreduce(&value, &total, 1, PAR_REAL, MPI_SUM, MPI_COMM_WORLD);
    comm_ns += time_ns() - begin;
    return total;
}

Real par_max_real(Real value) {
    Real largest;
    uint64_t begin = time_ns();
    MPI_Allreduce(&value, &largest, 1, PAR_REAL, MPI_MAX, MPI_COMM_WORLD);
    comm_ns += time_ns() - begin;
    return largest;
}

/* MPI_PROC_NULL makes a send or a recv towards nothing do nothing, so there is
 * no need to distinguish the edge case. */
static int mpi_neighbor(int axis, int step) {
    int neighbor = par_neighbor(axis, step);
    return (neighbor == PAR_NO_NEIGHBOR) ? MPI_PROC_NULL : neighbor;
}

void par_shift_real(int axis, int step,
                    const Real *send, Real *recv, int count) {
    require_topology();
    uint64_t begin = time_ns();
    MPI_Sendrecv(send, count, PAR_REAL, mpi_neighbor(axis, step), 0,
                 recv, count, PAR_REAL, mpi_neighbor(axis, -step), 0,
                 cart_comm, MPI_STATUS_IGNORE);
    comm_ns += time_ns() - begin;
}

void par_send_real(int axis, int step, const Real *send, int count, int tag) {
    require_topology();
    uint64_t begin = time_ns();
    MPI_Send(send, count, PAR_REAL, mpi_neighbor(axis, step), tag, cart_comm);
    comm_ns += time_ns() - begin;
}

void par_recv_real(int axis, int step, Real *recv, int count, int tag) {
    require_topology();
    uint64_t begin = time_ns();
    MPI_Recv(recv, count, PAR_REAL, mpi_neighbor(axis, step), tag, cart_comm,
             MPI_STATUS_IGNORE);
    comm_ns += time_ns() - begin;
}

void par_line_allgather(int axis, const Real *send, int count, Real *recv) {
    require_topology();
    uint64_t begin = time_ns();
    MPI_Allgather(send, count, PAR_REAL, recv, count, PAR_REAL,
                  line_comm[axis]);
    comm_ns += time_ns() - begin;
}

/*
 * Copies a face of the block between the field and a contiguous packet. `slot`
 * is the index along `axis`: 0 and n-1 are the owned faces, -1 and n are the
 * two rings of boundary cells.
 *
 * MPI could describe a non-contiguous face by itself, with derived types, but
 * internally it would still make this same copy into a contiguous area. Doing
 * it by hand costs the same and is readable.
 */
static void face_copy(const Decomp *d, Real *field, int axis, int slot,
                      Real *packet, int packing) {
    int first = (axis + 1) % 3;
    int second = (axis + 2) % 3;
    int halo = d->halo;
    int cell[3];
    size_t position = 0;

    /*
     * The face also includes the ring in the two transverse directions. It is
     * needed for the edge cells, which belong to a diagonal neighbour and
     * which no exchange would otherwise reach: doing the axes in order, X
     * fills its own ring, Y sends it back together with its own, and Z closes
     * the corners as well. The solver never reads diagonally, but the writing
     * of the files does, on the faces it shares with the neighbours.
     */
    cell[axis] = slot;
    for (int b = -halo; b < d->n[second] + halo; b++) {
        cell[second] = b;
        for (int a = -halo; a < d->n[first] + halo; a++) {
            cell[first] = a;
            size_t offset = decomp_index(d, cell[0], cell[1], cell[2]);

            if (packing) {
                packet[position] = field[offset];
            } else {
                field[offset] = packet[position];
            }
            position++;
        }
    }
}

/* Grows the packets if the largest face does not fit yet. */
static void halo_packets_reserve(size_t face) {
    if (face <= halo_capacity) {
        return;
    }

    free(halo_packets);
    halo_packets = xmalloc(4 * face * sizeof(Real));
    halo_capacity = face;
}

/*
 * The two directions of an axis do not depend on each other, so there is no
 * reason to wait for the first to start the second: all four messages start
 * and one waits just once at the end.
 *
 * The receives are opened before the sends (Lecture MPI, p. 31): this way the
 * message finds the place to go already prepared, and the library does not
 * have to park it elsewhere and then copy it again.
 *
 * The tag says the direction of travel, 0 up and 1 down, so the two receives
 * opened together cannot pick up the wrong message.
 *
 * N.B. This is not overlapping communication and computation: between the
 * start of the messages and the wait there is nothing to compute, and the
 * slides (p.
 * 29) warn that such an overlap requires a network card that takes care of it
 *     by itself. Here the gain comes from the two directions travelling
 *     together.
 */
void par_exchange_halo(const Decomp *d, Real *field) {
    require_topology();

    for (int axis = 0; axis < 3; axis++) {
        int below = mpi_neighbor(axis, -1);
        int above = mpi_neighbor(axis, +1);

        if (below == MPI_PROC_NULL && above == MPI_PROC_NULL) {
            continue;  /* direction not divided: nothing to exchange */
        }

        int face = (d->n[(axis + 1) % 3] + 2 * d->halo) *
                   (d->n[(axis + 2) % 3] + 2 * d->halo);
        halo_packets_reserve((size_t)face);

        Real *to_above = halo_packets;
        Real *to_below = halo_packets + face;
        Real *from_below = halo_packets + 2 * (size_t)face;
        Real *from_above = halo_packets + 3 * (size_t)face;

        /*
         * My last face goes to whoever is above, the first to whoever is
         * below.
         */
        face_copy(d, field, axis, d->n[axis] - 1, to_above, 1);
        face_copy(d, field, axis, 0, to_below, 1);

        MPI_Request wait_for[4];
        uint64_t begin = time_ns();

        MPI_Irecv(from_below, face, PAR_REAL, below, 0, cart_comm,
                  &wait_for[0]);
        MPI_Irecv(from_above, face, PAR_REAL, above, 1, cart_comm,
                  &wait_for[1]);
        MPI_Isend(to_above, face, PAR_REAL, above, 0, cart_comm,
                  &wait_for[2]);
        MPI_Isend(to_below, face, PAR_REAL, below, 1, cart_comm,
                  &wait_for[3]);

        MPI_Waitall(4, wait_for, MPI_STATUSES_IGNORE);
        comm_ns += time_ns() - begin;

        /* Where the neighbour does not exist the domain ends: that ring is
         * left to the boundary conditions and must not be touched. */
        if (below != MPI_PROC_NULL) {
            face_copy(d, field, axis, -1, from_below, 0);
        }
        if (above != MPI_PROC_NULL) {
            face_copy(d, field, axis, d->n[axis], from_above, 0);
        }
    }
}

#else

/* Serial build: a single process, occupying the whole domain. */

void par_init(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
}

void par_finalize(void) {
}

void par_abort(int code) {
    exit(code);
}

int par_rank(void) {
    return 0;
}

int par_size(void) {
    return 1;
}

void par_topology_init(const int procs[3]) {
    (void)procs;
}

void par_dims(int dims[3]) {
    dims[0] = dims[1] = dims[2] = 1;
}

void par_coords(int coords[3]) {
    coords[0] = coords[1] = coords[2] = 0;
}

void par_coords_of(int rank, int coords[3]) {
    (void)rank;
    coords[0] = coords[1] = coords[2] = 0;
}

int par_neighbor(int axis, int step) {
    (void)axis;
    (void)step;
    return PAR_NO_NEIGHBOR;
}

long long par_sum_long(long long value) {
    return value;
}

long long par_max_long(long long value) {
    return value;
}

int par_rank_of_max_long(long long value) {
    (void)value;
    return 0;
}

unsigned long long par_comm_nanoseconds(void) {
    return 0;   /* without MPI there is no communication */
}

Real par_sum_real(Real value) {
    return value;
}

Real par_max_real(Real value) {
    return value;
}

void par_shift_real(int axis, int step,
                    const Real *send, Real *recv, int count) {
    /* No neighbour: the caller keeps what it had put in recv. */
    (void)axis;
    (void)step;
    (void)send;
    (void)recv;
    (void)count;
}

void par_send_real(int axis, int step, const Real *send, int count, int tag) {
    /* No neighbour: there is nobody to send to. */
    (void)axis;
    (void)step;
    (void)send;
    (void)count;
    (void)tag;
}

void par_recv_real(int axis, int step, Real *recv, int count, int tag) {
    /* No neighbour: the caller keeps what it had put in recv. */
    (void)axis;
    (void)step;
    (void)recv;
    (void)count;
    (void)tag;
}

void par_line_allgather(int axis, const Real *send, int count, Real *recv) {
    (void)axis;
    for (int i = 0; i < count; i++) {
        recv[i] = send[i];
    }
}


void par_exchange_halo(const Decomp *d, Real *field) {
    /* No neighbour to copy from: the ring stays as it is. */
    (void)d;
    (void)field;
}

#endif
