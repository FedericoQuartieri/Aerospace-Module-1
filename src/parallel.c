/*
 * Le due implementazioni delle funzioni dichiarate in parallel.h: quella
 * vera, che chiama MPI, e quella finta, che descrive un processo solo.
 * Sceglie il compilatore in base a -DUSE_MPI, cioè a `make MPI=1`.
 *
 * È l'unico file del progetto che include <mpi.h>.
 */
#include "parallel.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef USE_MPI

#include <mpi.h>

#ifdef USE_FLOAT
#define PAR_REAL MPI_FLOAT
#else
#define PAR_REAL MPI_DOUBLE
#endif

/* La griglia di processi, creata una volta da par_topology_init. */
static MPI_Comm cart_comm = MPI_COMM_NULL;
static int cart_dims[3] = {1, 1, 1};
static int cart_coords[3] = {0, 0, 0};

/* Un comunicatore per direzione, con i soli processi allineati lungo di essa:
 * è il gruppo che si scambia le interfacce nel complemento di Schur. */
static MPI_Comm line_comm[3];

void par_init(int *argc, char ***argv) {
    MPI_Init(argc, argv);
}

void par_finalize(void) {
    MPI_Finalize();
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
    int periods[3] = {0, 0, 0};  /* le pareti non sono periodiche */

    /* Riempie con una scomposizione equilibrata le direzioni lasciate a 0. */
    MPI_Dims_create(size, 3, dims);

    /* reorder = 0: i rank della griglia restano quelli di MPI_COMM_WORLD,
     * così par_rank() e la posizione nella griglia non divergono mai. */
    MPI_Cart_create(MPI_COMM_WORLD, 3, dims, periods, 0, &cart_comm);
    MPI_Cart_get(cart_comm, 3, cart_dims, periods, cart_coords);

    /* MPI_Cart_sub è la forma cartesiana di MPI_Comm_split: tiene una sola
     * direzione e raggruppa i processi che condividono le altre due. */
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

int par_neighbor(int axis, int step) {
    int lower;
    int upper;

    require_topology();
    /* MPI restituisce MPI_PROC_NULL dove la griglia finisce. */
    MPI_Cart_shift(cart_comm, axis, 1, &lower, &upper);

    int neighbor = (step < 0) ? lower : upper;
    return (neighbor == MPI_PROC_NULL) ? PAR_NO_NEIGHBOR : neighbor;
}

long long par_sum_long(long long value) {
    long long total;
    MPI_Allreduce(&value, &total, 1, MPI_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

Real par_max_real(Real value) {
    Real largest;
    MPI_Allreduce(&value, &largest, 1, PAR_REAL, MPI_MAX, MPI_COMM_WORLD);
    return largest;
}

/* MPI_PROC_NULL fa sì che una send o una recv verso il vuoto non facciano
 * niente, così non serve distinguere il caso del bordo. */
static int mpi_neighbor(int axis, int step) {
    int neighbor = par_neighbor(axis, step);
    return (neighbor == PAR_NO_NEIGHBOR) ? MPI_PROC_NULL : neighbor;
}

void par_shift_real(int axis, int step,
                    const Real *send, Real *recv, int count) {
    require_topology();
    MPI_Sendrecv(send, count, PAR_REAL, mpi_neighbor(axis, step), 0,
                 recv, count, PAR_REAL, mpi_neighbor(axis, -step), 0,
                 cart_comm, MPI_STATUS_IGNORE);
}

void par_line_allgather(int axis, const Real *send, int count, Real *recv) {
    require_topology();
    MPI_Allgather(send, count, PAR_REAL, recv, count, PAR_REAL,
                  line_comm[axis]);
}

/*
 * Copia una faccia del blocco fra il campo e un pacchetto contiguo.
 * `slot` è l'indice lungo `axis`: 0 e n-1 sono le facce possedute,
 * -1 e n sono i due anelli di celle di contorno.
 *
 * MPI saprebbe descrivere una faccia non contigua da sé, con i tipi derivati,
 * ma internamente farebbe comunque questa stessa copia in un'area contigua.
 * Farla a mano costa uguale e si legge.
 */
static void face_copy(const Decomp *d, Real *field, int axis, int slot,
                      Real *packet, int packing) {
    int first = (axis + 1) % 3;
    int second = (axis + 2) % 3;
    int cell[3];
    size_t position = 0;

    cell[axis] = slot;
    for (int b = 0; b < d->n[second]; b++) {
        cell[second] = b;
        for (int a = 0; a < d->n[first]; a++) {
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

void par_exchange_halo(const Decomp *d, Real *field) {
    require_topology();

    for (int axis = 0; axis < 3; axis++) {
        int below = par_neighbor(axis, -1);
        int above = par_neighbor(axis, +1);

        if (below == PAR_NO_NEIGHBOR && above == PAR_NO_NEIGHBOR) {
            continue;  /* direzione non divisa: niente da scambiare */
        }

        int face = d->n[(axis + 1) % 3] * d->n[(axis + 2) % 3];
        Real *packet = xmalloc((size_t)face * sizeof(Real));
        Real *arrived = xmalloc((size_t)face * sizeof(Real));

        /* La mia ultima faccia va a chi sta sopra; dal basso arriva la sua. */
        face_copy(d, field, axis, d->n[axis] - 1, packet, 1);
        par_shift_real(axis, +1, packet, arrived, face);
        if (below != PAR_NO_NEIGHBOR) {
            face_copy(d, field, axis, -1, arrived, 0);
        }

        /* E specularmente verso il basso. */
        face_copy(d, field, axis, 0, packet, 1);
        par_shift_real(axis, -1, packet, arrived, face);
        if (above != PAR_NO_NEIGHBOR) {
            face_copy(d, field, axis, d->n[axis], arrived, 0);
        }

        free(arrived);
        free(packet);
    }
}

#else

/* Costruzione seriale: un processo solo, che occupa tutto il dominio. */

void par_init(int *argc, char ***argv) {
    (void)argc;
    (void)argv;
}

void par_finalize(void) {
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

int par_neighbor(int axis, int step) {
    (void)axis;
    (void)step;
    return PAR_NO_NEIGHBOR;
}

long long par_sum_long(long long value) {
    return value;
}

Real par_max_real(Real value) {
    return value;
}

void par_shift_real(int axis, int step,
                    const Real *send, Real *recv, int count) {
    /* Nessun vicino: chi chiama tiene quello che aveva messo in recv. */
    (void)axis;
    (void)step;
    (void)send;
    (void)recv;
    (void)count;
}

void par_line_allgather(int axis, const Real *send, int count, Real *recv) {
    (void)axis;
    for (int i = 0; i < count; i++) {
        recv[i] = send[i];
    }
}


void par_exchange_halo(const Decomp *d, Real *field) {
    /* Nessun vicino da cui copiare: l'anello resta com'è. */
    (void)d;
    (void)field;
}

#endif
