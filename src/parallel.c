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

/* Tempo speso dentro MPI, per il rendiconto di fine simulazione. */
static unsigned long long comm_ns = 0;

unsigned long long par_comm_nanoseconds(void) {
    return comm_ns;
}

/* La griglia di processi, creata una volta da par_topology_init. */
static MPI_Comm cart_comm = MPI_COMM_NULL;
static int cart_dims[3] = {1, 1, 1};
static int cart_coords[3] = {0, 0, 0};

/* Un comunicatore per direzione, con i soli processi allineati lungo di essa:
 * è il gruppo che si scambia le interfacce nel complemento di Schur. */
static MPI_Comm line_comm[3];

/* I quattro pacchetti dello scambio degli aloni (due da spedire, due da
 * ricevere), tenuti da un passo all'altro: le facce hanno sempre la stessa
 * dimensione, quindi allocarle e liberarle a ogni chiamata era lavoro
 * ripetuto per niente.  Lo scambio avviene una decina di volte per passo. */
static Real *halo_packets = NULL;
static size_t halo_capacity = 0;

void par_init(int *argc, char ***argv) {
#ifdef USE_OMP
    /*
     * I thread non chiamano mai MPI: le collettive del complemento di Schur e
     * lo scambio degli aloni stanno fuori dalle regioni parallele, e le fa il
     * thread principale.  FUNNELED e' esattamente questa promessa, ed e' il
     * livello che ogni implementazione fornisce senza prendere lock interni.
     */
    int provided = MPI_THREAD_SINGLE;

    MPI_Init_thread(argc, argv, MPI_THREAD_FUNNELED, &provided);

    if (provided < MPI_THREAD_FUNNELED) {
        int rank = 0;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            fprintf(stderr,
                    "questa MPI si ferma a MPI_THREAD_SINGLE: "
                    "ricompila senza OMP=1\n");
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

void par_coords_of(int rank, int coords[3]) {
    require_topology();
    MPI_Cart_coords(cart_comm, rank, 3, coords);
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

/* MPI_PROC_NULL fa sì che una send o una recv verso il vuoto non facciano
 * niente, così non serve distinguere il caso del bordo. */
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

/* Cresce i pacchetti se la faccia più grande non ci sta ancora. */
static void halo_packets_reserve(size_t face) {
    if (face <= halo_capacity) {
        return;
    }

    free(halo_packets);
    halo_packets = xmalloc(4 * face * sizeof(Real));
    halo_capacity = face;
}

/*
 * Le due direzioni di un asse non dipendono l'una dall'altra, quindi non c'è
 * motivo di aspettare la prima per cominciare la seconda: i quattro messaggi
 * partono tutti e si attende una volta sola alla fine.
 *
 * Le ricezioni sono aperte prima delle spedizioni (Lecture MPI, p. 31): così
 * il messaggio trova già pronto il posto dove andare, e la libreria non deve
 * parcheggiarlo altrove per poi ricopiarlo.
 *
 * Il tag dice il verso di marcia, 0 in su e 1 in giù, così le due ricezioni
 * aperte insieme non possono raccogliere il messaggio sbagliato.
 *
 * N.B. Questo non è sovrapporre comunicazione e calcolo: fra la partenza dei
 * messaggi e l'attesa non c'è nulla da calcolare, e le slide (p. 29) avvertono
 * che quella sovrapposizione richiede una scheda di rete che se ne occupi da
 * sola.  Qui si guadagna perché i due versi viaggiano insieme.
 */
void par_exchange_halo(const Decomp *d, Real *field) {
    require_topology();

    for (int axis = 0; axis < 3; axis++) {
        int below = mpi_neighbor(axis, -1);
        int above = mpi_neighbor(axis, +1);

        if (below == MPI_PROC_NULL && above == MPI_PROC_NULL) {
            continue;  /* direzione non divisa: niente da scambiare */
        }

        int face = d->n[(axis + 1) % 3] * d->n[(axis + 2) % 3];
        halo_packets_reserve((size_t)face);

        Real *to_above = halo_packets;
        Real *to_below = halo_packets + face;
        Real *from_below = halo_packets + 2 * (size_t)face;
        Real *from_above = halo_packets + 3 * (size_t)face;

        /* La mia ultima faccia va a chi sta sopra, la prima a chi sta sotto. */
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

        /* Dove il vicino non c'è finisce il dominio: quell'anello resta alle
         * condizioni al contorno e non va toccato. */
        if (below != MPI_PROC_NULL) {
            face_copy(d, field, axis, -1, from_below, 0);
        }
        if (above != MPI_PROC_NULL) {
            face_copy(d, field, axis, d->n[axis], from_above, 0);
        }
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

unsigned long long par_comm_nanoseconds(void) {
    return 0;   /* senza MPI non si comunica */
}

Real par_sum_real(Real value) {
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

void par_send_real(int axis, int step, const Real *send, int count, int tag) {
    /* Nessun vicino: non c'e' nessuno a cui mandare. */
    (void)axis;
    (void)step;
    (void)send;
    (void)count;
    (void)tag;
}

void par_recv_real(int axis, int step, Real *recv, int count, int tag) {
    /* Nessun vicino: chi chiama tiene quello che aveva messo in recv. */
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
    /* Nessun vicino da cui copiare: l'anello resta com'è. */
    (void)d;
    (void)field;
}

#endif
