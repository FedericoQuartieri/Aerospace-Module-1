#ifndef PARALLEL_H
#define PARALLEL_H

#include "decomp.h"
#include "types.h"

/*
 * Layer that isolates MPI from the rest of the program.
 *
 * The solver never includes <mpi.h>: it just calls these functions. This way
 * all the parallel code lives in a single file, and whoever reads knows where
 * to look. In return the serial version keeps compiling and running as before,
 * and remains the reference against which to compare the parallel results.
 *
 * Without -DUSE_MPI the functions below are stubs that describe a single
 * process, so the serial program pays nothing. With MPI enabled, on the other
 * hand, par_topology_init and decomp_init_mpi really divide the grid into
 * blocks: the rest of the solver sees only its own local Decomp.
 */

/* Starts and shuts down MPI. argc and argv may be NULL. */
void par_init(int *argc, char ***argv);
void par_finalize(void);

/* Stops all processes with the same error code. */
void par_abort(int code);

/* Number of this process, from 0 to par_size() - 1. */
int par_rank(void);

/* How many processes are running. */
int par_size(void);

/* Value returned when there is no neighbour on that side. */
#define PAR_NO_NEIGHBOR (-1)

/*
 * Arranges the processes on a 3D grid, which is how the computational grid
 * will be divided. Putting 0 in procs[c] lets MPI choose how many processes to
 * place along that direction: procs = {1, 1, 0} therefore gives slabs along Z.
 *
 * It must be called after par_init and before decomp_init_mpi.
 */
void par_topology_init(const int procs[3]);

/* How many processes there are along each direction. */
void par_dims(int dims[3]);

/* Where I am inside that process grid. */
void par_coords(int coords[3]);

/*
 * Where process `rank` is. Needed by whoever has to reason about the blocks of
 * the others without asking them: knowing the coordinates, decomp_share can
 * tell on its own which cells it owns.
 */
void par_coords_of(int rank, int coords[3]);

/*
 * Rank of the neighbour along direction `axis` (0 = X, 1 = Y, 2 = Z), with
 * step -1 towards the bottom and +1 towards the top. Returns PAR_NO_NEIGHBOR
 * if on that side there is the wall of the domain.
 */
int par_neighbor(int axis, int step);

/* Sum and maximum of an integer over all processes, returned to everyone. */
long long par_sum_long(long long value);
long long par_max_long(long long value);

/* Rank that owns the maximum value; in case of a tie it chooses the lowest. */
int par_rank_of_max_long(long long value);

/*
 * Nanoseconds spent so far inside MPI calls.
 *
 * It answers the question that matters when measuring scaling: how much of the
 * time goes into communicating instead of computing. It counts only the time
 * inside MPI, not the time to prepare the packets.
 */
unsigned long long par_comm_nanoseconds(void);

/* Sum and maximum over all processes, returned to everyone. */
Real par_sum_real(Real value);
Real par_max_real(Real value);

/*
 * Sends `count` numbers to the neighbour in direction `step` along `axis` and
 * receives as many from the neighbour on the opposite side. Where the
 * neighbour does not exist nothing happens and `recv` stays as it was.
 *
 * It uses MPI_Sendrecv: two processes facing each other exchange data at the
 * same instant, and with separate Send and Recv they would block each other as
 * soon as the messages exceed the threshold beyond which MPI stops buffering
 * them.
 */
void par_shift_real(int axis, int step,
                    const Real *send, Real *recv, int count);

/*
 * Separate send and receive towards the neighbour in direction `step` along
 * `axis`.
 *
 * par_shift_real is not enough for everyone: a symmetric exchange assumes that
 * the two processes facing each other both have something to tell each other
 * at the same instant. In a chain this is not so: the pipelined Thomas
 * receives from the neighbour below, works, and only afterwards sends to the
 * one above. The two halves must be separated because there is computation in
 * between, and that is exactly what keeps the other processes busy while they
 * wait.
 *
 * `tag` distinguishes the messages travelling together on the same pair of
 * neighbours: in the pipeline they are the axis, the component and the
 * direction of the sweep. Where the neighbour does not exist nothing happens,
 * and `recv` stays as it was.
 *
 * They are blocking on purpose: it is the wait that sets the rhythm of the
 * pipeline.
 */
void par_send_real(int axis, int step, const Real *send, int count, int tag);
void par_recv_real(int axis, int step, Real *recv, int count, int tag);

/*
 * Gathers `count` numbers from each of the processes aligned along `axis` and
 * hands everyone the complete vector, ordered by position along the axis.
 * `recv` must have room for count * (processes along axis) numbers.
 */
void par_line_allgather(int axis, const Real *send, int count, Real *recv);

/*
 * Fills the boundary cells of a field with the values of the neighbours.
 *
 * Each process keeps an extra ring of cells all around its own block. They do
 * not belong to it: they are a copy of the last row of cells of the neighbour,
 * and they are needed so that the computations that look one cell further can
 * also be done on the edge of the block, without asking anyone for anything.
 *
 * It must be called again every time the field changes and is about to be read
 * again. Where the neighbour does not exist the ring stays as it was: there is
 * the wall of the domain there, and the boundary conditions already deal with
 * it themselves.
 */
void par_exchange_halo(const Decomp *d, Real *field);

#endif
