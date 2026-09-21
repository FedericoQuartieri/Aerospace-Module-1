#ifndef WORKERS_H
#define WORKERS_H

#include <stdbool.h>

/*
 * Policy used by the line-based solvers. AUTO is the normal behaviour; PLANES,
 * LINES and SERIAL exist only to compare the two loop structures (and the
 * control with neither) on the same problem, with one process and without the
 * SIMD kernels.
 *
 * The symbolic values make it possible to compile, for example, with
 *   -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_LINES
 * without leaving magic numbers in the measurement scripts.
 */
/* It starts from 1: in the preprocessor an unknown identifier evaluates to 0,
 * so a misspelt name must fall into the check below. */
#define WORKERS_LINE_POLICY_AUTO   1
#define WORKERS_LINE_POLICY_PLANES 2
#define WORKERS_LINE_POLICY_LINES  3
#define WORKERS_LINE_POLICY_SERIAL 4

#ifndef WORKERS_LINE_POLICY
#define WORKERS_LINE_POLICY WORKERS_LINE_POLICY_AUTO
#endif

#if WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_AUTO && \
    WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_PLANES && \
    WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_LINES && \
    WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_SERIAL
#error "WORKERS_LINE_POLICY must be AUTO, PLANES, LINES or SERIAL"
#endif

/* A forced policy is a local experiment, not a mode of the distributed solver.
 * PLANES is not safe when a direction crosses several ranks; SIMD, on the
 * other hand, would override this choice along Y and Z and make the comparison
 * incomplete. */
#if WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_AUTO && !defined(USE_OMP)
#error "a forced WORKERS_LINE_POLICY requires USE_OMP"
#endif
#if WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_AUTO && defined(USE_MPI)
#error "a forced WORKERS_LINE_POLICY is only valid without USE_MPI"
#endif
#if WORKERS_LINE_POLICY != WORKERS_LINE_POLICY_AUTO && defined(USE_SIMD)
#error "a forced WORKERS_LINE_POLICY is only valid without USE_SIMD"
#endif

/*
 * The compute threads, and the little that is needed so they do not have to be
 * named everywhere.
 *
 * The shared-memory parallelism of this solver has a single form: the lines
 * along which the tridiagonal systems are solved are independent of each
 * other, so they are shared out among the threads. A line is never split: its
 * sums stay in the same order, and the result stays identical digit for digit
 * to that of a single thread. It is the same property that the Schur
 * complement guarantees between processes, and it is verified in the same way.
 *
 * MPI stays on the main thread: outside the parallel regions or inside
 * WORKERS_MASTER in the persistent teams of the backends, so
 * MPI_THREAD_FUNNELED is enough and no implementation needs internal locks.
 *
 * Without -DUSE_OMP everything in here describes a single thread and the
 * directives disappear, so the serial build stays what it was.
 */

#ifdef USE_OMP

#include <omp.h>

#define WORKERS_PRAGMA(x) _Pragma(#x)
/* schedule(static): the planes all cost the same, and the fixed partition
 * makes the distribution reproducible from one run to the next. */
#define WORKERS_PARALLEL_FOR(cond) \
    WORKERS_PRAGMA(omp parallel for schedule(static) if (cond))

/*
 * The same thing, but distributing two nested loops instead of one.
 *
 * It is needed where the outer loop alone can have fewer iterations than there
 * are threads. Filling a field walks the planes along Z, and when MPI splits
 * exactly that direction a block can have only a handful: with 56 threads and
 * 4 planes, fifty-two threads would receive nothing. By collapsing the two
 * loops the rows are distributed, and there are always enough of them.
 *
 * The two loops must be perfectly nested -- nothing between the brace of the
 * first and the `for' of the second -- otherwise collapse does not apply.
 */
#define WORKERS_PARALLEL_FOR_2(cond) \
    WORKERS_PRAGMA(omp parallel for schedule(static) collapse(2) if (cond))

/*
 * A team opened only once, with several distributed loops inside it.
 *
 * WORKERS_PARALLEL_FOR opens and closes a team at every loop, which is fine
 * when there is a single loop. Where instead there are two separated by an MPI
 * call -- assemble, communicate, write back -- opening one for each means
 * opening two per plane, and there are hundreds of planes.
 *
 * With these the team is opened outside the loop over the planes and stays
 * alive: inside, WORKERS_FOR distributes the lines and WORKERS_MASTER isolates
 * the MPI call to a single thread, which, the team being at the outermost
 * level, is the same one that called MPI_Init. This is exactly what
 * MPI_THREAD_FUNNELED requires, so the thread-safety level does not change.
 *
 * Beware of two OpenMP asymmetries, because getting them wrong gives no
 * compile errors but wrong results:
 *   - WORKERS_FOR has an implicit barrier on EXIT, not on entry;
 *   - WORKERS_MASTER has no barrier at all, neither before nor after.
 * So after the master an explicit WORKERS_BARRIER is always needed, otherwise
 * the other threads start reading what the master is still writing.
 */
#define WORKERS_PARALLEL(cond) WORKERS_PRAGMA(omp parallel if (cond))
#define WORKERS_FOR WORKERS_PRAGMA(omp for schedule(static))
#define WORKERS_MASTER WORKERS_PRAGMA(omp master)
#define WORKERS_BARRIER WORKERS_PRAGMA(omp barrier)

static inline int workers_available(void) { return omp_get_max_threads(); }
static inline int workers_id(void) { return omp_get_thread_num(); }

#else

/* Without OpenMP nobody reads the condition, but it must still be consumed:
 * this way the code that computes it gets no unused-variable warning. */
#define WORKERS_PARALLEL_FOR(cond) (void)(cond);
#define WORKERS_PARALLEL_FOR_2(cond) (void)(cond);

/* Without OpenMP the block that follows WORKERS_PARALLEL is just a block, and
 * the loops inside run one after the other: that is already the intended
 * behaviour. */
#define WORKERS_PARALLEL(cond) (void)(cond);
#define WORKERS_FOR
#define WORKERS_MASTER
#define WORKERS_BARRIER

static inline int workers_available(void) { return 1; }
static inline int workers_id(void) { return 0; }

#endif

/* If there is only one thread no parallel region is opened: opening it costs
 * something anyway, and there is nothing to distribute. */
static inline bool workers_many(void) {
    return workers_available() > 1;
}

/*
 * How many sets of work arrays are needed for `items` independent iterations:
 * one per thread, but without wasting more than there are iterations.
 * `allowed` is false where communication forces the work to proceed in
 * sequence.
 */
static inline int workers_slots(int allowed, int items) {
    int workers;

    if (!allowed || items < 2) {
        return 1;
    }
    workers = workers_available();
    return workers < items ? workers : items;
}

/*
 * The two things cannot be chosen separately: when distributing the planes one
 * scratch slot is needed for each active thread; when distributing the lines
 * everyone works on disjoint portions of a single shared slot.
 */
typedef struct {
    int slots;
    bool split_lines;
} WorkersLineSchedule;

static inline WorkersLineSchedule workers_line_schedule(bool planes_allowed,
                                                         int planes) {
    WorkersLineSchedule schedule;

#if WORKERS_LINE_POLICY == WORKERS_LINE_POLICY_PLANES
    (void)planes_allowed;
    schedule.slots = workers_slots(true, planes);
    schedule.split_lines = false;
#elif WORKERS_LINE_POLICY == WORKERS_LINE_POLICY_LINES
    (void)planes_allowed;
    (void)planes;
    schedule.slots = 1;
    /* It really forces the LINES branch even with OMP_NUM_THREADS=1: that
     * point is the check of the structural cost of the two implementations. */
    schedule.split_lines = true;
#elif WORKERS_LINE_POLICY == WORKERS_LINE_POLICY_SERIAL
    (void)planes_allowed;
    (void)planes;
    /* Check on the value of the LINES branch: the rest of the solver keeps the
     * threads, but the directional solvers walk the planes in series. */
    schedule.slots = 1;
    schedule.split_lines = false;
#else
    schedule.slots = workers_slots(planes_allowed, planes);
    schedule.split_lines = (schedule.slots < 2) && workers_many();
#endif

    return schedule;
}

static inline const char *workers_line_policy_name(void) {
#if WORKERS_LINE_POLICY == WORKERS_LINE_POLICY_PLANES
    return "planes";
#elif WORKERS_LINE_POLICY == WORKERS_LINE_POLICY_LINES
    return "lines";
#elif WORKERS_LINE_POLICY == WORKERS_LINE_POLICY_SERIAL
    return "serial";
#else
    return "auto";
#endif
}

/* The caller's slot, valid also outside a parallel region. */
static inline int workers_slot(int slots) {
    int id;

    if (slots < 2) {
        return 0;
    }
    id = workers_id();
    return id < slots ? id : slots - 1;
}

#endif
