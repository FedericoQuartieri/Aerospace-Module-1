# Navier Stokes Brinkman Equation Solver


## Build

Compile the main `solver` executable:

```sh
make solver
```

Enable the explicit SIMD momentum kernels with independently tunable blocks of SIMD vectors:

```sh
make SIMD=1 ZETA_SIMD_VECTORS=4 U_SIMD_VECTORS=8
```

Use `EXTRA_CFLAGS` and `EXTRA_CPPFLAGS` for local overrides without replacing
the backend flags that the Makefile adds:

```sh
make TRIDIAG=pipeline EXTRA_CPPFLAGS="-DDEFAULT_WIDTH=32 -DDEFAULT_STEPS=10"
```

Compile all tests:

```sh
make tests
```

The test executables are created in `build/tests/` and can be run separately:

```sh
./build/tests/paper_man
./build/tests/moving_sphere
./build/tests/channel_obstacle
```

The manufactured tests return a non-zero status when their L2 norms cross
conservative regression thresholds, so they can be used directly in scripts.
Run the fast backend equivalence check with:

```sh
make check
```

## Convergence study

Run the spatial and temporal convergence tests:

```sh
./scripts/run_convergence.sh
TRIDIAG=pipeline SIMD=1 OMP=1 ./scripts/run_convergence.sh
```

Errors and convergence rates are written to `build/convergence/results.csv`.
The script builds through the Makefile, so `TRIDIAG`, `SIMD` and `OMP` in the
environment select the backend and the kernels exactly as `make` would.

![Velocity convergence](docs/convergence/velocity.svg)

![Pressure convergence](docs/convergence/pressure.svg)

Generate and replace the static plots with, reading from `build/convergence/results.csv`:

```sh
./scripts/plot_convergence.py
```

## Parallel run

Build against MPI and run with several processes:

```sh
make MPI=1
mpirun -n 8 ./solver
```

The grid is split into blocks, one per process; `MPI_Dims_create` chooses the
shape unless the tests are given one on the command line.  The tridiagonal
solves that cross a block boundary are completed by the backend `TRIDIAG`
selects, a Schur complement by default, so the answer does not depend on how
many processes are used: `paper_man` prints the same error norms, digit for
digit, from one process up to eight.

The solver accepts an optional configuration file and scenario name:

```sh
./solver
./solver zero_pressure
./solver config.txt paper_data
./solver config.txt zero_pressure
./solver config.txt constant_forcing
```

`paper_data` is the default.  `zero_pressure` and `constant_forcing` mirror
the manufactured cases used by the tests, so a backend can be checked from the
normal solver entry point too.  With one argument the solver first checks
whether it is a known scenario name; if it is not, it treats it as a
configuration file.

## Threads

Build with OpenMP to spread the lines of each block over the cores of one
machine:

```sh
make OMP=1 SIMD=1
OMP_NUM_THREADS=4 ./solver
```

A thread takes whole lines, never part of one, so every sum keeps the order it
had before: the answer is the same digit for digit as with a single thread, the
same way it is across processes, and `paper_man` checks both.

Threads compose with MPI.  `--bind-to none` is needed because `mpirun`
otherwise pins each process to one core, which its threads then share:

```sh
make MPI=1 OMP=1 SIMD=1
OMP_NUM_THREADS=4 mpirun --bind-to none -n 2 ./solver
```

The two are not interchangeable.  A process takes a block, and a direction that
gets split loses the vectorized kernels and turns one local Thomas solve into
three; a thread takes lines, which stay independent however the domain is cut.
The `hybrid` study of the scaling script measures where the balance falls.

For an MPI-free A/B benchmark of the two OpenMP loop layouts, build the scalar
solver with an explicit policy:

```sh
make -B OMP=1 SIMD=0 MPI=0 OMP_SPLIT=planes
make -B OMP=1 SIMD=0 MPI=0 OMP_SPLIT=lines
make -B OMP=1 SIMD=0 MPI=0 OMP_SPLIT=serial  # directional-solver control
```

Forced policies intentionally reject MPI, SIMD and `TRIDIAG=pipeline` builds:
planes are not a valid forced choice across MPI collectives, SIMD would bypass
the scalar line solver along two directions and make the comparison
incomplete, and the pipeline backend has its own loop structure and never
reads the policy.

## Tridiagonal backend

`TRIDIAG` picks *how* a grid line that has been split across processes is
solved.  It is a different question from `MPI`, `OMP` and `SIMD`, which decide
whether the line is split at all, whether threads help inside a block, and how
wide the kernels are.  The four compose.

```sh
make TRIDIAG=schur MPI=1 OMP=1 SIMD=1     # the default
```

| value | method | cost |
|---|---|---|
| `schur` | Schur complement | three local Thomas solves per line; one exchange and one collective per group of lines |
| `pipeline` | pipelined Thomas | one local solve per line; the wait is hidden by sending many independent lines through the processes in batches |

Schur pays in arithmetic, the pipeline pays in latency.  The pipeline wins when
there are many more lines than processes, which is the normal case.

The choice is a directory — `src/tridiag/$(TRIDIAG)/` — not a chain of
`#ifdef`, so exactly one backend is ever compiled and the two cannot silently
drift into each other.  The physics they share lives in
`include/momentum_row.h`: one copy of the formulas, two memory layouts.

Both give the same answer, digit for digit, and the same answer the serial
solver gives: `paper_man` and `zero_pressure` print identical error norms
under either backend from one process up to eight.  That is the property to
check first after touching either one.

For a quick local backend check, run:

```sh
./scripts/check_pipeline.sh
```

It builds Schur once, rebuilds the pipeline with several
`PIPELINE_BATCH_LINES` values, runs `paper_man`, `zero_pressure` and
`constant_forcing_man`, and compares the L2-error values numerically.  Use
`GRID`, `STEPS`, `PIPELINE_BATCHES` and `TOLERANCE` to make the check larger
or broader.  With MPI enabled, `RANKS` controls `mpirun -n` and
`PROCESS_GRID="px py pz"` passes the decomposition shape to the tests.

`PIPELINE_BATCH_LINES` (default 64) sets how many lines travel together.  It
is the pipeline's one tuning knob: small batches fill the pipeline sooner but
send more messages, large ones the opposite.

`SIMD=1` vectorizes the Y and Z sweeps *across* lines: along those axes
consecutive lines are adjacent in the scratch and in the field, so one vector
holds the same level of `SIMD_LANES` neighbouring lines.  It is the kernel the
pipeline branch had, carried over as it was, and it keeps working when that
direction is split between processes, which is exactly what the Schur path
cannot do.  It covers the interior points; the two wall points, the lines left
over at the end of a batch or of a row, and the whole X sweep go through the
scalar path, and the two paths produce the same bits.

`OMP=1` does split the pipeline sweeps now: MPI messages still happen in the
same batch order, while the independent lines inside each batch are shared
among the threads.  This preserves the Thomas dependency along each line and
keeps MPI calls outside the OpenMP regions.

## Scaling study

Measure how much the parallel run gains:

```sh
./scripts/run_scaling.sh          # strong, weak and hybrid, results in build/scaling/
SIMD=0 RESULTS_SUFFIX=_scalar ./scripts/run_scaling.sh
HYBRID=0 ./scripts/run_scaling.sh # skip the hybrid study
./scripts/plot_scaling.py         # figure in docs/scaling/
```

The second run disables the vectorized kernels.  They only apply to directions
that are not split, so comparing with them enabled measures two things at once.
`TRIDIAG=pipeline RESULTS_SUFFIX=_pipeline` measures the other backend: the
script builds through the Makefile, so every `make` variable applies.

![Scaling](docs/scaling/scaling.svg)

### The exhaustive campaign

`scripts/study/` holds six phases that fill the matrix rather than answer one
question at a time --
every backend, with and without SIMD, for every process count, for **every**
shape of the process grid, crossed with every thread count that fits in the
node, over several problem sizes.  1890 cases with the defaults, each run
twice and the best kept, so 3780 solver runs.

| phase | what it sweeps |
|---|---|
| `10_matrix_threads` | one process, threads 1 to 112, both backends, SIMD on and off |
| `11_matrix_mpi` | one thread, every process count, all 80 shapes of the process grid; then the same question again with the *local block* held cubic, and with the block stretched at fixed shape |
| `12_matrix_hybrid` | the full process x thread rectangle, not just the diagonal |
| `13_matrix_batch` | `PIPELINE_BATCH_LINES` from 8 to 4096, against every placement |
| `14_matrix_size` | problem size, strong scaling, weak scaling, peak memory |
| `15_matrix_check` | the error norms: does the whole matrix still solve the same problem? |

```sh
./scripts/run_study.sh submit 1[0-5]   # six chains, PBS runs them in parallel
./scripts/run_study.sh status          # how far along they are
./scripts/run_study.sh merge           # one CSV, then the figures
./scripts/plot_matrix.py               # SVGs in docs/scaling/matrix/
```

Each phase is its own PBS job asking for a whole node, so the six run on six
nodes at once; each resumes where it stopped and re-submits itself until it is
done, so the campaign can span days without anyone watching it.  The axes are
environment variables, so it scales to the time available:

```sh
GRIDS="128 224 256" REPEATS=3 ./scripts/run_study.sh submit 1[0-5]
MATRIX_BACKENDS=pipeline BATCHES="64 1024" ./scripts/run_study.sh submit 13
DRY_RUN=1 ./scripts/study/11_matrix_mpi.sh   # list the cases, run nothing
```

Every phase writes the same 34 columns, so `merge` concatenates the six into
`build/study/all.csv` and `plot_matrix.py` reads it.  Nine of those columns are the stages of a time step timed separately
-- the three momentum sweeps, the three pressure solves, the pressure update,
the permeability fill, and whatever is left unaccounted -- plus the share spent
inside MPI; `matrix-12-composizione.svg` is the figure that reads them.  There is no multi-node sweep: on this cluster nothing launches
processes across nodes -- not `plm tm`, not ssh between compute nodes, not
`pbs_tmrsh` -- so the ceiling is one node.

The measurements run `test/bench.c`, which is the only binary that reads the
grid from a configuration file *and* takes the shape of the process grid on the
command line: which axis gets split is a variable of the study rather than
whatever `MPI_Dims_create` chose.

### Measuring one change

A campaign says how the solver behaves; it does not say what a given commit was
worth.  For that there is a separate A/B:

```sh
./scripts/run_patch_ab.sh                 # HEAD~1 against HEAD
./scripts/run_patch_ab.sh e64ef95 HEAD    # any two revisions
./scripts/plot_patch_ab.py                # two figures in docs/scaling/patch-ab/
```

It extracts both revisions with `git archive` into `build/patch-ab/`, builds
each with the same flags, and runs them **alternately** in the same job, keeping
every repeat rather than the best one.  The CSV carries the same per-stage
columns, which is the point: a change that touches one stage must show up in
that stage and nowhere else, so the untouched stages come along as a control
and their spread is the noise floor of that node.  The figures put the two side
by side, with the control stages greyed.

## Solver structure

`solver_init` allocates the numerical fields and initializes them through the
functions stored in `Data`. Then, `solver_solve` advances the solution for
`STEPS` time steps:

```text
solver_init
    |
    v
time-step loop
    |
    +-- momentum_step
    |      +-- eta: solve along X
    |      +-- zeta: solve along Y
    |      +-- u: solve along Z
    |
    +-- pressure_step
           +-- psi
           +-- phi_low
           +-- phi_high
           +-- pressure update
```

The momentum systems are solved one direction at a time with the Thomas
algorithm for tridiagonal matrices. The pressure correction is similarly
factorized into three directional solves. `momentum.c` and `pressure.c`
implement these two stages, while `physics.c`, `field.c`, and `data.c` provide
the physical terms, field utilities, and problem definition.

## Types

`Real` is the scalar type used by every numerical field. It is `double` by
default and becomes `float` when the code is compiled with `-DUSE_FLOAT`.

```text
ScalarField                         VectorField
+------------------+                +------------------+
| Real *v          |                | Real *v_x        | --> [x0][x1]...[xN]
+------------------+                | Real *v_y        | --> [y0][y1]...[yN]
                                    | Real *v_z        | --> [z0][z1]...[zN]
                                    +------------------+
```

The main structures are:

```text
Data
+-- name                      scenario name used for output
+-- bc_velocity()             boundary velocity
+-- forcing_fn()              forcing term
+-- porosity_fn()             porosity field
+-- porosity_time_dependent   boolean
+-- velocity_fn()             initial/exact velocity
+-- pressure_fn()             initial/exact pressure

SolverMemState
+-- eta, zeta, u, k           VectorField
+-- pressure, pressure_star   ScalarField

SolverStats
+-- execution times for the solver stages, stored in nanoseconds
```

Function pointers in `Data` keep the numerical solver independent from a
specific physical test case. `SolverMemState` groups all
fields that must remain available between time steps.

## Memory management

```text
solver_init
    +-- allocate persistent fields
        +-- 4 VectorField = 12 full-grid arrays
        +-- 2 ScalarField =  2 full-grid arrays

solver_solve
    +-- allocate pressure_buffer   1 full-grid temporary array
    +-- allocate rhs and tmp       2 reusable line/block buffers per thread
    +-- run all time steps
    +-- free pressure_buffer, rhs, and tmp
```
