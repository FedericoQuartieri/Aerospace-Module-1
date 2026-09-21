# A Parallel Solver for the Unsteady Stokes–Brinkman Equations

The solver advances the unsteady Stokes–Brinkman system on a 3D staggered
grid:

```text
du/dt - nu Laplacian(u) + (nu / K) u + grad(p) = f,    div(u) = 0
```

The permeability `K(x, t)` models obstacles and porous regions inside the
fluid. The convective term of Navier–Stokes is neglected, so the problem is
linear. Time stepping uses the direction-splitting fractional-step scheme of
Guermond and Minev, with Crank–Nicolson for the momentum equation. As a
result, every substep reduces to independent tridiagonal systems, one per
grid line, solved with the Thomas algorithm. The domain is split into blocks
over MPI processes. OpenMP threads share the lines of each block, and
optional SIMD kernels solve neighbouring lines together.

The report, with the method and the results, lives in the `report/`
submodule:

```sh
git submodule update --init
```

## Repository layout

| path | contents |
|---|---|
| `src/` | the shared solver: time loop, fields, physics, scenarios, MPI, output |
| `src/tridiag/schur/`, `src/tridiag/pipeline/` | the two tridiagonal backends, one compiled at a time |
| `include/` | shared headers; `momentum_row.h` holds the momentum formulas both backends use |
| `test/` | tests, the physical scenarios and the `bench` benchmark binary |
| `scripts/` | studies, checks and figures; `scripts/study/` is the scaling campaign |
| `docs/` | committed results and figures, and the course slides in `docs/slides/` |
| `report/` | the report (LaTeX, git submodule) |
| `quantum/qsvt_solver.py` | a standalone Python/Qiskit prototype that solves the tridiagonal stages with block encoding and QSVT; it does not use the C code |
| `config.txt` | an example configuration file |
| `all.csv` | the merged results of the scaling campaign, read by the report figures |

## Requirements

- A C11 compiler (`cc`, GNU C11 mode) and GNU Make.
- `MPI=1`: an MPI implementation that provides `mpicc` and `mpirun`.
- `OMP=1`: a compiler with OpenMP support.
- `SIMD=1`: on x86-64 the build adds `-mavx2`, so the CPU must support AVX2.
- Python 3.9 or newer for the scripts. The SVG plots use only the standard
  library. The report figures (`plot_paper_*.py`, `plot_brinkman.py`) need
  matplotlib, and `quantum/` needs numpy, scipy and qiskit.
- ParaView (`pvpython`) for the flow images, and PBS for the scaling
  campaign.

## Build

Compile the main `solver` executable (the default target):

```sh
make solver
```

The build is configured with `make` variables, which combine freely:

| variable | values | effect |
|---|---|---|
| `MPI` | `0` (default), `1` | build with `mpicc` and split the grid over processes |
| `OMP` | `0` (default), `1` | spread the lines of each block over OpenMP threads |
| `SIMD` | `0` (default), `1` | explicit SIMD kernels for the momentum solves |
| `ZETA_SIMD_VECTORS`, `U_SIMD_VECTORS` | `4`, `8` (defaults) | SIMD vectors per block in the Y and Z momentum kernels |
| `TRIDIAG` | `schur` (default), `pipeline` | how a grid line split across processes is solved |
| `PIPELINE_BATCH_LINES` | empty (default), `N` | fix the pipeline batch in the binary |
| `OMP_SPLIT` | `auto` (default), `planes`, `lines`, `serial` | force an OpenMP loop layout, for benchmarks only |
| `EXTRA_CFLAGS`, `EXTRA_CPPFLAGS` | flags | local additions that keep the flags the Makefile adds |

Enable the explicit SIMD momentum kernels with independently tunable blocks
of SIMD vectors:

```sh
make SIMD=1 ZETA_SIMD_VECTORS=4 U_SIMD_VECTORS=8
```

Grid, domain and time step have compile-time defaults: a 128^3 grid on
[0, pi]^3, 200 steps to T = 1, `nu = 1` and output every 5 steps. They can be
changed with `-DDEFAULT_WIDTH`, `-DDEFAULT_HEIGHT`, `-DDEFAULT_DEPTH`,
`-DDEFAULT_LX`/`LY`/`LZ`, `-DDEFAULT_T`, `-DDEFAULT_STEPS`, `-DDEFAULT_NU` and
`-DDEFAULT_WR_FREQ`. `-DUSE_FLOAT` switches every field from `double` to
`float`:

```sh
make TRIDIAG=pipeline EXTRA_CPPFLAGS="-DDEFAULT_WIDTH=32 -DDEFAULT_STEPS=10"
```

Each configuration builds into its own `build/variants/<build-id>/`
directory, where the id is a hash of compiler, compiler version and flags.
Changing backend, batch, compiler or flags therefore never needs
`make clean`. `./solver` and `build/tests/*` are convenience copies of the
last build. Scripts that run several configurations at once use the variant
paths instead, printed by `make <variables> print-build-id` and
`make <variables> print-test-dir`. Every compilation writes to a temporary
file and renames it, so concurrent builds never see partial output.

## Running the solver

```sh
./solver                              # defaults, paper_data scenario
./solver zero_pressure
./solver config.txt paper_data
./solver config.txt zero_pressure
./solver config.txt constant_forcing
```

`paper_data` is the default scenario. `zero_pressure` and `constant_forcing`
mirror the manufactured cases used by the tests, so a backend can be checked
from the normal solver entry point too. With one argument, the solver first
checks whether it is a known scenario name. If it is not, the solver reads
it as a configuration file.

A configuration file has one `key = value` per line. Lines starting with `#`
are comments, and a missing key keeps its compile-time default. An unknown
key or a malformed line stops the program and reports the file and line.

| key | meaning |
|---|---|
| `width`, `height`, `depth` | grid cells along X, Y, Z |
| `lx`, `ly`, `lz` | domain size |
| `t_end`, `steps` | final time and number of steps; `dt = t_end / steps` |
| `nu` | kinematic viscosity |
| `wr_freq` | output frequency in steps |
| `pipeline_batch_lines` | pipeline batch; `0` picks it at start-up, the Schur backend ignores it |

`./solver` does not write output files. It prints the time spent in each
stage of the step. The flow fields come from the physical scenarios in the
tests, which write VTK files (`.pvti` / `.vti`) for ParaView.

## Tests

Compile all tests:

```sh
make tests
```

The test executables are created in `build/tests/` and can be run
separately:

```sh
./build/tests/paper_man
./build/tests/moving_sphere
./build/tests/channel_obstacle
```

| test | what it checks |
|---|---|
| `paper_man` | manufactured solution from the reference paper, with time-dependent permeability |
| `zero_pressure` | manufactured solution with zero pressure |
| `constant_forcing_man` | manufactured solution with a constant force and a linear pressure |
| `brinkman_channel` | Brinkman flow in a plane channel, against its exact solution |
| `cavity` | lid-driven cavity (writes VTK output) |
| `channel_obstacle` | channel with a wall-attached, inclined obstacle of low permeability (writes VTK output) |
| `moving_sphere` | channel with a rigid sphere moving along it (writes VTK output) |
| `decomp_layout` | the same run on a compact and on a padded memory layout gives bit-identical fields |
| `decomp_mpi` | the process blocks cover the grid exactly once |
| `halo_mpi` | halo cells receive exactly the neighbour's data |
| `solver_snapshot` | binary dump of all fields, used by `check_pipeline.sh` |
| `bench` | the benchmark binary of the scaling studies |
| `tridiag_blocks`, `tridiag_mpi` | Schur backend only: the Schur complement against plain Thomas, in one process and over MPI |
| `pipeline_residual` | pipeline backend only: equation residuals, including MPI junctions and one-cell blocks |

The manufactured tests return a non-zero status when an L2 norm crosses a
conservative regression threshold, so they can be used directly in scripts.
`paper_man`, `zero_pressure`, `constant_forcing_man`, `decomp_mpi` and
`halo_mpi` accept a process grid as `px py pz` on the command line.
Otherwise `MPI_Dims_create` chooses it.

`brinkman_channel` checks the regime of the obstacles, where the permeability
is small or jumps, against the exact solution of Brinkman flow in a plane
channel. It fails when the relative L2 error of the velocity exceeds 5%:

```sh
./build/tests/brinkman_channel                     # uniform K = 1e-2
./build/tests/brinkman_channel layer 1e-4          # porous layer, free fluid above
./scripts/run_brinkman.sh                          # grids x permeabilities, into docs/brinkman/
```

`make check` runs the backend equivalence check, `./scripts/check_pipeline.sh`
(see [Tridiagonal backends](#tridiagonal-backends)).

## Results and figures of the report

| command | output |
|---|---|
| `./scripts/run_equivalence.sh` | runs `paper_man` under both backends, with and without MPI, threads and SIMD, and tabulates the norms in `docs/equivalence/` |
| `./scripts/run_brinkman.sh` | the `brinkman_channel` study in `docs/brinkman/`; `./scripts/plot_brinkman.py` draws `docs/brinkman/brinkman.pdf` |
| `./scripts/run_figures.sh [case]` | runs `cavity`, `channel_obstacle` and `moving_sphere` into `data/output/<case>/` for ParaView (`RANKS`, default 4, sets the processes) |
| `./scripts/paraview/report_images.sh` | turns those runs into the mid-plane and perspective images in `report/figures/`; set `PVPYTHON` if `pvpython` is not on the path |
| `./scripts/plot_paper_convergence.py` | `report/figures/convergence.pdf`, from `docs/convergence/results.csv` |
| `./scripts/plot_paper_scaling.py` | `report/figures/scaling.pdf`, from `all.csv` |
| `./scripts/plot_paper_matrix.py` | `report/figures/hybrid.pdf` and `shapes.pdf`, from `all.csv` |

## Convergence study

Run the spatial and temporal convergence tests:

```sh
./scripts/run_convergence.sh
TRIDIAG=pipeline SIMD=1 OMP=1 ./scripts/run_convergence.sh
qsub scripts/run_convergence_cluster.sh     # the same study as a PBS job
```

Both studies go up to a 256^3 grid. Errors and convergence rates are written
to `build/convergence/results.csv`. The copy the report uses, from the
cluster run, is `docs/convergence/results.csv`. The script builds through the
Makefile, so `TRIDIAG`, `SIMD`, `OMP`, `EXTRA_CFLAGS` and `EXTRA_CPPFLAGS` in
the environment select the backend and the kernels exactly as `make` would.

![Velocity convergence](docs/convergence/velocity.svg)

![Pressure convergence](docs/convergence/pressure.svg)

Regenerate the static plots in `docs/convergence/` from
`build/convergence/results.csv`:

```sh
./scripts/plot_convergence.py
```

## Parallel run

Build against MPI and run with several processes:

```sh
make MPI=1
mpirun -n 8 ./solver
```

The grid is split into blocks, one per process. The tridiagonal solves that
cross a block boundary are completed by the backend that `TRIDIAG` selects,
the Schur complement by default. The answer does not depend on the number of
processes beyond round-off. From one process to eight, the error norms of
`paper_man` agree to about 13 significant digits (see `docs/equivalence/`).

## Threads

Build with OpenMP to spread the lines of each block over the cores of one
machine:

```sh
make OMP=1 SIMD=1
OMP_NUM_THREADS=4 ./solver
```

A thread takes whole lines, never part of one, so every sum keeps its order.
The result is bit-identical to a single-thread run, and `docs/equivalence/`
confirms this for both backends.

Threads compose with MPI. `--bind-to none` is needed because `mpirun`
otherwise pins each process to one core, which its threads then share:

```sh
make MPI=1 OMP=1 SIMD=1
OMP_NUM_THREADS=4 mpirun --bind-to none -n 2 ./solver
```

The two are not interchangeable. A process takes a block. With the Schur
backend, a direction that gets split loses the vectorized kernels and turns
one local Thomas solve into three. A thread takes lines, which stay
independent however the domain is cut. The `hybrid` study of the scaling
script measures where the balance falls.

For an MPI-free A/B benchmark of the two OpenMP loop layouts, build the
scalar solver with an explicit policy:

```sh
make OMP=1 SIMD=0 MPI=0 OMP_SPLIT=planes
make OMP=1 SIMD=0 MPI=0 OMP_SPLIT=lines
make OMP=1 SIMD=0 MPI=0 OMP_SPLIT=serial  # directional-solver control
```

Each policy is a separate build variant. `./solver` is the last one built,
and `make OMP=1 SIMD=0 MPI=0 OMP_SPLIT=planes print-build-id` names the
directory under `build/variants/` that keeps each binary.

Forced policies require `OMP=1` and reject MPI, SIMD and `TRIDIAG=pipeline`
builds. Planes are not a valid forced choice across MPI collectives. SIMD
would bypass the scalar line solver along two directions and make the
comparison incomplete. The pipeline backend has its own loop structure and
never reads the policy.

## Tridiagonal backends

`TRIDIAG` picks *how* a grid line that has been split across processes is
solved. `MPI`, `OMP` and `SIMD` decide something else: whether the line is
split at all, whether threads help inside a block, and how wide the kernels
are. The four options compose.

```sh
make TRIDIAG=schur MPI=1 OMP=1 SIMD=1     # the default
```

| value | method | cost |
|---|---|---|
| `schur` | Schur complement | three local Thomas solves per line; one exchange and one collective per group of lines |
| `pipeline` | pipelined Thomas | one local solve per line; the wait is hidden by sending many independent lines through the processes in batches |

Schur pays in arithmetic, and the pipeline pays in latency. The pipeline wins
when there are many more lines than processes, which is the normal case.

The choice is a directory, `src/tridiag/$(TRIDIAG)/`, not a chain of
`#ifdef`. Exactly one backend is ever compiled, so the two cannot silently
drift into each other. The physics they share lives in
`include/momentum_row.h` and `src/pressure_common.c`: one copy of the
formulas, two memory layouts.

Both backends compute the same answer. On one process they are bit-identical
to each other and to the serial solver, with or without threads and SIMD.
Once lines are split across processes, some sums are taken in a different
order, so the results agree to round-off. That is the property to check
first after touching either backend.

For a quick local backend check, run:

```sh
./scripts/check_pipeline.sh
```

The check reconstructs all eleven state fields and compares every owned cell
with a serial scalar Schur reference. It also runs the manufactured cases and
checks the momentum and pressure equation residuals, including MPI junctions.
The scenarios cover constant and space/time-dependent permeability, cubic and
rectangular grids, SIMD tails and partial batches. Thin MPI blocks are
checked against the serial reference even when Schur cannot run that
decomposition. Logs and `results.json` go to `build/pipeline-check/run-*`.

```sh
MPI=1 OMP=1 SIMD=1 THREADS="1 4" RANKS=4 ./scripts/check_pipeline.sh
PRECISION=float SIMD=1 ./scripts/check_pipeline.sh
GRIDS="16;17 19 13" PIPELINE_BATCHES="auto 1 3 64" ./scripts/check_pipeline.sh
python3 scripts/test_pipeline_tools.py
```

`PROCESS_GRID="px py pz"` selects one MPI layout. Otherwise the check tries
the three pure-axis splits, plus some mixed layouts for 4 and 8 processes.
`TOLERANCE` controls the combined absolute/relative comparison, with a
default of `1e-10` in double and `1e-4` in float. Non-finite values and
missing cells always fail. `TIMEOUT` bounds each run, and `BUILD_JOBS` sets
the number of concurrent compilations (default 2).

### The pipeline backend

The pipeline sends independent lines through the MPI processes in batches.
Smaller batches start the next process sooner, and larger ones amortize the
communication. By default, the batch is the power of two nearest to
256 x threads per process, capped at 4096 and the same on every process.
`pipeline_batch_lines = N` in the configuration file overrides it for one
run, and compiling with `PIPELINE_BATCH_LINES=N` takes precedence over both.
The rule was calibrated on an earlier loop structure and should be retuned
on the target cluster. The effective batch is capped by the number of
available lines. On axes that no process splits, it is also capped so that
every thread gets work.

Each OpenMP worker owns complete lines (or SIMD groups of neighbouring Y/Z
lines), so there is no barrier between Thomas levels. A thread team persists
for a whole direction. MPI calls run only on the master thread, which keeps
the code within `MPI_THREAD_FUNNELED` and preserves the forward/reverse
message order. Along X, each worker prepares the source term `g_line` in a
private reusable buffer and streams contiguous scratch. Along Y and Z, the
backend keeps the adjacent-line SIMD layout, with scalar fallbacks at row
ends and momentum walls. The pressure sweeps also use SIMD and reuse
coefficients factored once at initialization, so their forward messages
carry only the transformed right-hand side.

On axes contained in one process, a worker completes both sweeps of a batch
before reusing its scratch. Along a split axis, the momentum solve keeps all
three components until the reverse sweep. That takes two arrays (`c'` and
`d'`) of three local volumes each, plus batch padding: 96 MiB for a 128^3
double block without padding. The pressure solve shares that storage.

The implementation review and the local validation results are in
[the pipeline review](docs/pipeline/review-unified.md) and
[the validation report](docs/pipeline/validation.md).

## Scaling study

Measure how much the parallel run gains:

```sh
./scripts/run_scaling.sh          # strong, weak and hybrid, results in build/scaling/
SIMD=0 RESULTS_SUFFIX=_scalar ./scripts/run_scaling.sh
HYBRID=0 ./scripts/run_scaling.sh # skip the hybrid study
./scripts/plot_scaling.py         # figure in docs/scaling/
```

The second run disables the vectorized kernels. With the Schur backend they
only apply to directions that are not split, so a comparison with them
enabled measures two things at once. `plot_scaling.py` needs both runs
(`results.csv` and `results_scalar.csv`). `TRIDIAG=pipeline
RESULTS_SUFFIX=_pipeline` measures the other backend: the script builds
through the Makefile, so every `make` variable applies.

![Scaling](docs/scaling/scaling.svg)

### The exhaustive campaign

`scripts/study/` holds six phases that fill the whole matrix instead of
answering one question at a time. The matrix covers every backend, with and
without SIMD, for every process count and every shape of the process grid,
crossed with every thread count that fits in the node, over several problem
sizes.

| phase | what it sweeps |
|---|---|
| `10_matrix_threads` | one process, threads 1 to 112, both backends, SIMD on and off |
| `11_matrix_mpi` | one thread, every process count, all 80 shapes of the process grid; then the same question with the *local block* held cubic, and with the block stretched at a fixed shape |
| `12_matrix_hybrid` | the full process x thread rectangle, not just the diagonal |
| `13_matrix_batch` | `PIPELINE_BATCH_LINES` from 8 to 8192, over ten process x thread placements |
| `14_matrix_size` | problem size, strong scaling, weak scaling, peak memory |
| `15_matrix_check` | the error norms: does the whole matrix still solve the same problem? |

With the defaults on a 112-CPU node, the campaign has 2307 cases. Phases 10
to 14 run each case twice and keep the better time, and phase 15 runs each
case once, which makes 4317 solver runs.

```sh
./scripts/run_study.sh submit          # six chains, PBS runs them in parallel
./scripts/run_study.sh status          # how far along they are
./scripts/run_study.sh merge           # one CSV, then the figures in docs/scaling/matrix/
./scripts/run_study.sh dry             # list the cases, run nothing
./scripts/run_study.sh local 10        # run one phase here, without PBS
```

Each phase is its own PBS job and asks for a whole node, so the six run on
six nodes at once. Each phase resumes where it stopped and resubmits itself
until it is done, so the campaign can span days without anyone watching it.
The axes are environment variables, so the campaign scales to the time
available:

```sh
GRIDS="128 224 256" REPEATS=3 ./scripts/run_study.sh submit
MATRIX_BACKENDS=pipeline BATCHES="64 1024" ./scripts/run_study.sh submit 13
DRY_RUN=1 ./scripts/study/11_matrix_mpi.sh   # list the cases, run nothing
```

Study outputs and resume keys live in `build/study/<source-toolchain-id>/`.
Changing the source contents, the compiler or the build options starts a
separate campaign, and previous campaigns remain available. `STUDY_BASE`
selects an explicit campaign for `status` or `merge`. Phase 15 requires all
expected cases, finite norms and a serial scalar Schur reference, and it
exits with an error on a mismatch. When a budgeted run is still incomplete,
the verdict waits for its continuation.

Every phase writes the same CSV schema. `merge` concatenates the six into
the campaign's `all.csv` and runs `plot_matrix.py` on it. To redraw the
figures from a saved CSV, pass its path:
`./scripts/plot_matrix.py build/study/<id>/all.csv`.

Nine of the CSV columns time the stages of a time step separately: the three
momentum sweeps, the three pressure solves, the pressure update, the
permeability fill, and the time no stage accounts for. Another column holds
the time spent inside MPI. `matrix-12-composizione.svg` is the figure that
plots these columns.

There is no multi-node sweep. On this cluster nothing launches processes
across nodes (not `plm tm`, not ssh between compute nodes, not `pbs_tmrsh`),
so the ceiling is one node.

The measurements run `test/bench.c` (`bench <config> [px py pz]`). It is the
only binary that both reads the grid from a configuration file and takes the
shape of the process grid on the command line. That makes the split axis a
variable of the study, not whatever `MPI_Dims_create` chose.
`BENCH_NORMS=1` adds the error norms, and `BENCH_SCENARIO=<name>` selects
another scenario.

### Measuring one change

A campaign shows how the solver behaves. It does not show what a given
commit was worth. For that there is a separate A/B:

```sh
./scripts/run_patch_ab.sh                 # HEAD~1 against HEAD
./scripts/run_patch_ab.sh e64ef95 HEAD    # any two revisions
qsub scripts/run_patch_ab.sh              # on the cluster
./scripts/plot_patch_ab.py                # two figures in docs/scaling/patch-ab/
```

The script extracts both revisions with `git archive` into
`build/patch-ab/`, builds each with the same flags, and runs them
**alternately** in the same job. It keeps every repeat, not only the best
one. The CSV carries the same per-stage columns, and that is the point: a
change that touches one stage must show up in that stage and nowhere else.
The untouched stages act as a control, and their spread is the noise floor
of that node. The figures put the two revisions side by side, with the
control stages in grey.

## Solver structure

`solver_init` allocates the numerical fields and initializes them through
the functions stored in `Data`. Then `solver_solve` advances the solution
for `STEPS` time steps:

```text
solver_init                     allocate the fields, fill them at t = 0
    |
    v
solver_solve
    +-- backend_init            backend scratch; pressure matrices factored once
    |
    +-- time-step loop
    |      +-- halo exchange    eta, zeta, u, pressure_star
    |      +-- porosity fill    only when K depends on time
    |      +-- momentum_step
    |      |      +-- eta: solve along X
    |      |      +-- zeta: solve along Y
    |      |      +-- u: solve along Z
    |      +-- halo exchange    u, read by the divergence
    |      +-- pressure_step
    |      |      +-- psi: solve along X, right-hand side -div(u)/dt
    |      |      +-- phi_low: solve along Y
    |      |      +-- phi_high: solve along Z
    |      |      +-- pressure update
    |      +-- write output     every wr_freq steps, when writing is enabled
    |
    +-- backend_free
```

The momentum systems are solved one direction at a time with the Thomas
algorithm for tridiagonal matrices, and the pressure correction is factored
into three directional solves in the same way. `src/solver.c` drives the
loop and is the same for both backends. `momentum_step` and `pressure_step`
come from the selected backend, in `src/tridiag/<backend>/momentum.c` and
`pressure.c`. The shared code covers the rest:

- `include/momentum_row.h` builds the momentum row of one cell.
- `src/pressure_common.c` holds the pressure right-hand side, line matrix
  and update.
- `src/physics.c`, `src/field.c` and `src/data.c` provide the physical
  terms, the field utilities and the scenarios.
- `src/decomp.c` and `src/parallel.c` handle the block decomposition and
  MPI.
- `src/output.c` writes the VTK files, and `src/params.c` reads the
  configuration file.

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
+-- forcing_line_fn()         optional: the forcing along a whole X line, or NULL
+-- porosity_fn()             permeability field K
+-- porosity_time_dependent   boolean
+-- velocity_fn()             initial/exact velocity
+-- pressure_fn()             initial/exact pressure

SolverMemState
+-- eta, zeta, u, k           VectorField
+-- pressure, pressure_star   ScalarField
+-- backend                   backend scratch, opaque to the shared code

SolverStats
+-- execution times for the solver stages, stored in nanoseconds
```

Function pointers in `Data` keep the numerical solver independent of any
specific physical test case. `SolverMemState` groups all fields that must
remain available between time steps.

## Memory management

Every field array holds the process's block plus a ring of halo cells
(`Decomp::n_cells` elements).

```text
solver_init
    +-- allocate persistent fields
        +-- 4 VectorField = 12 block-sized arrays
        +-- 2 ScalarField =  2 block-sized arrays

solver_solve
    +-- allocate pressure_buffer   1 block-sized temporary array
    +-- backend_init               allocate the backend scratch
    |       schur:    rhs and tmp, one line or SIMD-group buffer per thread,
    |                 plus the three factored pressure matrices
    |       pipeline: c' and d' for the lines in flight, plus the
    |                 factored pressure coefficients of each axis
    +-- run all time steps
    +-- backend_free, free pressure_buffer
```
