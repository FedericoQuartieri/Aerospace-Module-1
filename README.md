# Navier–Stokes–Brinkman Solver

## Build and run

Build the default executable with MPI and run it with any process
count :

```sh
make solver
mpirun -np 4 ./solver
```

The number of independent tridiagonal lines sent in each pipeline message is tunable:

```sh
make PIPELINE_BATCH_LINES=128
```

The default is 64 lines. Enable the explicit SIMD kernels used by the batched
Y and Z momentum sweeps with:

```sh
make SIMD=1
```

`Real` is `double` by default and becomes `float` with `-DUSE_FLOAT`.

## Tests

Compile every test with:

```sh
make tests
```

The manufactured-solution test can be run with :

```sh
mpirun -np 4 ./build/tests/paper_man
mpirun -np 4 ./build/tests/constant_forcing_man
```

## Convergence study

Run the spatial and temporal studies with:

```sh
./scripts/run_convergence.sh
```
Results are written to
`build/convergence/results.csv`.

![Velocity convergence](docs/convergence/velocity.svg)

![Pressure convergence](docs/convergence/pressure.svg)

## Solver structure

The solver state has the following lifetime:

```text
MPI_Init
    |
solver_init
    +-- create the Cartesian domain and halo datatypes
    +-- allocate and initialize persistent numerical fields
    +-- allocate the reusable Thomas pipeline workspace
    |
solver_solve
    +-- allocate one temporary pressure field
    +-- advance all time steps
    +-- gather timing statistics and free the temporary field
    |
solver_destroy
    +-- free fields and pipeline storage
    +-- destroy halo datatypes and the Cartesian communicator
    |
MPI_Finalize
```

Each time step contains three directional momentum stages and three
directional pressure stages:

```text
halo exchange for eta/X, zeta/Y, u/Z and pressure_star/X,Y,Z
    |
    +-- eta:  batched Thomas pipeline along X
    +-- zeta: batched Thomas pipeline along Y
    +-- u:    batched Thomas pipeline along Z
    |
halo exchange of u_x, u_y and u_z for divergence
    |
    +-- psi:      batched Thomas pipeline along X
    +-- phi_low:  batched Thomas pipeline along Y
    +-- phi_high: batched Thomas pipeline along Z
    +-- pressure update
```

For every momentum direction the complete forward phase is performed in
component order `v_x`, `v_y`, `v_z`. The backward phase then processes
`v_z`, `v_y`, `v_x`. Within a component, each batch is sent immediately to
the next Cartesian neighbour, allowing adjacent blocks to work concurrently.

The forward interface contains the last reduced `(c', d')` pair for every
line in the batch. The backward interface contains the first solution value
owned by the block on the right. Only ranks on physical domain boundaries
apply the physical boundary conditions.
Each rank must retain its local reduced coefficients until the backward phase arrives.

