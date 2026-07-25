# Navier Stokes Brinkman Equation Solver


## Build

Compile the main `solver` executable:

```sh
make solver
```

Compile all tests:

```sh
make tests
```

The test executables are created in `build/tests/` and can be run separately:

```sh
./build/tests/paper_man
./build/tests/zero_pressure
```

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
ScalarField                         VectorField (structure of arrays)
+------------------+                +------------------+
| Real *v          |                | Real *v_x        | --> [x0][x1]...[xN]
+------------------+                | Real *v_y        | --> [y0][y1]...[yN]
                                    | Real *v_z        | --> [z0][z1]...[zN]
                                    +------------------+
```

The main structures are:

```text
Data
+-- name
+-- bc_velocity()       boundary velocity
+-- forcing_fn()        forcing term
+-- porosity_fn()       porosity field
+-- velocity_fn()       initial/exact velocity
+-- pressure_fn()       initial/exact pressure

SolverMemState
+-- eta, zeta, u, k     VectorField
+-- pressure,
    pressure_star       ScalarField

SolverStats
+-- execution times for the solver stages, stored in nanoseconds
```

Function pointers in `Data` keep the numerical solver independent from a
specific physical test case. `SolverMemState` groups all fields that must
remain available between time steps.

## Memory management

```text
solver_init
    +-- allocate persistent fields
        +-- 4 VectorField = 12 full-grid arrays
        +-- 2 ScalarField =  2 full-grid arrays

solver_solve
    +-- allocate pressure_buffer   1 full-grid temporary array
    +-- allocate rhs and tmp       2 reusable line buffers
    +-- run all time steps
    +-- free pressure_buffer, rhs, and tmp
```

