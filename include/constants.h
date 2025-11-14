#ifndef CONSTANTS_H
#define CONSTANTS_H

// Grid dimensions
#define WIDTH 10
#define HEIGHT 6
#define DEPTH 17

// Grid spatial dimension (number of elements)
#define GRID_ELEMENTS WIDTH * HEIGHT * DEPTH

// Ghost cells (boundaries)
#define GHOST_CELLS 1

// Total number of grid points including ghost cells
#define TOTAL_GRID_POINTS ( (WIDTH + GHOST_CELLS) * (HEIGHT + GHOST_CELLS) * (DEPTH + GHOST_CELLS) )

// Total field size in bytes
#define GRID_SIZE (TOTAL_GRID_POINTS * sizeof(DTYPE))

// Data type
#define DTYPE double

// Physical constants for Navier-Stokes-Brinkman
#define NU 0.7                // Kinematic viscosity (nu)
#define PERMEABILITY 1.0      // Brinkman permeability (K)
#define POROSITY 0.5          // Porosity (phi)
#define DENSITY 1.0           // Fluid density (rho)

// Numerical parameters
#define DX 0.001                // Grid spacing in x
#define DY 0.001                // Grid spacing in y
#define DZ 0.001                // Grid spacing in z
#define DT 0.001              // Time step
#define TOTAL_TIME 0.1      // Total simulation time
#define STEPS (int)(TOTAL_TIME / DT)  // Number of time steps
#define WRITE_FREQUENCY 10  // Frequency of writing output

#define DX_INVERSE (1.0) / DX
#define DY_INVERSE (1.0) / DY
#define DZ_INVERSE (1.0) / DZ

#define DX_INVERSE_SQUARE (DX_INVERSE * DX_INVERSE)
#define DY_INVERSE_SQUARE (DY_INVERSE * DY_INVERSE)
#define DZ_INVERSE_SQUARE (DZ_INVERSE * DZ_INVERSE)

// Simulation parameters
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-6

#endif // CONSTANTS_H