#ifndef CONSTANTS_H
#define CONSTANTS_H

// Grid dimensions
#define WIDTH 3
#define HEIGHT 3
#define DEPTH 7

// Grid spatial dimension
#define GRID_ELEMENTS WIDTH * HEIGHT * DEPTH

// Ghost cells (boundaries)
#define GHOST_CELLS 1

// Total field size in byte
#define GRID_SIZE ( (WIDTH + GHOST_CELLS) * (HEIGHT + GHOST_CELLS) * (DEPTH + GHOST_CELLS) * sizeof(DTYPE) )

// Data type
#define DTYPE double

// Physical constants for Navier-Stokes-Brinkman
#define NU 0.7                // Kinematic viscosity (nu)
#define PERMEABILITY 1.0      // Brinkman permeability (K)
#define POROSITY 0.5          // Porosity (phi)
#define DENSITY 1.0           // Fluid density (rho)

// Numerical parameters
#define DX 0.1                // Grid spacing in x
#define DX_INVERSE (1.0 / DX) // Precomputed inverse of DX
#define DX_INVERSE_SQUARED (DX_INVERSE * DX_INVERSE) // Precomputed inverse squared of DX
#define DY 0.1                // Grid spacing in y
#define DY_INVERSE (1.0 / DY) // Precomputed inverse of DY
#define DY_INVERSE_SQUARED (DY_INVERSE * DY_INVERSE) // Precomputed inverse squared of DY
#define DZ 0.1                // Grid spacing in z
#define DZ_INVERSE (1.0 / DZ) // Precomputed inverse of DZ
#define DZ_INVERSE_SQUARED (DZ_INVERSE * DZ_INVERSE) // Precomputed inverse squared of DZ
#define DT 0.001              // Time step

// Simulation parameters
#define MAX_ITERATIONS 1000
#define TOLERANCE 1e-6

#endif // CONSTANTS_H