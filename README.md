# Stokes-Brinkman finite-difference fluid solver

## Gallery

| ![](images/obstacles-velocity.png) |  ![](images/obstacles-pressure.png) |
|:--------:|:-------:|
| Velocity field magnitude | Pressure colored velocity field |

## TODO

- [x] complete fused Dxx rhs computation
- [ ] add support for floats in fused Dxx rhs computation
- [ ] fuse pressure update into Dzz solve
- [ ] complete fused pressure solve
- [x] test with obstacles using low permeability
- [ ] convergence test with variable permeability
- [ ] parallelize kernels across cores (pthreads/openmp)
- [ ] implement schur complement
- [ ] parallelize schur complement across nodes (MPI)
- [ ] outflow boundary conditions
- [ ] benchmark AUTO_VEC performance
