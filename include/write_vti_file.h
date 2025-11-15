#ifndef WRITE_VTI_FILE_H
#define WRITE_VTI_FILE_H
#include <stdio.h>
#include <stdint.h>
#include "constants.h"
#include "solve.h"   

void write_vti_file(const char *filename,
                    const VelocityField *U,
                    const Pressure *P,
                    int Nx, int Ny, int Nz,
                    double dx, double dy, double dz);

#endif // WRITE_VTI_FILE_H
