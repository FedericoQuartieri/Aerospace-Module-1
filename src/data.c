#include <math.h>
#include "functions.h"

static DTYPE paper_bc_velocity(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component)
{
    switch (component) {
        case 0: return sin(x) * cos(t + y) * sin(z);
        case 1: return cos(x) * sin(t + y) * sin(z);
        case 2: return 2.0 * cos(x) * cos(t + y) * cos(z);
        default: return 0.0;
    }
}

static DTYPE paper_forcing(DTYPE x, DTYPE y, DTYPE z, DTYPE t, int component)
{  
    /* x ∈ [0,π] */
    switch (component) {
        case 0: 
            return - (sin(x) * sin(t + y) * sin(z)) + (3.0 * sin(x) * cos(t + y) * sin(z)) + (sin(x) * cos(t + y) * sin(z)) + (3.0 * sin(x) * cos(t + y) * cos(z));
        case 1:
            return (cos(x) * cos(t + y) * sin(z)) + (3.0 * cos(x) * sin(t + y) * sin(z)) + (cos(x) * sin(t + y) * sin(z)) + (3.0 * cos(x) * sin(t + y) * cos(z));
        case 2:
            return - (2.0 * cos(x) * sin(t + y) * cos(z)) + (6.0 * cos(x) * cos(t + y) * cos(z)) + (2.0 * cos(x) * cos(t + y) * cos(z)) + (3.0 * cos(x) * cos(t + y) * sin(z));    
        default: 
            return 0.0;
    }
}

const Data PAPER_DATA = {
    .name = "paper_data",
    .bc_velocity = paper_bc_velocity,
    .forcing = paper_forcing
};