#ifndef OUTPUT_H
#define OUTPUT_H

#include "alloc.h"
#include "field.h"

struct OutputVTK;

typedef struct OutputVTK OutputVTK;

OutputVTK *output_vtk_create(field_size grid_size,
                             ftype grid_spacing,
                             ArenaAllocator *arena);

void output_vtk_attach_field(OutputVTK *output,
                             const_field field,
                             const char *name,
                             ArenaAllocator *arena);

void output_vtk_attach_field3(OutputVTK *output,
                             const_field3 field,
                             const char *name,
                             ArenaAllocator *arena);

void output_vtk_write(const OutputVTK *output, const char *output_file_name);

#endif
