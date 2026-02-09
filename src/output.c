#include "output.h"

#include <string.h>
#include <stdio.h>

#include "alloc.h"
#include "field.h"

enum OutputNodeType { OUTPUT_NODE_SCALAR, OUTPUT_NODE_VECTOR };

struct OutputNode
{
    struct OutputNode *next;

    enum OutputNodeType type;
    const char *name;

    const_field data[3];
};

struct OutputVTK
{
    field_size grid_size;
    ftype grid_spacing;

    struct OutputNode *nodes_list_head;
};

OutputVTK *output_vtk_create(field_size grid_size,
                             ftype grid_spacing,
                             ArenaAllocator *arena)
{
    OutputVTK *output = arena_push_noalign(arena, sizeof(OutputVTK));
    output->grid_size = grid_size;
    output->grid_spacing = grid_spacing;
    output->nodes_list_head = NULL;
    return output;
}

void output_vtk_attach_field(OutputVTK *output,
                             const_field field,
                             const char *name,
                             ArenaAllocator *arena)
{
    struct OutputNode *node = arena_push_noalign(arena, sizeof(struct OutputNode));
    node->data[0] = field;
    node->name = name;
    node->type = OUTPUT_NODE_SCALAR;

    node->next = output->nodes_list_head;
    output->nodes_list_head = node;
}

void output_vtk_attach_field3(OutputVTK *output,
                              const_field3 field,
                              const char *name,
                              ArenaAllocator *arena)
{
    struct OutputNode *node = arena_push_noalign(arena, sizeof(struct OutputNode));
    node->data[0] = field.x;
    node->data[1] = field.y;
    node->data[2] = field.z;
    node->name = name;
    node->type = OUTPUT_NODE_VECTOR;

    node->next = output->nodes_list_head;
    output->nodes_list_head = node;
}

void output_vtk_write(const OutputVTK *output, const char *output_file_name)
{
    FILE *output_file = fopen(output_file_name, "w");

    fprintf(output_file,
            "# vtk DataFile Version 3.0\n" \
            "stokes-brinkman\n"            \
            "ASCII\n"                      \
            "DATASET RECTILINEAR_GRID\n"   \
            "DIMENSIONS %u %u %u\n",
            output->grid_size.width,
            output->grid_size.height,
            output->grid_size.depth);

    const char *axes_label[] = { "X", "Y", "Z" };
    const uint32_t axes_size[] = { output->grid_size.width,
                                   output->grid_size.height,
                                   output->grid_size.depth };

    for (int axis = 0; axis < 3; ++axis) {
        fprintf(output_file,
                "%s_COORDINATES %u float\n",
                axes_label[axis],
                axes_size[axis]);

        for (uint32_t i = 0; i < axes_size[axis]; ++i) {
            fprintf(output_file, "%.4f ", i * output->grid_spacing);
        }
    }

    fprintf(output_file,
            "\nPOINT_DATA %lu\n",
            field_num_points(output->grid_size));

    struct OutputNode *curr = output->nodes_list_head;
    while (curr != NULL) {

        if (curr->type == OUTPUT_NODE_SCALAR) {
            fprintf(output_file,
                    "\nSCALARS %s double 1\nLOOKUP_TABLE default\n",
                    curr->name);

            for (uint64_t i = 0; i < field_num_points(output->grid_size); ++i) {
                fprintf(output_file, "%f ", curr->data[0][i]);
            }

        } else if (curr->type == OUTPUT_NODE_VECTOR) {
            fprintf(output_file, "\nVECTORS %s double\n", curr->name);

            for (uint64_t i = 0; i < field_num_points(output->grid_size); ++i) {
                fprintf(output_file, "%f %f %f ", curr->data[0][i],
                                                  curr->data[1][i],
                                                  curr->data[2][i]);
            }
        }

        curr = curr->next;
    }

    fclose(output_file);
}
