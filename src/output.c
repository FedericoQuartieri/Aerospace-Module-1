#include "output.h"

#include <string.h>
#include <stdio.h>
#include <stddef.h>

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

static inline void to_big_endian(const ftype *src, char *dst)
{
    for (int b = 0; b < sizeof(ftype); ++b) {
        dst[b] = *((char *) (src + 1) - (b + 1));
    }
}

void output_vtk_write(const OutputVTK *output, const char *output_file_name)
{
    FILE *output_file = fopen(output_file_name, "wb");

    fprintf(output_file,
            "# vtk DataFile Version 3.0\n" \
            "flow\n"                       \
            "BINARY\n"                     \
            "DATASET STRUCTURED_POINTS\n"  \
            "DIMENSIONS %u %u %u\n"        \
            "ORIGIN 0 0 0\n"               \
            "SPACING %f %f %f\n"           \
            "POINT_DATA %lu\n",
            output->grid_size.width,
            output->grid_size.height,
            output->grid_size.depth,
            output->grid_spacing,
            output->grid_spacing,
            output->grid_spacing,
            field_num_points(output->grid_size));

#ifdef FLOAT
    static const char *ftype_str = "float";
#else
    static const char *ftype_str = "double";
#endif

    struct OutputNode *curr = output->nodes_list_head;
    while (curr != NULL) {

        if (curr->type == OUTPUT_NODE_SCALAR) {
            fprintf(output_file,
                    "\nSCALARS %s %s 1\nLOOKUP_TABLE default\n",
                    curr->name, ftype_str);

            for (uint64_t i = 0;
                          i < field_num_points(output->grid_size); ++i) {

                char bytes[sizeof(ftype)];
                to_big_endian(curr->data[0] + i, bytes);
                fwrite(bytes, sizeof(ftype), 1, output_file);
            }

        } else if (curr->type == OUTPUT_NODE_VECTOR) {
            fprintf(output_file, "\nVECTORS %s %s\n", curr->name, ftype_str);

            for (uint64_t i = 0;
                          i < field_num_points(output->grid_size); ++i) {

                char bytes[sizeof(ftype) * 3];
                /* Compiler please fuse these. */
                to_big_endian(curr->data[0] + i, bytes);
                to_big_endian(curr->data[1] + i, bytes + sizeof(ftype) * 1);
                to_big_endian(curr->data[2] + i, bytes + sizeof(ftype) * 2);

                fwrite(bytes, sizeof(ftype), 3, output_file);
            }
        }

        curr = curr->next;
    }

    fclose(output_file);
}
