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
    ArenaAllocator *arena;

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
    output->arena = arena;
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

static inline void to_big_endian(const ftype *src, ftype *dst)
{
#ifdef FLOAT
    uint32_t word = *((uint32_t *) src);
    *((uint32_t *) dst) = ((word & 0xff000000) >> 24) |
                          ((word & 0x00ff0000) >> 8) |
                          ((word & 0x0000ff00) << 8) |
                          ((word & 0x000000ff) << 24);
#else
    uint64_t word = *((uint64_t *) src);
    *((uint64_t *) dst) = ((word & 0xff00000000000000) >> 56) |
                          ((word & 0x00ff000000000000) >> 40) |
                          ((word & 0x0000ff0000000000) >> 24) |
                          ((word & 0x000000ff00000000) >> 8) |
                          ((word & 0x00000000ff000000) << 8) |
                          ((word & 0x0000000000ff0000) << 24) |
                          ((word & 0x000000000000ff00) << 40) |
                          ((word & 0x00000000000000ff) << 56);
#endif
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

    arena_enter(output->arena);

    uint64_t num_points = field_num_points(output->grid_size);

    ftype *tmp = arena_push_count(output->arena, ftype, 3 * num_points);

    struct OutputNode *curr = output->nodes_list_head;
    while (curr != NULL) {

        if (curr->type == OUTPUT_NODE_SCALAR) {
            fprintf(output_file,
                    "\nSCALARS %s %s 1\nLOOKUP_TABLE default\n",
                    curr->name, ftype_str);

            for (uint64_t i = 0; i < num_points; ++i) {
                to_big_endian(curr->data[0] + i, tmp + i);
            }

            fwrite(tmp, sizeof(ftype), num_points, output_file);

        } else if (curr->type == OUTPUT_NODE_VECTOR) {
            fprintf(output_file, "\nVECTORS %s %s\n", curr->name, ftype_str);

            for (uint64_t i = 0; i < num_points; ++i) {
                /* Compiler please fuse and vectorize these. */
                to_big_endian(curr->data[0] + i, tmp + (3 * i + 0));
                to_big_endian(curr->data[1] + i, tmp + (3 * i + 1));
                to_big_endian(curr->data[2] + i, tmp + (3 * i + 2));
            }

            fwrite(tmp, 3 * sizeof(ftype), num_points, output_file);
        }

        curr = curr->next;
    }

    fclose(output_file);

    arena_exit(output->arena);
}
