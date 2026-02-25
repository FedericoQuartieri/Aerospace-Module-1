#include "output.h"

#include <string.h>
#include <stdio.h>
#include <stddef.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>

#include "alloc.h"
#include "field.h"
#include "thread-array.h"

#define MAX_VTK_HEADER_SIZE 256
#define MAX_VTK_ATTRIBUTE_SPEC_SIZE 64

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
    char *mmapped_file;
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
    struct OutputNode *node = arena_push_noalign(arena,
                                                 sizeof(struct OutputNode));
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
    struct OutputNode *node = arena_push_noalign(arena,
                                                 sizeof(struct OutputNode));
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

struct UnmapFileArgs
{
    
};

static void *munmap_output_file(void *args)
{


}

void output_vtk_write(OutputVTK *output,
                      const char *output_file_name,
                      Thread *thread)
{
    uint64_t num_points = field_num_points(output->grid_size);

    uint64_t file_size = MAX_VTK_HEADER_SIZE;
    /* Estimate output file size. */
    struct OutputNode *curr = output->nodes_list_head;
    while (curr != NULL) {
        file_size += MAX_VTK_ATTRIBUTE_SPEC_SIZE;

        uint64_t field_size = sizeof(ftype) * num_points;
        if (curr->type == OUTPUT_NODE_VECTOR) {
            field_size *= 3;
        }
        file_size += field_size;

        curr = curr->next;
    }

    int fd = 0;
    if (thread->t_id == 0) {
        /* TODO: Handle syscall errors. */
        fd = creat(output_file_name, S_IRUSR | S_IWUSR |
                                     S_IRGRP | S_IWGRP | S_IROTH);

        /* Apparently access modes apply only for future accesses
         * of the file, so we need to reopen the file. */
        fd = open(output_file_name, O_RDWR); /* rw mode to allow mmap. */
        ftruncate(fd, file_size);

        output->mmapped_file = mmap(NULL, file_size,
                                    PROT_WRITE, MAP_SHARED, fd, 0);
    }

    thread_wait_on_barrier(thread);

    char buff[MAX_VTK_HEADER_SIZE];
    sprintf(buff,
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

    uint64_t header_size = strlen(buff);

    if (thread->t_id == 0) {
        memcpy(output->mmapped_file, buff, header_size);
    }

    uint32_t num_threads = thread_get_array_size(thread);
    uint64_t num_points_per_thread = num_points / num_threads;
    uint64_t p_start = num_points_per_thread * thread->t_id;
    uint64_t p_end = p_start + num_points_per_thread;

    if (thread->t_id == num_threads - 1) {
        p_end = num_points;
    }

    uint64_t offset = header_size;
    curr = output->nodes_list_head;
    while (curr != NULL) {
#ifdef FLOAT
        static const char *ftype_str = "float";
#else
        static const char *ftype_str = "double";
#endif
        if (curr->type == OUTPUT_NODE_SCALAR) {
            sprintf(buff, "\nSCALARS %s %s 1\nLOOKUP_TABLE default\n",
                    curr->name, ftype_str);
        } else {
            sprintf(buff, "\nVECTORS %s %s\n", curr->name, ftype_str);
        }

        uint64_t attribute_spec_size = strlen(buff);
        if (thread->t_id == 0) {
            memcpy(output->mmapped_file + offset,
                   buff, attribute_spec_size);
        }
        offset += attribute_spec_size;

        if (curr->type == OUTPUT_NODE_SCALAR) {
            for (uint64_t i = 0; i < num_points; ++i) {
                to_big_endian(curr->data[0] + i,
                              output->mmapped_file + offset
                                                   + i * sizeof(ftype));
            }
            offset += num_points * sizeof(ftype);
        } else {
            char *dst = output->mmapped_file + offset;
            for (uint64_t i = p_start; i < p_end; ++i) {
                /* Compiler please fuse and vectorize these. */
                to_big_endian(curr->data[0] + i,
                              dst + sizeof(ftype) * (3 * i + 0));
                to_big_endian(curr->data[1] + i,
                              dst + sizeof(ftype) * (3 * i + 1));
                to_big_endian(curr->data[2] + i,
                              dst + sizeof(ftype) * (3 * i + 2));
            }
            offset += num_points * 3 * sizeof(ftype);
        }

        curr = curr->next;
    }

    thread_wait_on_barrier(thread);

    /* TODO: Spawn a thread to munmap to hide the flush overhead.
     * Moreover, two threads are enough to saturate output performance,
     * so limit the number of threads involved. */

    if (thread->t_id == 0) {
        ftruncate(fd, offset);
        close(fd);

        /*
        struct {
            void *addr;
            uint64_t size;
        } *munmap_args = ..

        munmap_args->addr = output->mmapped_file;
        munmap_args->size = file_size;

        thread_run_orpan(thread, munmap_file, munmap_args);
        */

        munmap(output->mmapped_file, file_size);
    }
}
