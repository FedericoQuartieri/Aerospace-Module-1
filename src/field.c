#include "solver_internal.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { FIELD_ALIGNMENT = 64 };

/* count is expressed in Real values.  OOM is fatal by design, so the numerical
 * path never has to carry partially recovered allocation state. */
static void *xmalloc(size_t count)
{
    void *memory = NULL;
    size_t bytes;
    int allocation_error;

    if (count > SIZE_MAX / sizeof(Real)) {
        fprintf(stderr, "ERROR: field allocation size overflow\n");
        exit(EXIT_FAILURE);
    }
    bytes = count * sizeof(Real);
    if (bytes == 0) {
        return NULL;
    }
    allocation_error = posix_memalign(&memory, FIELD_ALIGNMENT, bytes);
    if (allocation_error != 0) {
        fprintf(stderr, "ERROR: unable to allocate %zu aligned bytes: %s\n",
                bytes, strerror(allocation_error));
        exit(EXIT_FAILURE);
    }
    return memory;
}

bool scalar_field_init(ScalarField *field, size_t count)
{
    if (field == NULL) {
        return false;
    }
    field->data = NULL;
    field->count = 0;
    if (count == 0) {
        return true;
    }
    field->data = xmalloc(count);
    field->count = count;
    memset(field->data, 0, count * sizeof(Real));
    return true;
}

void scalar_field_destroy(ScalarField *field)
{
    if (field == NULL) {
        return;
    }
    free(field->data);
    field->data = NULL;
    field->count = 0;
}

void scalar_field_fill(ScalarField *field, Real value)
{
    size_t q;
    for (q = 0; q < field->count; ++q) {
        field->data[q] = value;
    }
}

void scalar_field_copy(ScalarField *destination, const ScalarField *source)
{
    if (destination == NULL || source == NULL ||
        destination->count != source->count) {
        return;
    }
    memcpy(destination->data, source->data, source->count * sizeof(Real));
}

bool vector_field_init(VectorField *field, size_t count)
{
    Direction component;

    if (field == NULL) {
        return false;
    }
    memset(field, 0, sizeof(*field));
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        if (!scalar_field_init(&field->component[component], count)) {
            vector_field_destroy(field);
            return false;
        }
    }
    return true;
}

void vector_field_destroy(VectorField *field)
{
    Direction component;

    if (field == NULL) {
        return;
    }
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        scalar_field_destroy(&field->component[component]);
    }
}

void vector_field_fill(VectorField *field, Real value)
{
    Direction component;
    for (component = DIRECTION_X;
         component < DIRECTION_COUNT;
         component = (Direction)(component + 1)) {
        scalar_field_fill(&field->component[component], value);
    }
}

bool real_buffer_init(RealBuffer *buffer, size_t capacity)
{
    if (buffer == NULL) {
        return false;
    }
    buffer->data = NULL;
    buffer->capacity = 0;
    if (capacity == 0) {
        return true;
    }
    buffer->data = xmalloc(capacity);
    buffer->capacity = capacity;
    memset(buffer->data, 0, capacity * sizeof(Real));
    return true;
}

void real_buffer_destroy(RealBuffer *buffer)
{
    if (buffer == NULL) {
        return;
    }
    free(buffer->data);
    buffer->data = NULL;
    buffer->capacity = 0;
}
