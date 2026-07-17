#include "solver_internal.h"

#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

enum { OUTPUT_TUPLES_PER_CHUNK = 16384 };

static const char *vtk_real_type(void)
{
    return sizeof(Real) == sizeof(float) ? "Float32" : "Float64";
}

static bool write_bytes(FILE *file, const void *data, size_t bytes)
{
    return bytes == 0 || fwrite(data, 1, bytes, file) == bytes;
}

bool output_writer_init(OutputWriter *writer, const SolverConfig *config)
{
    if (writer == NULL || config == NULL) {
        return false;
    }
    memset(writer, 0, sizeof(*writer));
    if (config->output_frequency == 0) {
        return true;
    }
    if (config->output_directory == NULL ||
        config->output_directory[0] == '\0') {
        return false;
    }
    if (mkdir(config->output_directory, 0755) != 0 && errno != EEXIST) {
        return false;
    }
    writer->enabled = true;
    writer->frequency = config->output_frequency;
    writer->directory = config->output_directory;
    return real_buffer_init(&writer->pack_buffer,
                            3 * OUTPUT_TUPLES_PER_CHUNK);
}

static bool write_velocity(FILE *file,
                           OutputWriter *writer,
                           const VectorField *velocity,
                           size_t count)
{
    const size_t tuple_capacity = writer->pack_buffer.capacity / 3;
    size_t base;

    /* VTK expects interleaved XYZ tuples; solver fields remain independent SoA
     * allocations, so conversion is streamed through the fixed-size buffer. */
    for (base = 0; base < count; base += tuple_capacity) {
        size_t tuple;
        size_t chunk = count - base;
        if (chunk > tuple_capacity) {
            chunk = tuple_capacity;
        }
        for (tuple = 0; tuple < chunk; ++tuple) {
            writer->pack_buffer.data[3 * tuple] =
                velocity->component[DIRECTION_X].data[base + tuple];
            writer->pack_buffer.data[3 * tuple + 1] =
                velocity->component[DIRECTION_Y].data[base + tuple];
            writer->pack_buffer.data[3 * tuple + 2] =
                velocity->component[DIRECTION_Z].data[base + tuple];
        }
        if (!write_bytes(file, writer->pack_buffer.data,
                         3 * chunk * sizeof(Real))) {
            return false;
        }
    }
    return true;
}

bool output_writer_write(OutputWriter *writer,
                         const Grid *grid,
                         size_t timestep,
                         Real velocity_time,
                         Real pressure_time,
                         const VectorField *velocity,
                         const ScalarField *pressure)
{
    char filename[1024];
    FILE *file;
    const size_t scalar_bytes = grid->cell_count * sizeof(Real);
    const size_t vector_bytes = 3 * scalar_bytes;
    const uint32_t scalar_block = (uint32_t)scalar_bytes;
    const uint32_t vector_block = (uint32_t)vector_bytes;
    const size_t velocity_offset = sizeof(uint32_t) + scalar_bytes;
    int name_length;
    bool success = true;

    if (!writer->enabled) {
        return true;
    }
    if (scalar_bytes > UINT32_MAX || vector_bytes > UINT32_MAX) {
        return false;
    }
    name_length = snprintf(filename, sizeof(filename), "%s/solution_%06zu.vti",
                           writer->directory, timestep);
    if (name_length < 0 || (size_t)name_length >= sizeof(filename)) {
        return false;
    }
    file = fopen(filename, "wb");
    if (file == NULL) {
        return false;
    }

    /* Raw appended arrays are each prefixed by a UInt32 byte count.  Pressure
     * is written directly; velocity follows after chunked SoA interleaving. */
    if (fprintf(file,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        "  <ImageData WholeExtent=\"0 %zu 0 %zu 0 %zu\" Origin=\"0 0 0\" "
        "Spacing=\"%.17g %.17g %.17g\">\n"
        "    <FieldData>\n"
        "      <DataArray type=\"Float64\" Name=\"TimeValue\" NumberOfTuples=\"1\" format=\"ascii\">%.17g</DataArray>\n"
        "      <DataArray type=\"Float64\" Name=\"PressureTime\" NumberOfTuples=\"1\" format=\"ascii\">%.17g</DataArray>\n"
        "    </FieldData>\n"
        "    <Piece Extent=\"0 %zu 0 %zu 0 %zu\">\n"
        "      <PointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n"
        "        <DataArray type=\"%s\" Name=\"Pressure\" format=\"appended\" offset=\"0\"/>\n"
        "        <DataArray type=\"%s\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n"
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "  <AppendedData encoding=\"raw\">\n   _",
        grid->extent[DIRECTION_X] - 1,
        grid->extent[DIRECTION_Y] - 1,
        grid->extent[DIRECTION_Z] - 1,
        (double)grid->spacing[DIRECTION_X],
        (double)grid->spacing[DIRECTION_Y],
        (double)grid->spacing[DIRECTION_Z],
        (double)velocity_time,
        (double)pressure_time,
        grid->extent[DIRECTION_X] - 1,
        grid->extent[DIRECTION_Y] - 1,
        grid->extent[DIRECTION_Z] - 1,
        vtk_real_type(), vtk_real_type(), velocity_offset) < 0) {
        success = false;
    }

    if (success &&
        (!write_bytes(file, &scalar_block, sizeof(scalar_block)) ||
         !write_bytes(file, pressure->data, scalar_bytes) ||
         !write_bytes(file, &vector_block, sizeof(vector_block)) ||
         !write_velocity(file, writer, velocity, grid->cell_count) ||
         fprintf(file, "\n  </AppendedData>\n</VTKFile>\n") < 0)) {
        success = false;
    }
    if (fclose(file) != 0) {
        success = false;
    }
    return success;
}

void output_writer_destroy(OutputWriter *writer)
{
    if (writer == NULL) {
        return;
    }
    real_buffer_destroy(&writer->pack_buffer);
    memset(writer, 0, sizeof(*writer));
}
