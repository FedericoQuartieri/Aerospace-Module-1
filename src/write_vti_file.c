#include "write_vti_file.h"

#include <errno.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static const char* vtk_dtype()
{
    if (sizeof(DTYPE) == sizeof(float))
        return "Float32";
    if (sizeof(DTYPE) == sizeof(double))
        return "Float64";
    fprintf(stderr, "ERROR: Unsupported DTYPE size\n");
    return NULL;
}

static bool write_bytes(FILE *f, const void *buffer, size_t n_bytes)
{
    if (n_bytes == 0) {
        return true;
    }

    return fwrite(buffer, 1, n_bytes, f) == n_bytes;
}

static bool write_interleaved_velocity(FILE *f,
                                       const VelocityField *U,
                                       size_t n_elems)
{
    const size_t tuples_per_chunk = 16384;
    const size_t chunk_elems = 3 * tuples_per_chunk;
    DTYPE *buffer = (DTYPE *)xmalloc(chunk_elems * sizeof(DTYPE));
    if (!buffer) {
        fprintf(stderr, "ERROR: unable to allocate VTI write buffer\n");
        return false;
    }

    for (size_t base = 0; base < n_elems; base += tuples_per_chunk) {
        size_t chunk_len = n_elems - base;
        if (chunk_len > tuples_per_chunk) {
            chunk_len = tuples_per_chunk;
        }

        for (size_t i = 0; i < chunk_len; ++i) {
            size_t src_idx = base + i;
            size_t dst_idx = 3 * i;
            buffer[dst_idx]     = U->v_x[src_idx];
            buffer[dst_idx + 1] = U->v_y[src_idx];
            buffer[dst_idx + 2] = U->v_z[src_idx];
        }

        if (!write_bytes(f, buffer, 3 * chunk_len * sizeof(DTYPE))) {
            free(buffer);
            return false;
        }
    }

    free(buffer);
    return true;
}

static bool close_file(FILE *f, const char *filename)
{
    if (fclose(f) != 0) {
        fprintf(stderr, "ERROR: failed to close %s: %s\n", filename, strerror(errno));
        return false;
    }

    return true;
}

bool write_vti_file(const char *filename,
                    const VelocityField *U,
                    const Pressure      *P)
{
    const int Nx = WIDTH;
    const int Ny = HEIGHT;
    const int Nz = DEPTH;

    const double dx = DX;
    const double dy = DY;
    const double dz = DZ;

    FILE *f = fopen(filename, "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s for writing: %s\n", filename, strerror(errno));
        return false;
    }

    const char *vtk_type = vtk_dtype();
    if (!vtk_type) {
        fclose(f);
        return false;
    }

    size_t elem_size = sizeof(DTYPE);
    size_t n_elems   = (size_t)Nx * Ny * Nz; 
    size_t scalar_bytes = n_elems * elem_size;
    size_t vector_bytes = 3 * n_elems * elem_size;

    if (scalar_bytes > UINT32_MAX || vector_bytes > UINT32_MAX) {
        fprintf(stderr, "ERROR: VTI appended block exceeds 32-bit size limit for %s\n", filename);
        fclose(f);
        return false;
    }

    uint32_t scalar_block_size = (uint32_t)scalar_bytes;
    uint32_t vector_block_size = (uint32_t)vector_bytes;

    // -------- HEADER --------
    if (fprintf(f,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%g %g %g\">\n"
        "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
        "      <PointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n",
        Nx-1, Ny-1, Nz-1,
        dx, dy, dz,
        Nx-1, Ny-1, Nz-1
    ) < 0) {
        fprintf(stderr, "ERROR: failed to write VTI header for %s\n", filename);
        fclose(f);
        return false;
    }

    // -------- OFFSETS --------
    // Pressure: scalar field (1 component)
    // Velocity: vector field (3 components interleaved)
    size_t offset_P = 0;
    size_t offset_V = offset_P + sizeof(uint32_t) + scalar_block_size;

    if (fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Pressure\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_P) < 0) {
        fprintf(stderr, "ERROR: failed to write pressure metadata for %s\n", filename);
        fclose(f);
        return false;
    }

    if (fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_V) < 0) {
        fprintf(stderr, "ERROR: failed to write velocity metadata for %s\n", filename);
        fclose(f);
        return false;
    }

    if (fprintf(f,
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "  <AppendedData encoding=\"raw\">\n   _") < 0) {
        fprintf(stderr, "ERROR: failed to write appended-data header for %s\n", filename);
        fclose(f);
        return false;
    }

    // -------- RAW BLOCKS --------
    
    // Write Pressure (scalar)
    if (!write_bytes(f, &scalar_block_size, sizeof(uint32_t)) ||
        !write_bytes(f, P->p, scalar_bytes)) {
        fprintf(stderr, "ERROR: failed to write pressure block for %s\n", filename);
        fclose(f);
        return false;
    }

    // Write velocity in moderate chunks to avoid millions of tiny fwrite calls.
    if (!write_bytes(f, &vector_block_size, sizeof(uint32_t)) ||
        !write_interleaved_velocity(f, U, n_elems)) {
        fprintf(stderr, "ERROR: failed to write velocity block for %s\n", filename);
        fclose(f);
        return false;
    }

    if (fprintf(f, "\n  </AppendedData>\n</VTKFile>\n") < 0) {
        fprintf(stderr, "ERROR: failed to finalize VTI file %s\n", filename);
        fclose(f);
        return false;
    }

    return close_file(f, filename);
}
