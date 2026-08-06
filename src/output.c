#include "output.h"
#include "solver.h"
#include "utils.h"
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef USE_FLOAT
#define VTK_REAL_TYPE "Float32"
#else
#define VTK_REAL_TYPE "Float64"
#endif

/*
 * The written piece covers the cells owned by the caller, expressed in global
 * indices, while WholeExtent always describes the full grid.  With a single
 * block the two coincide; once the grid is split each process can write its
 * own piece of the same dataset without changing anything else here.
 */
static void write_extents(FILE *fp, const Decomp *d) {
    fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%.6e %.6e %.6e\">\n",
            d->n_global[0] - 1, d->n_global[1] - 1, d->n_global[2] - 1,
            DX, DY, DZ);
    fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n",
            d->start[0], d->start[0] + d->n[0] - 1,
            d->start[1], d->start[1] + d->n[1] - 1,
            d->start[2], d->start[2] + d->n[2] - 1);
}

/*
 * 1. STANDARD ASCII OUTPUT
 * Plain and human-readable format inside the <DataArray format="ascii"> tag.
 * Ideal for debugging, quick verification, or small grid sizes.
 */
void write_vti_ascii(const Decomp *d,
                     const SolverMemState *solver_mem_state,
                     const char *output_directory,
                     int t_step) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/sol_ascii_%04d.vti",
             output_directory, t_step);

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        perror("Error opening ASCII output file");
        return;
    }

    // XML header with grid dimensions and spacing
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    write_extents(fp, d);
    fprintf(fp, "      <PointData Scalars=\"pressure\" Vectors=\"velocity\">\n");

    // Write pressure
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"pressure\" format=\"ascii\">\n          ", VTK_REAL_TYPE);
    size_t written = 0;
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);
            for (int i = 0; i < d->n[0]; i++) {
                fprintf(fp, "%g ", solver_mem_state->pressure.v[row + (size_t)i]);
                if (++written % 16 == 0) fprintf(fp, "\n          ");
            }
        }
    }
    fprintf(fp, "\n        </DataArray>\n");

    // Write velocity (interleaving u_x, u_y, u_z for each node)
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n          ", VTK_REAL_TYPE);
    written = 0;
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);
            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                fprintf(fp, "%g %g %g ", solver_mem_state->u.v_x[index],
                                         solver_mem_state->u.v_y[index],
                                         solver_mem_state->u.v_z[index]);
                if (++written % 8 == 0) fprintf(fp, "\n          ");
            }
        }
    }
    fprintf(fp, "\n        </DataArray>\n");

    // Write Brinkman permeability (interleaving K_x, K_y, K_z)
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"permeability\" NumberOfComponents=\"3\" format=\"ascii\">\n          ", VTK_REAL_TYPE);
    written = 0;
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);
            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;
                fprintf(fp, "%g %g %g ", solver_mem_state->k.v_x[index],
                                         solver_mem_state->k.v_y[index],
                                         solver_mem_state->k.v_z[index]);
                if (++written % 8 == 0) fprintf(fp, "\n          ");
            }
        }
    }
    fprintf(fp, "\n        </DataArray>\n");

    // XML footer
    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}

/*
 * 2. OPTIMIZED BINARY OUTPUT (Appended Raw Binary)
 * Leverages the native VTK XML binary format ("appended").
 * Data arrays are dumped directly to disk using fwrite() at memory bus speed.
 * Instead of allocating a ~50MB temporary heap array to convert velocity
 * from SoA (Structure of Arrays) to AoS (Array of Structures), we use a small
 * 1024-cell stack buffer to maximize L1 cache efficiency and avoid heap overhead.
 */
#define CHUNK_SIZE 1024

/* Interleave three component arrays into the appended data block. */
static void write_vector_block(FILE *fp,
                               const Decomp *d,
                               const Real *restrict cx,
                               const Real *restrict cy,
                               const Real *restrict cz) {
    Real vec_buf[3 * CHUNK_SIZE];
    size_t filled = 0;

    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                size_t index = row + (size_t)i;

                vec_buf[3 * filled + 0] = cx[index];
                vec_buf[3 * filled + 1] = cy[index];
                vec_buf[3 * filled + 2] = cz[index];

                if (++filled == CHUNK_SIZE) {
                    fwrite(vec_buf, sizeof(Real), 3 * filled, fp);
                    filled = 0;
                }
            }
        }
    }

    if (filled > 0) {
        fwrite(vec_buf, sizeof(Real), 3 * filled, fp);
    }
}

void write_vti_binary(const Decomp *d,
                      const SolverMemState *solver_mem_state,
                      const char *output_directory,
                      int t_step) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/sol_%04d.vti",
             output_directory, t_step);

    /* Binary mode: the appended section holds raw bytes, and on Windows a
     * text-mode stream would rewrite every 0x0A inside them. */
    FILE *fp = fopen(filepath, "wb");
    if (!fp) {
        perror("Error opening binary output file");
        return;
    }

    size_t local_cells = (size_t)d->n[0] * (size_t)d->n[1] * (size_t)d->n[2];

    // Calculate byte offsets for the AppendedData section.
    // Each appended block starts with a uint64_t header indicating the block size in bytes.
    uint64_t bytes_p = (uint64_t)local_cells * sizeof(Real);
    uint64_t bytes_u = (uint64_t)local_cells * 3 * sizeof(Real);
    uint64_t bytes_k = (uint64_t)local_cells * 3 * sizeof(Real);
    size_t offset_p  = 0;
    size_t offset_u  = sizeof(uint64_t) + bytes_p;
    size_t offset_k  = offset_u + sizeof(uint64_t) + bytes_u;

    // XML header
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    write_extents(fp, d);
    fprintf(fp, "      <PointData Scalars=\"pressure\" Vectors=\"velocity\">\n");
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"pressure\" format=\"appended\" offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_p);
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_u);
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"permeability\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_k);
    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

    // 1. Write binary pressure: block size header + row-by-row dump
    fwrite(&bytes_p, sizeof(uint64_t), 1, fp);
    for (int k = 0; k < d->n[2]; k++) {
        for (int j = 0; j < d->n[1]; j++) {
            size_t row = decomp_index(d, 0, j, k);
            fwrite(solver_mem_state->pressure.v + row, sizeof(Real),
                   (size_t)d->n[0], fp);
        }
    }

    // 2. Write binary velocity: block size header + chunked SoA -> AoS interleaving
    fwrite(&bytes_u, sizeof(uint64_t), 1, fp);
    write_vector_block(fp, d, solver_mem_state->u.v_x,
                       solver_mem_state->u.v_y, solver_mem_state->u.v_z);

    // 3. Write permeability: block size header + chunked SoA -> AoS
    fwrite(&bytes_k, sizeof(uint64_t), 1, fp);
    write_vector_block(fp, d, solver_mem_state->k.v_x,
                       solver_mem_state->k.v_y, solver_mem_state->k.v_z);

    // XML footer (after raw binary data)
    fprintf(fp, "\n  </AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}

#undef CHUNK_SIZE

static int make_directory(const char *path) {
    int result;

#if defined(_WIN32)
    result = mkdir(path);
#else
    result = mkdir(path, 0777);
#endif

    if (result == 0 || errno == EEXIST) {
        return 1;
    }

    perror(path);
    return 0;
}

void write_to_file(const Decomp *d,
                   const SolverMemState *solver_mem_state,
                   const char *data_name,
                   int t_step) {
    static const char fallback_name[] = "unnamed";
    const char *scenario_name =
        data_name != NULL && data_name[0] != '\0'
            ? data_name
            : fallback_name;
    char output_directory[512];
    int written = snprintf(output_directory,
                                 sizeof(output_directory),
                                 "output/%s",
                                 scenario_name);

    if (written < 0 || (size_t)written >= sizeof(output_directory)) {
        fprintf(stderr, "Output directory path is too long\n");
        return;
    }

    if (!make_directory("output") ||
        !make_directory(output_directory)) {
        return;
    }

    // By default, use the high-performance binary writer.
    // Replace with write_vti_ascii(d, solver_mem_state, ...) if human-readable output is preferred.
    write_vti_binary(d, solver_mem_state, output_directory, t_step);
}
