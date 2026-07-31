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
 * 1. STANDARD ASCII OUTPUT
 * Plain and human-readable format inside the <DataArray format="ascii"> tag.
 * Ideal for debugging, quick verification, or small grid sizes.
 */
void write_vti_ascii(const SolverMemState *solver_mem_state,
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
    fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%.6e %.6e %.6e\">\n",
            WIDTH - 1, HEIGHT - 1, DEPTH - 1, DX, DY, DZ);
    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n",
            WIDTH - 1, HEIGHT - 1, DEPTH - 1);
    fprintf(fp, "      <PointData Scalars=\"pressure\" Vectors=\"velocity\">\n");

    // Write pressure
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"pressure\" format=\"ascii\">\n          ", VTK_REAL_TYPE);
    for (size_t i = 0; i < GRID_CELLS; i++) {
        fprintf(fp, "%g ", solver_mem_state->pressure.v[i]);
        if ((i + 1) % 16 == 0) fprintf(fp, "\n          ");
    }
    fprintf(fp, "\n        </DataArray>\n");

    // Write velocity (interleaving u_x, u_y, u_z for each node)
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"ascii\">\n          ", VTK_REAL_TYPE);
    for (size_t i = 0; i < GRID_CELLS; i++) {
        fprintf(fp, "%g %g %g ", solver_mem_state->u.v_x[i],
                                 solver_mem_state->u.v_y[i],
                                 solver_mem_state->u.v_z[i]);
        if ((i + 1) % 8 == 0) fprintf(fp, "\n          ");
    }
    fprintf(fp, "\n        </DataArray>\n");

    // Write Brinkman permeability (interleaving K_x, K_y, K_z)
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"permeability\" NumberOfComponents=\"3\" format=\"ascii\">\n          ", VTK_REAL_TYPE);
    for (size_t i = 0; i < GRID_CELLS; i++) {
        fprintf(fp, "%g %g %g ", solver_mem_state->k.v_x[i],
                                 solver_mem_state->k.v_y[i],
                                 solver_mem_state->k.v_z[i]);
        if ((i + 1) % 8 == 0) fprintf(fp, "\n          ");
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
void write_vti_binary(const SolverMemState *solver_mem_state,
                      const char *output_directory,
                      int t_step) {
    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/sol_%04d.vti",
             output_directory, t_step);

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        perror("Error opening binary output file");
        return;
    }

    // Calculate byte offsets for the AppendedData section.
    // Each appended block starts with a uint64_t header indicating the block size in bytes.
    uint64_t bytes_p = (uint64_t)GRID_CELLS * sizeof(Real);
    uint64_t bytes_u = (uint64_t)GRID_CELLS * 3 * sizeof(Real);
    uint64_t bytes_k = (uint64_t)GRID_CELLS * 3 * sizeof(Real);
    size_t offset_p  = 0;
    size_t offset_u  = sizeof(uint64_t) + bytes_p;
    size_t offset_k  = offset_u + sizeof(uint64_t) + bytes_u;

    // XML header
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%.6e %.6e %.6e\">\n",
            WIDTH - 1, HEIGHT - 1, DEPTH - 1, DX, DY, DZ);
    fprintf(fp, "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n",
            WIDTH - 1, HEIGHT - 1, DEPTH - 1);
    fprintf(fp, "      <PointData Scalars=\"pressure\" Vectors=\"velocity\">\n");
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"pressure\" format=\"appended\" offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_p);
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_u);
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"permeability\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_k);
    fprintf(fp, "      </PointData>\n");
    fprintf(fp, "    </Piece>\n");
    fprintf(fp, "  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

    // 1. Write binary pressure: block size header + direct memory dump
    fwrite(&bytes_p, sizeof(uint64_t), 1, fp);
    fwrite(solver_mem_state->pressure.v, sizeof(Real), GRID_CELLS, fp);

    // 2. Write binary velocity: block size header + chunked SoA -> AoS interleaving
    fwrite(&bytes_u, sizeof(uint64_t), 1, fp);
#define CHUNK_SIZE 1024
    Real vec_buf[3 * CHUNK_SIZE];
    for (size_t i = 0; i < GRID_CELLS; i += CHUNK_SIZE) {
        size_t n = (i + CHUNK_SIZE <= GRID_CELLS) ? CHUNK_SIZE : (GRID_CELLS - i);
        for (size_t j = 0; j < n; j++) {
            vec_buf[3 * j + 0] = solver_mem_state->u.v_x[i + j];
            vec_buf[3 * j + 1] = solver_mem_state->u.v_y[i + j];
            vec_buf[3 * j + 2] = solver_mem_state->u.v_z[i + j];
        }
        fwrite(vec_buf, sizeof(Real), 3 * n, fp);
    }

    // 3. Write permeability: block size header + chunked SoA -> AoS
    fwrite(&bytes_k, sizeof(uint64_t), 1, fp);
    for (size_t i = 0; i < GRID_CELLS; i += CHUNK_SIZE) {
        size_t n = (i + CHUNK_SIZE <= GRID_CELLS) ? CHUNK_SIZE : (GRID_CELLS - i);
        for (size_t j = 0; j < n; j++) {
            vec_buf[3 * j + 0] = solver_mem_state->k.v_x[i + j];
            vec_buf[3 * j + 1] = solver_mem_state->k.v_y[i + j];
            vec_buf[3 * j + 2] = solver_mem_state->k.v_z[i + j];
        }
        fwrite(vec_buf, sizeof(Real), 3 * n, fp);
    }
#undef CHUNK_SIZE

    // XML footer (after raw binary data)
    fprintf(fp, "\n  </AppendedData>\n");
    fprintf(fp, "</VTKFile>\n");

    fclose(fp);
}

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

void write_to_file(const SolverMemState *solver_mem_state,
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
    // Replace with write_vti_ascii(solver_mem_state, t_step) if human-readable output is preferred.
    write_vti_binary(solver_mem_state, output_directory, t_step);
}
