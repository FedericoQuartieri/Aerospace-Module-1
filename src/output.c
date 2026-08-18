#include "output.h"
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

static void piece_extent(const Domain *domain, int extent[6]) {
    for (int axis = 0; axis < AXIS_COUNT; axis++) {
        extent[2 * axis] = domain->start[axis];
        /*
         * ImageData extents describe points, even when the arrays contain
         * cell data.  A block owning N cells therefore spans N + 1 points.
         * Adjacent MPI blocks consequently share their interface point.
         */
        extent[2 * axis + 1] =
            domain->start[axis] + domain->local[axis];
    }
}

static void write_header(FILE *fp, const Domain *domain) {
    int extent[6];

    piece_extent(domain, extent);
    fprintf(fp, "<VTKFile type=\"ImageData\" version=\"1.0\" "
                "byte_order=\"LittleEndian\" header_type=\"UInt64\">\n");
    fprintf(fp, "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
                "Origin=\"%.6e %.6e %.6e\" "
                "Spacing=\"%.6e %.6e %.6e\">\n",
            WIDTH, HEIGHT, DEPTH,
            -0.5 * (double)DX, -0.5 * (double)DY, -0.5 * (double)DZ,
            DX, DY, DZ);
    fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\">\n",
            extent[0], extent[1], extent[2], extent[3],
            extent[4], extent[5]);
    fprintf(fp, "      <CellData Scalars=\"pressure\" Vectors=\"velocity\">\n");

    uint64_t cells = (uint64_t)domain->owned_cells;
    uint64_t bytes_p = cells * sizeof(Real);
    uint64_t bytes_u = cells * 3 * sizeof(Real);
    size_t offset_u = sizeof(uint64_t) + bytes_p;
    size_t offset_k = offset_u + sizeof(uint64_t) + bytes_u;

    fprintf(fp, "        <DataArray type=\"%s\" Name=\"pressure\" "
                "format=\"appended\" offset=\"0\"/>\n", VTK_REAL_TYPE);
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"velocity\" "
                "NumberOfComponents=\"3\" format=\"appended\" "
                "offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_u);
    fprintf(fp, "        <DataArray type=\"%s\" Name=\"permeability\" "
                "NumberOfComponents=\"3\" format=\"appended\" "
                "offset=\"%zu\"/>\n", VTK_REAL_TYPE, offset_k);
}

static void write_scalar_binary(FILE *fp, const Real *field,
                                const Domain *domain) {
    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            size_t index = domain_index(domain, 0, j, k);
            fwrite(&field[index], sizeof(Real),
                   (size_t)domain->local[AXIS_X], fp);
        }
    }
}

static void write_vector_binary(FILE *fp, const VectorField *field,
                                const Domain *domain, Real *row_buffer) {
    int nx = domain->local[AXIS_X];

    for (int k = 0; k < domain->local[AXIS_Z]; k++) {
        for (int j = 0; j < domain->local[AXIS_Y]; j++) {
            size_t index = domain_index(domain, 0, j, k);
            for (int i = 0; i < nx; i++) {
                row_buffer[3 * i] = field->v_x[index + (size_t)i];
                row_buffer[3 * i + 1] = field->v_y[index + (size_t)i];
                row_buffer[3 * i + 2] = field->v_z[index + (size_t)i];
            }
            fwrite(row_buffer, sizeof(Real), 3 * (size_t)nx, fp);
        }
    }
}

void write_vti_binary(const SolverMemState *state,
                      const char *output_directory, int t_step) {
    const Domain *domain = &state->domain;
    char filepath[512];
    uint64_t cells = (uint64_t)domain->owned_cells;
    uint64_t bytes_p = cells * sizeof(Real);
    uint64_t bytes_v = cells * 3 * sizeof(Real);
    Real *row_buffer = xmalloc(3 * (size_t)domain->local[AXIS_X] *
                               sizeof(Real));

    snprintf(filepath, sizeof(filepath), "%s/sol_%04d_rank_%05d.vti",
             output_directory, t_step, domain->rank);
    FILE *fp = fopen(filepath, "wb");
    if (fp == NULL) {
        perror(filepath);
        free(row_buffer);
        return;
    }

    write_header(fp, domain);
    fprintf(fp, "      </CellData>\n    </Piece>\n  </ImageData>\n");
    fprintf(fp, "  <AppendedData encoding=\"raw\">\n_");

    fwrite(&bytes_p, sizeof(bytes_p), 1, fp);
    write_scalar_binary(fp, state->pressure.v, domain);
    fwrite(&bytes_v, sizeof(bytes_v), 1, fp);
    write_vector_binary(fp, &state->u, domain, row_buffer);
    fwrite(&bytes_v, sizeof(bytes_v), 1, fp);
    write_vector_binary(fp, &state->k, domain, row_buffer);
    fprintf(fp, "\n  </AppendedData>\n</VTKFile>\n");

    fclose(fp);
    free(row_buffer);
}

static int make_directory(const char *path) {
    int result;

#if defined(_WIN32)
    result = mkdir(path);
#else
    result = mkdir(path, 0777);
#endif
    if (result == 0 || errno == EEXIST) return 1;
    perror(path);
    return 0;
}

static void write_pvti(const Domain *domain, const char *output_directory,
                       int t_step, const int *extents) {
    char filepath[512];

    snprintf(filepath, sizeof(filepath), "%s/sol_%04d.pvti",
             output_directory, t_step);
    FILE *fp = fopen(filepath, "w");
    if (fp == NULL) {
        perror(filepath);
        return;
    }

    fprintf(fp, "<VTKFile type=\"PImageData\" version=\"1.0\" "
                "byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PImageData WholeExtent=\"0 %d 0 %d 0 %d\" "
                "Origin=\"%.6e %.6e %.6e\" "
                "Spacing=\"%.6e %.6e %.6e\" "
                "GhostLevel=\"0\">\n",
            WIDTH, HEIGHT, DEPTH,
            -0.5 * (double)DX, -0.5 * (double)DY, -0.5 * (double)DZ,
            DX, DY, DZ);
    fprintf(fp, "    <PCellData Scalars=\"pressure\" Vectors=\"velocity\">\n"
                "      <PDataArray type=\"%s\" Name=\"pressure\"/>\n"
                "      <PDataArray type=\"%s\" Name=\"velocity\" "
                "NumberOfComponents=\"3\"/>\n"
                "      <PDataArray type=\"%s\" Name=\"permeability\" "
                "NumberOfComponents=\"3\"/>\n"
                "    </PCellData>\n",
            VTK_REAL_TYPE, VTK_REAL_TYPE, VTK_REAL_TYPE);

    for (int rank = 0; rank < domain->size; rank++) {
        const int *extent = &extents[6 * rank];
        fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\" "
                    "Source=\"sol_%04d_rank_%05d.vti\"/>\n",
                extent[0], extent[1], extent[2], extent[3],
                extent[4], extent[5], t_step, rank);
    }
    fprintf(fp, "  </PImageData>\n</VTKFile>\n");
    fclose(fp);
}

void write_to_file(const SolverMemState *state, const char *data_name,
                   int t_step) {
    const Domain *domain = &state->domain;
    const char *scenario = data_name != NULL && data_name[0] != '\0'
        ? data_name : "unnamed";
    char output_directory[512];
    int local_extent[6];
    int *all_extents = domain->rank == 0
        ? xmalloc(6 * (size_t)domain->size * sizeof(int)) : NULL;

    snprintf(output_directory, sizeof(output_directory), "output/%s", scenario);
    if (domain->rank == 0) {
        if (!make_directory("output") || !make_directory(output_directory)) {
            MPI_Abort(domain->cart, EXIT_FAILURE);
        }
    }
    MPI_Barrier(domain->cart);

    piece_extent(domain, local_extent);
    MPI_Gather(local_extent, 6, MPI_INT, all_extents, 6, MPI_INT,
               0, domain->cart);
    write_vti_binary(state, output_directory, t_step);
    MPI_Barrier(domain->cart);
    if (domain->rank == 0) {
        write_pvti(domain, output_directory, t_step, all_extents);
        free(all_extents);
    }
}
