#include "output.h"
#include "solver.h"
#include "parallel.h"
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

/* Un pezzo per processo; l'indice .pvti li elenca tutti. */
#define PIECE_NAME "sol_%04d_p%03d.vti"

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
 * Le componenti della velocita' vivono su griglia staggered (MAC): rispetto al
 * nodo (i, j, k) dove sta la pressione, v_x e' mezza cella avanti lungo X, v_y
 * lungo Y e v_z lungo Z -- e' quello che fa vectorField_fill chiamando
 * staggered_physical_coord sulla direzione della componente.
 *
 * Il formato ImageData pero' dichiara un solo Origin per tutti gli array di
 * punti, quindi scrivere le tre componenti grezze dello stesso indice come un
 * vettore mette insieme tre valori presi in tre posti diversi: il modulo che ne
 * esce non e' il modulo di nessuna velocita' reale, e sbaglia di O(h) anche
 * quando il campo e' esatto.  Prima di scrivere si riportano quindi tutte e tre
 * sul nodo centrato, lo stesso dove sta gia' la pressione.
 *
 *   nodo(i) = ( v[i-1] + v[i] ) / 2      v[i-1] sta in (i-0.5)h, v[i] in (i+0.5)h
 *
 * Sul muro inferiore del dominio il vicino in (-0.5)h non esiste: si estrapola
 * dai due nodi staggered piu' vicini, che resta del secondo ordine come la
 * media.
 *
 *   nodo(0) = 1.5 * v[0] - 0.5 * v[1]
 *
 * Al confine fra due blocchi il vicino invece c'e', e sta nel ghost layer che
 * solver.c aggiorna subito prima di scrivere: li' si usa la media, cosi' il
 * campo scritto non ha cuciture e non dipende da come la griglia e' spartita.
 */
enum node_rule {
    NODE_AVERAGE,      /* il vicino indietro esiste: media dei due */
    NODE_EXTRAPOLATE,  /* muro inferiore: estrapola dai due in avanti */
    NODE_RAW           /* blocco spesso una cella: non c'e' altro da usare */
};

static Real node_value(const Real *restrict component, size_t index,
                       ptrdiff_t step, enum node_rule rule) {
    switch (rule) {
    case NODE_EXTRAPOLATE:
        return (Real)1.5 * component[index] -
               (Real)0.5 * component[(size_t)((ptrdiff_t)index + step)];
    case NODE_RAW:
        return component[index];
    default:
        return (Real)0.5 * (component[(size_t)((ptrdiff_t)index - step)] +
                            component[index]);
    }
}

/* Quale regola vale per l'indice `i` lungo `axis`. */
static enum node_rule rule_for(const Decomp *d, int axis, int i) {
    if (i > 0) {
        return NODE_AVERAGE;
    }
    if (!d->is_first[axis]) {
        return NODE_AVERAGE;   /* i - 1 e' il ghost, riempito dall'halo */
    }
    return (d->n[axis] > 1) ? NODE_EXTRAPOLATE : NODE_RAW;
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
    snprintf(filepath, sizeof(filepath), "%s/sol_ascii_%04d_p%03d.vti",
             output_directory, t_step, par_rank());

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
        enum node_rule rule_z = rule_for(d, 2, k);
        for (int j = 0; j < d->n[1]; j++) {
            enum node_rule rule_y = rule_for(d, 1, j);
            size_t row = decomp_index(d, 0, j, k);
            for (int i = 0; i < d->n[0]; i++) {
                enum node_rule rule_x = rule_for(d, 0, i);
                size_t index = row + (size_t)i;
                fprintf(fp, "%g %g %g ",
                        node_value(solver_mem_state->u.v_x, index,
                                   (ptrdiff_t)d->stride[0], rule_x),
                        node_value(solver_mem_state->u.v_y, index,
                                   (ptrdiff_t)d->stride[1], rule_y),
                        node_value(solver_mem_state->u.v_z, index,
                                   (ptrdiff_t)d->stride[2], rule_z));
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

/*
 * Interleave three component arrays into the appended data block, riportando
 * ogni componente dal proprio punto staggered al nodo centrato.
 */
static void write_velocity_block(FILE *fp,
                                 const Decomp *d,
                                 const Real *restrict cx,
                                 const Real *restrict cy,
                                 const Real *restrict cz) {
    Real vec_buf[3 * CHUNK_SIZE];
    size_t filled = 0;
    ptrdiff_t step_x = (ptrdiff_t)d->stride[0];
    ptrdiff_t step_y = (ptrdiff_t)d->stride[1];
    ptrdiff_t step_z = (ptrdiff_t)d->stride[2];

    for (int k = 0; k < d->n[2]; k++) {
        enum node_rule rule_z = rule_for(d, 2, k);

        for (int j = 0; j < d->n[1]; j++) {
            enum node_rule rule_y = rule_for(d, 1, j);
            size_t row = decomp_index(d, 0, j, k);

            for (int i = 0; i < d->n[0]; i++) {
                enum node_rule rule_x = rule_for(d, 0, i);
                size_t index = row + (size_t)i;

                vec_buf[3 * filled + 0] = node_value(cx, index, step_x, rule_x);
                vec_buf[3 * filled + 1] = node_value(cy, index, step_y, rule_y);
                vec_buf[3 * filled + 2] = node_value(cz, index, step_z, rule_z);

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

/*
 * La permeabilita' e' staggered come la velocita', ma resta grezza: e' una
 * proprieta' del materiale, spesso un indicatore netto di ostacolo, e mediarla
 * ne sfumerebbe il bordo.  Meglio mezza cella di scarto che un contorno finto.
 */
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
    snprintf(filepath, sizeof(filepath), "%s/" PIECE_NAME,
             output_directory, t_step, par_rank());

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
    write_velocity_block(fp, d, solver_mem_state->u.v_x,
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

/*
 * L'indice che tiene insieme i pezzi.
 *
 * Nessuno raccoglie i dati: ogni processo scrive il proprio file, e il rank 0
 * scrive questo elenco di poche righe che dice a ParaView dove sta ciascun
 * pezzo. Aprendo il .pvti si vede il dominio intero.
 *
 * Le posizioni degli altri blocchi non se le fa dire da nessuno: conoscendo la
 * forma della griglia di processi, decomp_share ricalcola quali celle spettano
 * a ciascuno con la stessa regola con cui se le sono prese.
 */
static void write_pvti(const Decomp *d,
                       const char *output_directory,
                       int t_step) {
    if (par_rank() != 0) {
        return;
    }

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/sol_%04d.pvti",
             output_directory, t_step);

    FILE *fp = fopen(filepath, "w");
    if (!fp) {
        perror("Error opening pvti index");
        return;
    }

    int dims[3];
    par_dims(dims);

    fprintf(fp, "<VTKFile type=\"PImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n");
    fprintf(fp, "  <PImageData WholeExtent=\"0 %d 0 %d 0 %d\" GhostLevel=\"0\" Origin=\"0 0 0\" Spacing=\"%.6e %.6e %.6e\">\n",
            d->n_global[0] - 1, d->n_global[1] - 1, d->n_global[2] - 1,
            DX, DY, DZ);
    fprintf(fp, "    <PPointData Scalars=\"pressure\" Vectors=\"velocity\">\n");
    fprintf(fp, "      <PDataArray type=\"%s\" Name=\"pressure\"/>\n", VTK_REAL_TYPE);
    fprintf(fp, "      <PDataArray type=\"%s\" Name=\"velocity\" NumberOfComponents=\"3\"/>\n", VTK_REAL_TYPE);
    fprintf(fp, "      <PDataArray type=\"%s\" Name=\"permeability\" NumberOfComponents=\"3\"/>\n", VTK_REAL_TYPE);
    fprintf(fp, "    </PPointData>\n");

    for (int rank = 0; rank < par_size(); rank++) {
        int coords[3];
        int begin[3];
        int end[3];

        par_coords_of(rank, coords);
        for (int c = 0; c < 3; c++) {
            decomp_share(d->n_global[c], dims[c], coords[c],
                         &begin[c], &end[c]);
        }

        char piece[128];
        snprintf(piece, sizeof(piece), PIECE_NAME, t_step, rank);

        fprintf(fp, "    <Piece Extent=\"%d %d %d %d %d %d\" Source=\"%s\"/>\n",
                begin[0], end[0] - 1, begin[1], end[1] - 1,
                begin[2], end[2] - 1, piece);
    }

    fprintf(fp, "  </PImageData>\n");
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

void write_to_file(const Decomp *d,
                   const SolverMemState *solver_mem_state,
                   const char *data_name,
                   int t_step) {
    static const char fallback_name[] = "unnamed";
    const char *scenario_name =
        data_name != NULL && data_name[0] != '\0'
            ? data_name
            : fallback_name;
    /* Piu' corta dei buffer che la useranno, cosi' il nome del file ci sta. */
    char output_directory[256];
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
    write_pvti(d, output_directory, t_step);
}
