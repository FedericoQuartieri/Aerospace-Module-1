#include "write_vti_file.h"

static const char* vtk_dtype()
{
    if (sizeof(DTYPE) == sizeof(float))
        return "Float32";
    if (sizeof(DTYPE) == sizeof(double))
        return "Float64";
    fprintf(stderr, "ERROR: Unsupported DTYPE size\n");
    return "Float32";
}

void write_vti_file(const char *filename,
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
        fprintf(stderr, "ERROR: cannot open %s for writing\n", filename);
        return;
    }

    const char *vtk_type = vtk_dtype();
    size_t elem_size = sizeof(DTYPE);
    size_t n_elems   = (size_t)Nx * Ny * Nz; 
    uint32_t scalar_block_size = (uint32_t)(n_elems * elem_size);
    uint32_t vector_block_size = (uint32_t)(3 * n_elems * elem_size);

    // -------- HEADER --------
    fprintf(f,
        "<?xml version=\"1.0\"?>\n"
        "<VTKFile type=\"ImageData\" version=\"1.0\" byte_order=\"LittleEndian\">\n"
        "  <ImageData WholeExtent=\"0 %d 0 %d 0 %d\" Origin=\"0 0 0\" Spacing=\"%g %g %g\">\n"
        "    <Piece Extent=\"0 %d 0 %d 0 %d\">\n"
        "      <PointData Scalars=\"Pressure\" Vectors=\"Velocity\">\n",
        Nx-1, Ny-1, Nz-1,
        dx, dy, dz,
        Nx-1, Ny-1, Nz-1
    );

    // -------- OFFSETS --------
    // Pressure: scalar field (1 component)
    // Velocity: vector field (3 components interleaved)
    size_t offset_P = 0;
    size_t offset_V = offset_P + sizeof(uint32_t) + scalar_block_size;

    fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Pressure\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_P);
    fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Velocity\" NumberOfComponents=\"3\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_V);

    fprintf(f,
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "  <AppendedData encoding=\"raw\">\n   _");  // Space + underscore marker

    // -------- RAW BLOCKS --------
    
    // Write Pressure (scalar)
    fwrite(&scalar_block_size, sizeof(uint32_t), 1, f);
    fwrite(P->p, elem_size, n_elems, f);

    // Write Velocity (3-component vector, interleaved: vx,vy,vz for each point)
    fwrite(&vector_block_size, sizeof(uint32_t), 1, f);
    for (size_t idx = 0; idx < n_elems; idx++) {
        fwrite(&U->v_x[idx], elem_size, 1, f);
        fwrite(&U->v_y[idx], elem_size, 1, f);
        fwrite(&U->v_z[idx], elem_size, 1, f);
    }

    fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);
}
