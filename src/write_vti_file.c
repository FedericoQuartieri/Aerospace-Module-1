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
    uint32_t block_size = n_elems * elem_size;

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
    size_t offset_P  = 0;
    size_t offset_Ux = offset_P  + sizeof(uint32_t) + block_size;
    size_t offset_Uy = offset_Ux + sizeof(uint32_t) + block_size;
    size_t offset_Uz = offset_Uy + sizeof(uint32_t) + block_size;

    fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Pressure\"  format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_P);
    fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Velocity_x\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_Ux);
    fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Velocity_y\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_Uy);
    fprintf(f,
        "        <DataArray type=\"%s\" Name=\"Velocity_z\" format=\"appended\" offset=\"%zu\"/>\n",
        vtk_type, offset_Uz);

    fprintf(f,
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "  <AppendedData encoding=\"raw\">\n   _");  // Spazio + underscore

    // -------- RAW BLOCKS --------
    fwrite(&block_size, sizeof(uint32_t), 1, f);
    fwrite(P->p, elem_size, n_elems, f);

    fwrite(&block_size, sizeof(uint32_t), 1, f);
    fwrite(U->v_x, elem_size, n_elems, f);

    fwrite(&block_size, sizeof(uint32_t), 1, f);
    fwrite(U->v_y, elem_size, n_elems, f);

    fwrite(&block_size, sizeof(uint32_t), 1, f);
    fwrite(U->v_z, elem_size, n_elems, f);

    fprintf(f, "\n  </AppendedData>\n</VTKFile>\n");
    fclose(f);
}
