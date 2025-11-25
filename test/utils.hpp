#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <fstream>
#include <sstream>
#include <string>
#include <iostream>
#include <cctype>
#include <cstdint>

extern "C"
{
#include "g_field.h"
#include "velocity_field.h" 
#include "pressure.h"
}

/* ----- Helper function to read VTK file ----- */ 
bool read_last_vtk_file(VelocityField &U_numerical, Pressure &P_numerical)
{
    int last_step = STEPS - 1;

    std::stringstream filename;
    filename << "../build/output/solution_0000" << last_step << ".vti";

    std::ifstream file(filename.str(), std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Failed to open " << filename.str() << std::endl;
        return false;
    }

    std::string line;

    // 1. Find <AppendedData>
    while (std::getline(file, line)) {
        if (line.find("<AppendedData") != std::string::npos) {
            std::cout << "[DEBUG] Found <AppendedData>" << std::endl;
            break;
        }
    }

    // 2. Skip whitespace until we find '_'
    char ch;
    bool found_underscore = false;
    while (file.get(ch)) {
        if (ch == '_') {
            found_underscore = true;
            std::cout << "[DEBUG] Found '_' marker" << std::endl;
            break;
        }
        // Skip whitespace (spaces, newlines, etc.)
        if (!std::isspace(ch)) {
            std::cerr << "[ERROR] Expected whitespace or '_', got '" << ch << "' (ASCII " << (int)ch << ")\n";
            return false;
        }
    }

    if (!found_underscore) {
        std::cerr << "[ERROR] Could not find '_' marker after <AppendedData>\n";
        return false;
    }

    std::streampos data_start = file.tellg();
    std::cout << "[DEBUG] Binary data starts at position: " << data_start << std::endl;

    //file.seekg(data_start);

    uint32_t block_size = 0;
    const uint32_t expected = WIDTH * HEIGHT * DEPTH * sizeof(DTYPE);

    auto read_block = [&](const char* label, char* dst)
    {
        std::cout << "\n[DEBUG] Reading block: " << label << std::endl;

        file.read(reinterpret_cast<char*>(&block_size), sizeof(uint32_t));

        std::cout << "[DEBUG]   block_size read = " << block_size << std::endl;
        std::cout << "[DEBUG]   expected       = " << expected << std::endl;

        if (!file.good()) {
            std::cout << "[ERROR] file.read(block_size) failed while reading header of " << label << std::endl;
            return false;
        }

        if (block_size != expected) {
            std::cout << "[ERROR] Block size mismatch for " << label
                      << ". block_size=" << block_size
                      << " expected=" << expected << std::endl;
            return false;
        }

        // read block_size bytes
        file.read(dst, block_size);

        std::cout << "[DEBUG]   bytes actually read = " << file.gcount()
                  << " (should be " << block_size << ")" << std::endl;

        if (!file.good()) {
            std::cout << "[ERROR] file.read(data) failed while reading data for " << label << std::endl;
            return false;
        }

        return true;
    };

    // ---- PRESSURE ----
    if (!read_block("Pressure", (char*)P_numerical.p)) return false;

    // ---- U_x ----
    if (!read_block("Velocity_x", (char*)U_numerical.v_x)) return false;

    // ---- U_y ----
    if (!read_block("Velocity_y", (char*)U_numerical.v_y)) return false;

    // ---- U_z ----
    if (!read_block("Velocity_z", (char*)U_numerical.v_z)) return false;

    std::cout << "\n[DEBUG] Successfully read all VTI fields\n";

    return true;
} 


#endif // TEST_UTILS_HPP