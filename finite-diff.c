#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <immintrin.h>

/* WARNING: multiple of tile size */
#define WIDTH 512
#define HEIGHT 512
#define DEPTH 512

#ifdef FLOAT
    #define TILE_WIDTH 128
    #define DTYPE float
#else
    #define TILE_WIDTH 64
    #define DTYPE double
#endif

#define TILE_HEIGHT 64
#define TILE_DEPTH 1

#define TIME_IT(func_call, avg_iter)                            \
do {                                                            \
    long elapsed_ns_avg = 0;                                    \
    func_call; /* Warmup run. */                                \
    for (int i = 0; i < avg_iter; ++i) {                        \
        struct timespec start, stop;                            \
        clock_gettime(CLOCK_MONOTONIC, &start);                 \
        func_call;                                              \
        clock_gettime(CLOCK_MONOTONIC, &stop);                  \
        elapsed_ns_avg += (stop.tv_sec - start.tv_sec) * 1e9 +  \
                          (stop.tv_nsec - start.tv_nsec);       \
    }                                                           \
    printf(#func_call " [" #avg_iter " runs avg] %f ms\n",      \
           elapsed_ns_avg / (1e6 * avg_iter));                  \
} while (0)

static void rand_fill(DTYPE *dst, size_t count)
{
    for (size_t i = 0; i < count; ++i) {
        dst[i] = ((DTYPE) rand()) / RAND_MAX;
    }
}

static inline __attribute__((always_inline)) size_t rowmaj_idx(size_t i,
                                                               size_t j,
                                                               size_t k,
                                                               size_t height,
                                                               size_t width)
{
    size_t face_size = width * height;
    return i * face_size + j * width + k;
}

void rowmaj_to_tiled(const DTYPE *__restrict__ src,
                     int depth,
                     int height,
                     int width,
                     int tile_depth,
                     int tile_height,
                     int tile_width,
                     DTYPE *__restrict__ dst)
{
    size_t depth_in_tiles = (depth - 1) / tile_depth + 1;
    size_t height_in_tiles = (height - 1) / tile_height + 1;
    size_t width_in_tiles = (width - 1) / tile_width + 1;
    size_t tile_size = tile_depth * tile_height * tile_width;
    size_t tile_row_size = tile_size * width_in_tiles;
    size_t tile_face_size = tile_row_size * height_in_tiles;

    for (int tile_i = 0; tile_i < depth_in_tiles; ++tile_i) {
        for (int tile_j = 0; tile_j < height_in_tiles; ++tile_j) {
            for (int tile_k = 0; tile_k < width_in_tiles; ++tile_k) {

                size_t dst_idx_offset = tile_face_size * tile_i +
                                        tile_row_size * tile_j +
                                        tile_size * tile_k;

                for (int i = 0; i < tile_depth; ++i) {
                    for (int j = 0; j < tile_height; ++j) {
                        for (int k = 0; k < tile_width; ++k) {

                            size_t dst_idx = dst_idx_offset +
                                rowmaj_idx(i, j, k, tile_height, tile_width);

                            size_t src_idx = rowmaj_idx(tile_i * tile_depth + i,
                                                        tile_j * tile_height + j,
                                                        tile_k * tile_width + k,
                                                        height,
                                                        width);

                            dst[dst_idx] = src[src_idx];
                        }
                    }
                }
            }
        }
    }
}

void comp_grad(const DTYPE *__restrict__ field,
               int depth,
               int height,
               int width,
               DTYPE *__restrict__ grad_i,
               DTYPE *__restrict__ grad_j,
               DTYPE *__restrict__ grad_k)
{
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; ++k) {

                size_t idx = rowmaj_idx(i, j, k, height, width);
                DTYPE value = field[idx];

                grad_i[idx] =
                    field[rowmaj_idx(i + 1, j, k, height, width)] - value;
                grad_j[idx] =
                    field[rowmaj_idx(i, j + 1, k, height, width)] - value;
                grad_k[idx] =
                    field[rowmaj_idx(i, j, k + 1, height, width)] - value;
            }
        }
    }
}

void comp_grad_tiled(const DTYPE *__restrict__ field,
                     int depth,
                     int height,
                     int width,
                     int tile_depth,
                     int tile_height,
                     int tile_width,
                     DTYPE *__restrict__ grad_i,
                     DTYPE *__restrict__ grad_j,
                     DTYPE *__restrict__ grad_k)
{
    size_t height_in_tiles = (height - 1) / tile_height + 1;
    size_t width_in_tiles = (width - 1) / tile_width + 1;

    size_t tile_size = tile_depth * tile_height * tile_width;
    size_t tile_row_size = tile_size * width_in_tiles;
    size_t tile_face_size = tile_row_size * height_in_tiles;

    size_t max_idx = width * height * depth;
    for (size_t tile_offset = 0; tile_offset < max_idx;
                                 tile_offset += tile_size) {

        for (int i = 0; i < tile_depth; ++i) {
            for (int j = 0; j < tile_height; ++j) {
                for (int k = 0; k < tile_width; ++k) {

                    size_t idx = tile_offset +
                        rowmaj_idx(i, j, k, tile_height, tile_width);

                    DTYPE value = field[idx];

                    size_t idx_next_i = idx + ((i == tile_depth - 1) ?
                        (tile_face_size -
                         i * tile_width * tile_height) :
                        (tile_width * tile_height));

                    size_t idx_next_j = idx + ((j == tile_height - 1) ?
                        (tile_row_size - j * tile_width) : tile_width);

                    size_t idx_next_k = idx + ((k == tile_width - 1) ?
                        (tile_size - k) : 1);

                    grad_i[idx] = field[idx_next_i] - value;
                    grad_j[idx] = field[idx_next_j] - value;
                    grad_k[idx] = field[idx_next_k] - value;
                }
            }
        }
    }
}

#ifdef FLOAT
    #define VLEN 8
    #define VTYPE __m256
    #define VLOAD _mm256_load_ps
    #define VLOADU _mm256_loadu_ps
    #define VSTORE _mm256_store_ps
    #define VSUB _mm256_sub_ps
#else
    #define VLEN 4
    #define VTYPE __m256d
    #define VLOAD _mm256_load_pd
    #define VLOADU _mm256_loadu_pd
    #define VSTORE _mm256_store_pd
    #define VSUB _mm256_sub_pd
#endif

void comp_grad_vectorized(const DTYPE *__restrict__ src,
                          int depth,
                          int height,
                          int width,
                          DTYPE *__restrict__ dst_i,
                          DTYPE *__restrict__ dst_j,
                          DTYPE *__restrict__ dst_k)
{
    for (int i = 0; i < depth; ++i) {
        for (int j = 0; j < height; ++j) {
            for (int k = 0; k < width; k += VLEN) {

                size_t idx = rowmaj_idx(i, j, k, height, width);

                VTYPE vcurr = VLOAD(&src[idx]);
                VTYPE vnextk = VLOADU(&src[idx + 1]);
                VTYPE vnextj = VLOAD(&src[idx + width]);
                VTYPE vnexti = VLOAD(&src[idx + (width * height)]);

                VTYPE vsub = VSUB(vnextk, vcurr);
                VSTORE(&dst_k[idx], vsub);
                vsub = VSUB(vnextj, vcurr);
                VSTORE(&dst_j[idx], vsub);
                vsub = VSUB(vnexti, vcurr);
                VSTORE(&dst_i[idx], vsub);
            }
        }
    }
}

/* WARNING: Tiling only access, this suffers a lot from TLB misses. */
void comp_grad_vectorized_tiled_loop(const DTYPE *__restrict__ src,
                                     int depth,
                                     int height,
                                     int width,
                                     int tile_depth,
                                     int tile_height,
                                     int tile_width,
                                     DTYPE *__restrict__ dst_i,
                                     DTYPE *__restrict__ dst_j,
                                     DTYPE *__restrict__ dst_k)
{
    for (int ti = 0; ti < depth; ti += tile_depth) {
        for (int tj = 0; tj < height; tj += tile_height) {
            for (int tk = 0; tk < width; tk += tile_width) {

                for (int i = 0; i < tile_depth; ++i) {
                    for (int j = 0; j < tile_height; ++j) {
                        for (int k = 0; k < tile_width; k += VLEN) {

                            size_t idx = rowmaj_idx(ti + i, tj + j, tk + k, height, width);

                            VTYPE vcurr = VLOAD(&src[idx]);
                            VTYPE vnextk = VLOADU(&src[idx + 1]);
                            VTYPE vnextj = VLOAD(&src[idx + width]);
                            VTYPE vnexti = VLOAD(&src[idx + (width * height)]);

                            VTYPE vsub = VSUB(vnextk, vcurr);
                            VSTORE(&dst_k[idx], vsub);
                            vsub = VSUB(vnextj, vcurr);
                            VSTORE(&dst_j[idx], vsub);
                            vsub = VSUB(vnexti, vcurr);
                            VSTORE(&dst_i[idx], vsub);
                        }
                    }
                }
            }
        }
    }
}

int main(void)
{
    /* +1 for ghost cells. */
    const size_t size = (WIDTH + VLEN) * (HEIGHT + VLEN) * (DEPTH + VLEN) * sizeof(DTYPE);

    /* Allocation must be aligned to 32-byte
     * boundaries for AVX aligned loads/stores. */
    DTYPE *field = aligned_alloc(32, size);
    DTYPE *field_tiled = aligned_alloc(32, size);
    DTYPE *grad_x = aligned_alloc(32, size);
    DTYPE *grad_y = aligned_alloc(32, size);
    DTYPE *grad_z = aligned_alloc(32, size);
    DTYPE *grad_x_tiled = aligned_alloc(32, size);
    DTYPE *grad_y_tiled = aligned_alloc(32, size);
    DTYPE *grad_z_tiled = aligned_alloc(32, size);

    memset(field, 0, size);
    memset(field_tiled, 0, size);
    memset(grad_x, 0, size);
    memset(grad_y, 0, size);
    memset(grad_z, 0, size);
    memset(grad_x_tiled, 0, size);
    memset(grad_y_tiled, 0, size);
    memset(grad_z_tiled, 0, size);

    rand_fill(field, WIDTH * HEIGHT * DEPTH);

    rowmaj_to_tiled(field,
                    DEPTH,
                    HEIGHT,
                    WIDTH,
                    TILE_DEPTH,
                    TILE_HEIGHT,
                    TILE_WIDTH,
                    field_tiled);

    TIME_IT(comp_grad(field, DEPTH, HEIGHT, WIDTH, grad_z, grad_y, grad_x), 5);
    TIME_IT(comp_grad_vectorized(field, DEPTH, HEIGHT, WIDTH, grad_z, grad_y, grad_x), 5);
    TIME_IT(comp_grad_vectorized_tiled_loop(field,
                                            DEPTH,
                                            HEIGHT,
                                            WIDTH,
                                            TILE_DEPTH,
                                            TILE_HEIGHT,
                                            TILE_WIDTH,
                                            grad_z,
                                            grad_y,
                                            grad_x), 5);
    TIME_IT(comp_grad_tiled(field_tiled,
                            DEPTH,
                            HEIGHT,
                            WIDTH,
                            TILE_DEPTH,
                            TILE_HEIGHT,
                            TILE_WIDTH,
                            grad_z_tiled,
                            grad_y_tiled,
                            grad_x_tiled), 5);

    /* Manually comparing the results, a memcmp won't do here. */
    size_t height_in_tiles = (HEIGHT - 1) / TILE_HEIGHT + 1;
    size_t width_in_tiles = (WIDTH - 1) / TILE_WIDTH + 1;
    size_t tile_size = TILE_DEPTH * TILE_HEIGHT * TILE_WIDTH;
    size_t tile_row_size = tile_size * width_in_tiles;
    size_t tile_face_size = tile_row_size * height_in_tiles;
    int same_result = 1;
    /* We don't care about the last row/face. */
    for (int i = 0; i < DEPTH - 1; ++i) {
        for (int j = 0; j < HEIGHT - 1; ++j) {
            for (int k = 0; k < WIDTH - 1; ++k) {

                int tile_i = i / TILE_DEPTH;
                int tile_rel_i = i % TILE_DEPTH;
                int tile_j = j / TILE_HEIGHT;
                int tile_rel_j = j % TILE_HEIGHT;
                int tile_k = k / TILE_WIDTH;
                int tile_rel_k = k % TILE_WIDTH;

                size_t tiled_idx = tile_face_size * tile_i +
                                   tile_row_size * tile_j +
                                   tile_size * tile_k +
                                   rowmaj_idx(tile_rel_i,
                                              tile_rel_j,
                                              tile_rel_k,
                                              TILE_HEIGHT,
                                              TILE_WIDTH);

                if (grad_x[rowmaj_idx(i, j, k, HEIGHT, WIDTH)] !=
                    grad_x_tiled[tiled_idx] ||
                    grad_y[rowmaj_idx(i, j, k, HEIGHT, WIDTH)] !=
                    grad_y_tiled[tiled_idx] ||
                    grad_z[rowmaj_idx(i, j, k, HEIGHT, WIDTH)] !=
                    grad_z_tiled[tiled_idx]) {
                    same_result = 0;
                }
            }
        }
    }

    if (same_result) {
        printf("[SUCCESS] matching outputs\n");
    } else {
        printf("[FAILURE] mismatching outputs\n");
    }

    free(grad_z_tiled);
    free(grad_y_tiled);
    free(grad_x_tiled);

    free(grad_z);
    free(grad_y);
    free(grad_x);

    free(field_tiled);
    free(field);

    return 0;
}
