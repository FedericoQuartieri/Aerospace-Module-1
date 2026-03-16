#include "test_common.h"
#include "../include/force_field.h"

/*
 * Tests on the ForceField data structure.
 *
 * ForceField holds three DTYPE arrays (f_x, f_y, f_z) of size GRID_ELEMENTS,
 * one for each velocity component. These tests verify:
 *   1. Correct allocation (non-NULL pointers, distinct arrays)
 *   2. rand_fill produces values in [0,1] across all components
 *   3. The three components are filled independently (not the same pointer)
 */

/* Test 1: Allocation – all pointers non-NULL and distinct */
int test_force_field_allocation(void) {
    printf("\n====== TEST: ForceField Allocation ======\n");

    ForceField f;
    initialize_force_field(&f);

    ASSERT_TRUE(f.f_x != NULL, "f_x should not be NULL after initialization");
    ASSERT_TRUE(f.f_y != NULL, "f_y should not be NULL after initialization");
    ASSERT_TRUE(f.f_z != NULL, "f_z should not be NULL after initialization");

    ASSERT_TRUE(f.f_x != f.f_y, "f_x and f_y must be distinct arrays");
    ASSERT_TRUE(f.f_x != f.f_z, "f_x and f_z must be distinct arrays");
    ASSERT_TRUE(f.f_y != f.f_z, "f_y and f_z must be distinct arrays");

    free_force_field(&f);

    printf("PASS: ForceField allocation is correct.\n");
    return TEST_PASS;
}

/* Test 2: rand_fill – all values in [0, 1] for every component */
int test_force_field_rand_fill(void) {
    printf("\n====== TEST: ForceField rand_fill Values in [0,1] ======\n");

    ForceField f;
    initialize_force_field(&f);
    rand_fill_force_field(&f);

    int out_of_range = 0;

    for (size_t i = 0; i < GRID_ELEMENTS; i++) {
        if (f.f_x[i] < 0.0 || f.f_x[i] > 1.0) out_of_range++;
        if (f.f_y[i] < 0.0 || f.f_y[i] > 1.0) out_of_range++;
        if (f.f_z[i] < 0.0 || f.f_z[i] > 1.0) out_of_range++;
    }

    free_force_field(&f);

    if (out_of_range > 0) {
        fprintf(stderr, "FAIL: %d values out of [0,1] range\n", out_of_range);
        return TEST_FAIL;
    }

    printf("PASS: All %zu values per component are in [0,1].\n", GRID_ELEMENTS);
    return TEST_PASS;
}

/* Test 3: Component independence – f_x, f_y, f_z are filled independently */
int test_force_field_components_independent(void) {
    printf("\n====== TEST: ForceField Component Independence ======\n");

    ForceField f;
    initialize_force_field(&f);
    rand_fill_force_field(&f);

    /*
     * With GRID_ELEMENTS random values it is statistically impossible for
     * two components to be identical element-by-element. We count mismatches
     * and expect at least one (i.e. the arrays differ).
     */
    int diff_xy = 0, diff_xz = 0;
    for (size_t i = 0; i < GRID_ELEMENTS; i++) {
        if (f.f_x[i] != f.f_y[i]) diff_xy++;
        if (f.f_x[i] != f.f_z[i]) diff_xz++;
    }

    free_force_field(&f);

    ASSERT_TRUE(diff_xy > 0, "f_x and f_y should differ (independently filled)");
    ASSERT_TRUE(diff_xz > 0, "f_x and f_z should differ (independently filled)");

    printf("PASS: Components are independently filled (%d / %d different pairs).\n",
           diff_xy, (int)GRID_ELEMENTS);
    return TEST_PASS;
}

/* Test 4: Free does not crash (basic smoke test) */
int test_force_field_free(void) {
    printf("\n====== TEST: ForceField Free (smoke test) ======\n");

    ForceField f;
    initialize_force_field(&f);
    rand_fill_force_field(&f);
    free_force_field(&f);   /* must not crash */

    printf("PASS: free_force_field completed without errors.\n");
    return TEST_PASS;
}

int main(void) {
    printf("========== ForceField Data Structure Tests ==========\n");
    printf("Grid: WIDTH=%d, HEIGHT=%d, DEPTH=%d  (%zu elements per component)\n",
           WIDTH, HEIGHT, DEPTH, (size_t)GRID_ELEMENTS);
    printf("=====================================================\n");

    int result = TEST_PASS;

    if (test_force_field_allocation()          != TEST_PASS) result = TEST_FAIL;
    if (test_force_field_rand_fill()           != TEST_PASS) result = TEST_FAIL;
    if (test_force_field_components_independent() != TEST_PASS) result = TEST_FAIL;
    if (test_force_field_free()                != TEST_PASS) result = TEST_FAIL;

    printf("\n=====================================================\n");
    printf(result == TEST_PASS ? "ALL TESTS PASSED\n" : "SOME TESTS FAILED\n");
    printf("=====================================================\n");

    return result;
}
