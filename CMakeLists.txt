cmake_minimum_required(VERSION 3.16)
project(navier_stokes_brinkman LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(GRID_WIDTH "128" CACHE STRING "Default grid width")
set(GRID_HEIGHT "128" CACHE STRING "Default grid height")
set(GRID_DEPTH "128" CACHE STRING "Default grid depth")
set(TIME_STEP "0.001" CACHE STRING "Default time step")
set(SIM_TOTAL_TIME "0.01" CACHE STRING "Default total time")
set(SOLVER_PRECISION "DOUBLE" CACHE STRING "FLOAT or DOUBLE")
set_property(CACHE SOLVER_PRECISION PROPERTY STRINGS FLOAT DOUBLE)

if(SOLVER_PRECISION STREQUAL "FLOAT")
    set(PRECISION_DEFINITION USE_FLOAT)
elseif(SOLVER_PRECISION STREQUAL "DOUBLE")
    set(PRECISION_DEFINITION USE_DOUBLE)
else()
    message(FATAL_ERROR "SOLVER_PRECISION must be FLOAT or DOUBLE")
endif()

set(SOLVER_COMMON_SOURCES
    src/field.c
    src/grid.c
    src/solver.c
    src/momentum.c
    src/pressure.c
    src/physics.c
    src/output.c
)

function(configure_solver_target target backend)
    target_include_directories(${target} PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
    )
    target_compile_definitions(${target} PUBLIC
        ${PRECISION_DEFINITION}
        SOLVER_BACKEND=${backend}
        SOLVER_DEFAULT_NX=${GRID_WIDTH}
        SOLVER_DEFAULT_NY=${GRID_HEIGHT}
        SOLVER_DEFAULT_NZ=${GRID_DEPTH}
        SOLVER_DEFAULT_DT=${TIME_STEP}
        SOLVER_DEFAULT_TOTAL_TIME=${SIM_TOTAL_TIME}
        _POSIX_C_SOURCE=200809L
    )
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(${target} PUBLIC m)
endfunction()

add_library(solver_core_standard STATIC
    ${SOLVER_COMMON_SOURCES}
    src/kernels_standard.c
)
configure_solver_target(solver_core_standard SOLVER_BACKEND_STANDARD)

add_library(solver_core_optimized STATIC
    ${SOLVER_COMMON_SOURCES}
    src/kernels_optimized.c
)
configure_solver_target(solver_core_optimized SOLVER_BACKEND_OPTIMIZED)

add_executable(solver_standard src/main.c)
target_link_libraries(solver_standard PRIVATE solver_core_standard)

add_executable(solver_optimized src/main.c)
target_link_libraries(solver_optimized PRIVATE solver_core_optimized)

add_executable(navier_stokes src/main.c)
target_link_libraries(navier_stokes PRIVATE solver_core_optimized)

enable_testing()

set(TEST_SUPPORT_SOURCES
    test/manufactured_cases.c
    test/test_support.c
)

add_executable(test_correctness_standard
    test/test_correctness.c
    ${TEST_SUPPORT_SOURCES}
)
target_include_directories(test_correctness_standard PRIVATE test)
target_link_libraries(test_correctness_standard PRIVATE solver_core_standard)

add_executable(test_correctness_optimized
    test/test_correctness.c
    ${TEST_SUPPORT_SOURCES}
)
target_include_directories(test_correctness_optimized PRIVATE test)
target_link_libraries(test_correctness_optimized PRIVATE solver_core_optimized)

add_executable(test_convergence
    test/test_convergence.c
    ${TEST_SUPPORT_SOURCES}
)
target_include_directories(test_convergence PRIVATE test)
target_link_libraries(test_convergence PRIVATE solver_core_standard)

add_executable(test_kernel_equivalence
    test/test_kernel_equivalence.c
    test/manufactured_cases.c
    src/field.c
    src/grid.c
    src/physics.c
    src/kernels_standard.c
    src/kernels_optimized.c
)
target_include_directories(test_kernel_equivalence PRIVATE
    include src test
)
target_compile_definitions(test_kernel_equivalence PRIVATE
    ${PRECISION_DEFINITION}
    _POSIX_C_SOURCE=200809L
)
target_compile_options(test_kernel_equivalence PRIVATE
    -Wall -Wextra -Wpedantic
)
target_link_libraries(test_kernel_equivalence PRIVATE m)

add_executable(test_output test/test_output.c)
target_link_libraries(test_output PRIVATE solver_core_standard)

add_executable(test_stats test/test_stats.c)
target_link_libraries(test_stats PRIVATE solver_core_standard)

# Performance probes are intentionally not registered with CTest: correctness
# tests remain capped at 64^3, while these executables may exercise 128^3 with
# output disabled.
add_executable(benchmark_x_standard
    test/benchmark_x.c
    test/manufactured_cases.c
)
target_include_directories(benchmark_x_standard PRIVATE test src)
target_link_libraries(benchmark_x_standard PRIVATE solver_core_standard)

add_executable(benchmark_x_optimized
    test/benchmark_x.c
    test/manufactured_cases.c
)
target_include_directories(benchmark_x_optimized PRIVATE test src)
target_link_libraries(benchmark_x_optimized PRIVATE solver_core_optimized)

add_library(solver_core_sanitized STATIC
    ${SOLVER_COMMON_SOURCES}
    src/kernels_standard.c
)
configure_solver_target(solver_core_sanitized SOLVER_BACKEND_STANDARD)
target_compile_options(solver_core_sanitized PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
)

add_executable(test_sanitizer_standard
    test/test_sanitizer.c
    ${TEST_SUPPORT_SOURCES}
)
target_include_directories(test_sanitizer_standard PRIVATE test)
target_compile_options(test_sanitizer_standard PRIVATE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
)
target_link_options(test_sanitizer_standard PRIVATE
    -fsanitize=address,undefined
)
target_link_libraries(test_sanitizer_standard PRIVATE solver_core_sanitized)

add_test(NAME correctness_standard COMMAND test_correctness_standard)
add_test(NAME correctness_optimized COMMAND test_correctness_optimized)
add_test(NAME kernel_equivalence COMMAND test_kernel_equivalence)
add_test(NAME convergence_spatial COMMAND test_convergence --spatial)
add_test(NAME convergence_temporal COMMAND test_convergence --temporal)
add_test(NAME output_synchronous COMMAND test_output)
add_test(NAME stats_report COMMAND test_stats)
add_test(NAME sanitizer_standard COMMAND test_sanitizer_standard)

set_tests_properties(correctness_standard correctness_optimized
    PROPERTIES LABELS correctness)
set_tests_properties(kernel_equivalence PROPERTIES LABELS kernel)
set_tests_properties(convergence_spatial convergence_temporal
    PROPERTIES LABELS convergence TIMEOUT 600)
set_tests_properties(output_synchronous PROPERTIES LABELS output)
set_tests_properties(stats_report PROPERTIES LABELS stats)
set_tests_properties(sanitizer_standard PROPERTIES LABELS sanitizer)

add_custom_target(run
    COMMAND $<TARGET_FILE:solver_optimized>
    DEPENDS solver_optimized
)
