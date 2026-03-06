cmake_minimum_required(VERSION 3.11)
project(navier_stokes LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable/disable Eigen-dependent tests
option(ENABLE_EIGEN_TESTS "Build tests that depend on Eigen" ON)

# Find pthread library
find_package(Threads REQUIRED)

# Find Eigen3 only if we want the tests that use it
if(ENABLE_EIGEN_TESTS)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# --- Tests (GoogleTest) ---
enable_testing()
add_subdirectory(third_party/googletest)

# Include GoogleTest module for gtest_discover_tests
include(GoogleTest)

# Global include directory for project headers
include_directories(include)

# Main sources (exclude main.c to create a reusable core library)
file(GLOB SRCS "src/*.c")
set(MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")
list(REMOVE_ITEM SRCS ${MAIN_SRC})

# Add tinyexpr.c directly to the core library sources
list(APPEND SRCS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexpr/tinyexpr.c)

# Build the core C library from all sources except main.c
add_library(navier_core STATIC ${SRCS})
target_include_directories(navier_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexpr
)

# Link the math library for functions from <math.h> used by tinyexpr and others
# Also link pthread library for threading support
target_link_libraries(navier_core PUBLIC m Threads::Threads)

# Main executable linked against the core library
add_executable(navier_stokes ${MAIN_SRC})
target_link_libraries(navier_stokes PRIVATE navier_core)

# Collect all test sources
file(GLOB TEST_SOURCES "test/*.cpp")

# If Eigen is disabled, exclude tests that require it
if(NOT ENABLE_EIGEN_TESTS)
    list(FILTER TEST_SOURCES EXCLUDE REGEX "momentum_linear_solver_test\\.cpp$")
    list(FILTER TEST_SOURCES EXCLUDE REGEX "boundaries_test\\.cpp$")
    list(FILTER TEST_SOURCES EXCLUDE REGEX "pressure_linear_solver_test\\.cpp$")
endif()

list(FILTER TEST_SOURCES EXCLUDE REGEX "boundaries_test\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "functions\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "g_field_test\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "manufactered_solution\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "momentum_linear_solver_test\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "pressure_linear_solver_test\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "pressure_test\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "test_utils\\.cpp$")
list(FILTER TEST_SOURCES EXCLUDE REGEX "write_to_file_test\\.cpp$")
# Collect all test target names
set(ALL_TEST_TARGETS "")

# Create a separate executable for each test file
foreach(TEST_SRC ${TEST_SOURCES})
    # Get the filename without path and extension
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    
    # Add to list of all test targets
    list(APPEND ALL_TEST_TARGETS ${TEST_NAME})
    
    # Create executable for this specific test
    add_executable(${TEST_NAME} ${TEST_SRC})
    
    # Link against GoogleTest and core library
    if(ENABLE_EIGEN_TESTS AND ${TEST_NAME} MATCHES "momentum_linear_solver_test|boundaries_test|pressure_linear_solver_test")
        target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main navier_core m Eigen3::Eigen)
    else()
        target_link_libraries(${TEST_NAME} PRIVATE GTest::gtest_main navier_core m)
    endif()
    
    # Register with CTest
    gtest_discover_tests(${TEST_NAME})
endforeach()

# Make 'all' target build everything (main executable + all tests)
add_custom_target(build_everything ALL
    DEPENDS navier_stokes ${ALL_TEST_TARGETS}
    COMMENT "Building main executable and all tests"
)

# Debug build flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -Wall -Wextra -O2")

# Utility target: run valgrind on the main executable
add_custom_target(valgrind
    COMMAND valgrind --leak-check=full --show-leak-kinds=all $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Utility target: run the main executable (assumes already built)
add_custom_target(run-main
    COMMAND $<TARGET_FILE:navier_stokes>
    COMMENT "Running navier_stokes"
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Utility target: clean the entire build directory
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build files"
)

# Utility target: run all tests (assumes already built)
add_custom_target(run-all-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure || true
    COMMENT "Running all tests"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Create individual run targets for each test (assumes already built)
foreach(TEST_SRC ${TEST_SOURCES})
    get_filename_component(TEST_NAME ${TEST_SRC} NAME_WE)
    add_custom_target(run-${TEST_NAME}
        COMMAND $<TARGET_FILE:${TEST_NAME}>
        COMMENT "Running ${TEST_NAME}"
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    )
endforeach()

# ==================== Test Manufactured Solution (C) ====================
add_executable(test_manufactured
    test/C_test/test_manufactured.c
    test/C_test/test_common.c
)

target_link_libraries(test_manufactured
    navier_core
    m
)

target_include_directories(test_manufactured PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/tinyexpr
)

add_test(NAME ManufacturedSolutionTest COMMAND test_manufactured)

# ==================== Test Paper Manufactured Solution (C) ====================
add_executable(test_paper_manufactured
    test/C_test/test_paper_manufactured.c
    test/C_test/test_common.c
)

target_link_libraries(test_paper_manufactured
    navier_core
    m
)

target_include_directories(test_paper_manufactured PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/tinyexpr
)

add_test(NAME PaperManufacturedSolutionTest COMMAND test_paper_manufactured)

# ==================== Test Zero Pressure Paper Manufactured Solution (C) ====================
add_executable(test_zero_pressure_manufactured
    test/C_test/test_zero_pressure_manufactured.c
    test/C_test/test_common.c
)

target_link_libraries(test_zero_pressure_manufactured
    navier_core
    m
)

target_include_directories(test_zero_pressure_manufactured PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/tinyexpr
)

add_test(NAME ZeroPressureManufacturedSolutionTest COMMAND test_zero_pressure_manufactured)

# ==================== Test Convergence (C) ====================
add_executable(test_convergence
    test/C_test/test_convergence.c
    test/C_test/test_common.c
)

target_link_libraries(test_convergence
    navier_core
    m
)

target_include_directories(test_convergence PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/tinyexpr
)

add_test(NAME ConvergenceTest COMMAND test_convergence)

# ==================== Test Init Velocity (C) ====================
add_executable(test_init_velocity
    test/C_test/test_init_velocity.c
    test/C_test/test_common.c
)

target_link_libraries(test_init_velocity
    navier_core
    m
)

target_include_directories(test_init_velocity PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/third_party/tinyexpr
)

add_test(NAME InitVelocityTest COMMAND test_init_velocity)

# ==================== CONFIGURABLE CONSTANTS ====================
option(GRID_WIDTH "Grid width" "")
option(GRID_HEIGHT "Grid height" "")
option(GRID_DEPTH "Grid depth" "")
option(TIME_STEP "Time step DT" "")
option(SIM_TOTAL_TIME "Total simulation time" "")

set(OVERRIDE_DEFINES "")
if(GRID_WIDTH)
    set(OVERRIDE_DEFINES "${OVERRIDE_DEFINES} -DWIDTH_OVERRIDE=${GRID_WIDTH}")
endif()
if(GRID_HEIGHT)
    set(OVERRIDE_DEFINES "${OVERRIDE_DEFINES} -DHEIGHT_OVERRIDE=${GRID_HEIGHT}")
endif()
if(GRID_DEPTH)
    set(OVERRIDE_DEFINES "${OVERRIDE_DEFINES} -DDEPTH_OVERRIDE=${GRID_DEPTH}")
endif()
if(TIME_STEP)
    set(OVERRIDE_DEFINES "${OVERRIDE_DEFINES} -DDT_OVERRIDE=${TIME_STEP}")
endif()
if(SIM_TOTAL_TIME)
    set(OVERRIDE_DEFINES "${OVERRIDE_DEFINES} -DTOTAL_TIME_OVERRIDE=${SIM_TOTAL_TIME}")
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OVERRIDE_DEFINES}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OVERRIDE_DEFINES}")