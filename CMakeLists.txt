cmake_minimum_required(VERSION 3.11)
project(navier_stokes LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option to enable/disable Eigen-dependent tests
option(ENABLE_EIGEN_TESTS "Build tests that depend on Eigen" OFF)

# Find Eigen3 only if we want the tests that use it
if(ENABLE_EIGEN_TESTS)
    find_package(Eigen3 3.3 REQUIRED NO_MODULE)
endif()

# --- Tests (GoogleTest) ---
enable_testing()
add_subdirectory(third_party/googletest)

# Collect all test sources
file(GLOB TEST "test/*.cpp")

# If Eigen is disabled, exclude tests that require it
if(NOT ENABLE_EIGEN_TESTS)
    list(FILTER TEST EXCLUDE REGEX "momentum_linear_solver_test\\.cpp$")
    list(FILTER TEST EXCLUDE REGEX "boundaries_test\\.cpp$")
    list(FILTER TEST EXCLUDE REGEX "pressure_linear_solver_test\\.cpp$")
endif()

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
target_link_libraries(navier_core PUBLIC m)

# Main executable linked against the core library
add_executable(navier_stokes ${MAIN_SRC})
target_link_libraries(navier_stokes PRIVATE navier_core)

# Tests executable: link GoogleTest and the core library
add_executable(tests ${TEST})

if(ENABLE_EIGEN_TESTS)
    target_link_libraries(tests PRIVATE GTest::gtest_main navier_core m Eigen3::Eigen)
else()
    target_link_libraries(tests PRIVATE GTest::gtest_main navier_core m)
endif()

include(GoogleTest)
gtest_discover_tests(tests)

# Debug build flags
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -Wall -Wextra -O2")

# Utility target: run valgrind on the main executable
add_custom_target(valgrind
    COMMAND valgrind --leak-check=full --show-leak-kinds=all $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Utility target: run the main executable
add_custom_target(run-main
    COMMAND $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Utility target: clean the entire build directory
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build files"
)

# Utility target: build and immediately run tests
add_custom_target(run-tests
    DEPENDS tests
    COMMAND $<TARGET_FILE:tests>
    COMMENT "Running tests after build"
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
