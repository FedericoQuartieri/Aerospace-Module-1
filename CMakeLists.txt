cmake_minimum_required(VERSION 3.15)
project(dual_build LANGUAGES C CXX) 

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- test (googletest) ---
enable_testing()
add_subdirectory(third_party/googletest)

add_executable(sample_test tests/sample_test.cpp)
target_link_libraries(sample_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sample_test)

# --- main ---

# Warning -Wall
add_library(common_warnings INTERFACE)
target_compile_options(common_warnings INTERFACE
  $<$<COMPILE_LANGUAGE:C>:-Wall>
  $<$<COMPILE_LANGUAGE:CXX>:-Wall>
)

# double (no define)
add_executable(main_double main.c helpers.c g.c)
target_link_libraries(main_double PRIVATE common_warnings)
set_target_properties(main_double PROPERTIES OUTPUT_NAME "main-double")

# float (-DFLOAT)
add_executable(main_float main.c helpers.c g.c)
target_link_libraries(main_float PRIVATE common_warnings)
target_compile_definitions(main_float PRIVATE FLOAT)
set_target_properties(main_float PROPERTIES OUTPUT_NAME "main-float")

# Note: compile helpers.c and g.c into each executable so they pick up per-target
# compile definitions (e.g. -DFLOAT). Building a single static library caused
# helpers/g to be compiled once (with double) and linked into the float build,
# producing runtime crashes.
