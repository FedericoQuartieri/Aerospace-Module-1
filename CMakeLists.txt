cmake_minimum_required(VERSION 3.10)
project(navier_stokes LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- test (googletest) ---
enable_testing()
add_subdirectory(third_party/googletest)

add_executable(sample_test test/sample_test.cpp)
target_link_libraries(sample_test PRIVATE GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(sample_test)

#------------------


include_directories(include)

# Sorgenti principali
file(GLOB SRCS "src/*.c")
add_executable(navier_stokes ${SRCS})
target_link_libraries(navier_stokes m)

# Debug build type
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -Wall -Wextra -O2")

# Target custom per valgrind
add_custom_target(valgrind
    COMMAND valgrind --leak-check=full --show-leak-kinds=all $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Target custom per run
add_custom_target(run
    COMMAND $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)

# Target custom per clean (come nel Makefile)
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build files"
)
