cmake_minimum_required(VERSION 3.11)
project(navier_stokes LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- test (googletest) ---
enable_testing()
add_subdirectory(third_party/googletest)

file(GLOB TEST "test/*.cpp")
# include(GoogleTest) later after target
#------------------

include_directories(include)

# Sorgenti principali (escludi main.c per poter creare una libreria riusabile)
file(GLOB SRCS "src/*.c")
set(MAIN_SRC "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")
list(REMOVE_ITEM SRCS ${MAIN_SRC})

# Compila le sorgenti C in una libreria statica
add_library(navier_core STATIC ${SRCS})
target_include_directories(navier_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(navier_core PUBLIC m)

# Eseguibile principale linkato alla libreria
add_executable(navier_stokes ${MAIN_SRC})
target_link_libraries(navier_stokes PRIVATE navier_core)

# Test: linka googletest e la libreria core
add_executable(tests ${TEST})
target_link_libraries(tests PRIVATE GTest::gtest_main navier_core m)

include(GoogleTest)
gtest_discover_tests(tests)

# Debug build type
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -DDEBUG -Wall -Wextra -O2")

# Targets utility...
add_custom_target(valgrind
    COMMAND valgrind --leak-check=full --show-leak-kinds=all $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(run
    COMMAND $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
)
add_custom_target(clean-all
    COMMAND ${CMAKE_COMMAND} -E rm -rf ${CMAKE_BINARY_DIR}/*
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Cleaning all build files"
)
