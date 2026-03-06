cmake_minimum_required(VERSION 3.10)
project(navier_stokes LANGUAGES C)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)

# ==================== FLAGS ====================
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_RELEASE "-O2")
set(CMAKE_C_FLAGS_DEBUG "-g -O0 -DDEBUG")

# ==================== CONFIGURABLE CONSTANTS ====================
# These can be overridden via cmake -DGRID_WIDTH=64 etc.
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

# ==================== THIRD PARTY ====================
add_library(tinyexpr STATIC third_party/tinyexpr/tinyexpr.c)
target_include_directories(tinyexpr PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/tinyexpr)

# ==================== LIBRERIA CORE ====================
file(GLOB SRCS "src/*.c")
list(REMOVE_ITEM SRCS "${CMAKE_CURRENT_SOURCE_DIR}/src/main.c")

add_library(navier_core STATIC ${SRCS})
target_include_directories(navier_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(navier_core PUBLIC m tinyexpr)

# ==================== ESEGUIBILE PRINCIPALE ====================
add_executable(navier_stokes src/main.c)
target_link_libraries(navier_stokes PRIVATE navier_core)

# ==================== TEST ====================
enable_testing()

set(TEST_COMMON "${CMAKE_CURRENT_SOURCE_DIR}/test/test_common.c")

set(TEST_NAMES
    test_convergence
    test_init_velocity
    test_manufactured
    test_paper_manufactured
    test_zero_pressure_manufactured
    test_setted_boundaries
)

foreach(TEST_NAME ${TEST_NAMES})
    add_executable(${TEST_NAME} test/${TEST_NAME}.c ${TEST_COMMON})
    target_link_libraries(${TEST_NAME} PRIVATE navier_core)
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/test
    )
endforeach()

# ==================== TARGETS UTILI ====================
add_custom_target(run
    COMMAND $<TARGET_FILE:navier_stokes>
    DEPENDS navier_stokes
)