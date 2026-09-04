CC = cc
CFLAGS = -std=gnu11 -O3 -Wall -Wextra -Iinclude
SIMD ?= 0
MPI ?= 0
OMP ?= 0
OMP_SPLIT ?= auto
ZETA_SIMD_VECTORS ?= 4
U_SIMD_VECTORS ?= 8

VALID_OMP_SPLITS = auto planes lines serial
ifneq ($(words $(OMP_SPLIT)),1)
$(error OMP_SPLIT must contain exactly one value)
endif
ifeq ($(filter $(OMP_SPLIT),$(VALID_OMP_SPLITS)),)
$(error OMP_SPLIT must be one of: $(VALID_OMP_SPLITS))
endif

# Build with MPI=1 to compile against MPI and run with mpirun.
ifeq ($(MPI),1)
CC = mpicc
CFLAGS += -DUSE_MPI
endif

# Build with OMP=1 to spread the independent lines over the cores of one
# machine.  It composes with MPI=1: the processes divide the domain, the
# threads divide the lines of each block.
ifeq ($(OMP),1)
CFLAGS += -fopenmp -DUSE_OMP
endif

# Benchmark-only override for comparing the two OpenMP work-sharing layouts.
# The forced builds deliberately exclude MPI and SIMD so both executables run
# the same scalar kernels over the same complete, local lines.
ifneq ($(OMP_SPLIT),auto)
ifneq ($(OMP),1)
$(error OMP_SPLIT=$(OMP_SPLIT) requires OMP=1)
endif
ifneq ($(MPI),0)
$(error OMP_SPLIT=$(OMP_SPLIT) requires MPI=0)
endif
ifneq ($(SIMD),0)
$(error OMP_SPLIT=$(OMP_SPLIT) requires SIMD=0)
endif
endif

ifeq ($(OMP_SPLIT),planes)
CFLAGS += -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_PLANES
endif
ifeq ($(OMP_SPLIT),lines)
CFLAGS += -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_LINES
endif
ifeq ($(OMP_SPLIT),serial)
CFLAGS += -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_SERIAL
endif

ifeq ($(SIMD),1)
CFLAGS += -DUSE_SIMD \
	-DZETA_SIMD_VECTORS=$(ZETA_SIMD_VECTORS) \
	-DU_SIMD_VECTORS=$(U_SIMD_VECTORS)
HOST_ARCH := $(shell uname -m)
ifneq ($(filter x86_64 amd64,$(HOST_ARCH)),)
CFLAGS += -mavx2
endif
endif

TARGET = solver
# simd_example.c documents the previous prototype and is not part of the solver.
SOURCES = $(filter-out src/simd_example.c,$(wildcard src/*.c))
HEADERS = $(wildcard include/*.h)

TEST_DIR = test
TEST_BIN_DIR = build/tests
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_HEADERS = $(wildcard $(TEST_DIR)/*.h)
CORE_SOURCES = $(filter-out src/main.c,$(SOURCES))
TEST_TARGETS = $(patsubst $(TEST_DIR)/%.c,$(TEST_BIN_DIR)/%,$(TEST_SOURCES))
CHANNEL_CFLAGS = -DDEFAULT_LX=2.0 -DDEFAULT_LY=1.0 -DDEFAULT_LZ=1.0 \
	-DDEFAULT_WIDTH=192 -DDEFAULT_HEIGHT=96 -DDEFAULT_DEPTH=96

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) -lm

tests: $(TEST_TARGETS)

test: tests

$(TEST_BIN_DIR)/channel_obstacle $(TEST_BIN_DIR)/moving_sphere: CFLAGS += $(CHANNEL_CFLAGS)

$(TEST_BIN_DIR)/%: $(TEST_DIR)/%.c $(CORE_SOURCES) $(HEADERS) $(TEST_HEADERS) Makefile
	mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) $< $(CORE_SOURCES) -o $@ -lm

clean:
	rm -f $(TARGET) $(TEST_TARGETS)

.PHONY: clean test tests
