CC = cc
CFLAGS = -std=gnu11 -O3 -Wall -Wextra -Iinclude
SIMD ?= 0
MPI ?= 0
OMP ?= 0
TRIDIAG ?= schur
PIPELINE_BATCH_LINES ?= 64
ZETA_SIMD_VECTORS ?= 4
U_SIMD_VECTORS ?= 8

# TRIDIAG picks how a grid line split across processes is solved.  It is a
# different question from MPI/OMP/SIMD, which decide whether the line is split
# at all, whether threads help inside a block, and how wide the kernels are:
# the four options compose.
#
#   schur     Schur complement.  Three local Thomas solves per line, one
#             exchange and one collective per group of lines.
#   pipeline  Pipelined Thomas.  One local solve per line, the wait hidden by
#             sending many independent lines through the processes in batches.
#
# The choice is a directory, not a chain of #ifdef: only one backend is ever
# compiled, so the two cannot silently drift into each other.
ifeq ($(filter $(TRIDIAG),schur pipeline),)
$(error TRIDIAG must be schur or pipeline, not '$(TRIDIAG)')
endif

ifeq ($(TRIDIAG),schur)
CFLAGS += -DTRIDIAG_SCHUR
endif

ifeq ($(TRIDIAG),pipeline)
CFLAGS += -DTRIDIAG_PIPELINE -DPIPELINE_BATCH_LINES=$(PIPELINE_BATCH_LINES)
endif

# The backend's own headers are private to its directory: include/ holds only
# what the shared solver is allowed to know.
CFLAGS += -Isrc/tridiag/$(TRIDIAG)

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
# src/*.c is the shared solver; the tridiagonal backend comes from its own
# directory.  src/*.c does not reach into subdirectories, so exactly one
# backend is compiled and never both.
# simd_example.c documents the previous prototype and is not part of the solver.
SOURCES = $(filter-out src/simd_example.c,$(wildcard src/*.c)) \
	$(wildcard src/tridiag/$(TRIDIAG)/*.c)
HEADERS = $(wildcard include/*.h) $(wildcard include/*/*.h) \
	$(wildcard src/tridiag/$(TRIDIAG)/*.h)

TEST_DIR = test
TEST_BIN_DIR = build/tests
# Tests that exercise the backend directly live with it, so selecting a
# backend selects its tests too and never tries to link the other one's.
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c) \
	$(wildcard $(TEST_DIR)/tridiag/$(TRIDIAG)/*.c)
TEST_HEADERS = $(wildcard $(TEST_DIR)/*.h)
CORE_SOURCES = $(filter-out src/main.c,$(SOURCES))
TEST_TARGETS = $(patsubst %.c,$(TEST_BIN_DIR)/%,$(notdir $(TEST_SOURCES)))
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

# The same recipe for the tests that live with their backend.  Two rules
# rather than a vpath: make picks whichever prerequisite actually exists, and
# a missing test fails loudly instead of being silently searched for elsewhere.
$(TEST_BIN_DIR)/%: $(TEST_DIR)/tridiag/$(TRIDIAG)/%.c $(CORE_SOURCES) $(HEADERS) $(TEST_HEADERS) Makefile
	mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) $< $(CORE_SOURCES) -o $@ -lm

clean:
	rm -f $(TARGET) $(TEST_TARGETS)

.PHONY: clean test tests
