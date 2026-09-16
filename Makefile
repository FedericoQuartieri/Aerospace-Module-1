CC = cc
CFLAGS = -std=gnu11 -O3 -Wall -Wextra
CPPFLAGS =
SIMD ?= 0
MPI ?= 0
OMP ?= 0
TRIDIAG ?= schur
OMP_SPLIT ?= auto
# Empty, the default: the pipeline picks its batch at start-up from the threads
# per process (src/tridiag/pipeline/backend.c), and `pipeline_batch_lines = N'
# in the configuration file overrides it for one run.  A number fixes it in the
# binary, which is what the batch sweep of the study and check_pipeline.sh need.
PIPELINE_BATCH_LINES ?=
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
ifneq ($(words $(TRIDIAG)),1)
$(error TRIDIAG must contain exactly one value)
endif
ifeq ($(filter $(TRIDIAG),schur pipeline),)
$(error TRIDIAG must be schur or pipeline, not '$(TRIDIAG)')
endif

override CPPFLAGS += -Iinclude

ifeq ($(TRIDIAG),schur)
override CPPFLAGS += -DTRIDIAG_SCHUR
endif

ifeq ($(TRIDIAG),pipeline)
override CPPFLAGS += -DTRIDIAG_PIPELINE
ifneq ($(PIPELINE_BATCH_LINES),)
ifeq ($(shell expr "$(PIPELINE_BATCH_LINES)" : '[1-9][0-9]*$$'),0)
$(error PIPELINE_BATCH_LINES must be one positive integer, not '$(PIPELINE_BATCH_LINES)')
endif
override CPPFLAGS += -DPIPELINE_BATCH_LINES=$(PIPELINE_BATCH_LINES)
endif
endif

# The backend's own headers are private to its directory: include/ holds only
# what the shared solver is allowed to know.
override CPPFLAGS += -Isrc/tridiag/$(TRIDIAG)

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
override CPPFLAGS += -DUSE_MPI
endif

# Build with OMP=1 to spread the independent lines over the cores of one
# machine.  It composes with MPI=1: the processes divide the domain, the
# threads divide the lines of each block.
ifeq ($(OMP),1)
CFLAGS += -fopenmp
override CPPFLAGS += -DUSE_OMP
endif

# Benchmark-only override for comparing the two OpenMP work-sharing layouts.
# The forced builds deliberately exclude MPI and SIMD so both executables run
# the same scalar kernels over the same complete, local lines.  Only the schur
# backend reads the policy: the pipeline one has its own loop structure, so a
# forced split there would compile and measure nothing.
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
ifneq ($(TRIDIAG),schur)
$(error OMP_SPLIT=$(OMP_SPLIT) requires TRIDIAG=schur)
endif
endif

ifeq ($(OMP_SPLIT),planes)
override CPPFLAGS += -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_PLANES
endif
ifeq ($(OMP_SPLIT),lines)
override CPPFLAGS += -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_LINES
endif
ifeq ($(OMP_SPLIT),serial)
override CPPFLAGS += -DWORKERS_LINE_POLICY=WORKERS_LINE_POLICY_SERIAL
endif

ifeq ($(SIMD),1)
override CPPFLAGS += -DUSE_SIMD \
	-DZETA_SIMD_VECTORS=$(ZETA_SIMD_VECTORS) \
	-DU_SIMD_VECTORS=$(U_SIMD_VECTORS)
HOST_ARCH := $(shell uname -m)
ifneq ($(filter x86_64 amd64,$(HOST_ARCH)),)
CFLAGS += -mavx2
endif
endif

CFLAGS += $(EXTRA_CFLAGS)
override CPPFLAGS += $(EXTRA_CPPFLAGS)

# Configuration-specific outputs prevent stale binaries after changing flags.
# Quote each value before passing it to the shell (flags can contain quotes).
quote = '$(subst ','"'"',$(1))'
COMPILER_VERSION := $(shell $(CC) --version 2>/dev/null | head -n 1)
BUILD_ID := $(shell printf '%s\n' $(call quote,$(CC)) $(call quote,$(COMPILER_VERSION)) $(call quote,$(CPPFLAGS)) $(call quote,$(CFLAGS)) | cksum | awk '{print $$1 "-" $$2}')
BUILD_ROOT ?= build/variants
BUILD_DIR := $(BUILD_ROOT)/$(BUILD_ID)
.DEFAULT_GOAL := solver
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
TEST_BIN_DIR = $(BUILD_DIR)/tests
# Tests that exercise the backend directly live with it, so selecting a
# backend selects its tests too and never tries to link the other one's.
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c) \
	$(wildcard $(TEST_DIR)/tridiag/$(TRIDIAG)/*.c)
TEST_HEADERS = $(wildcard $(TEST_DIR)/*.h)
CORE_SOURCES = $(filter-out src/main.c,$(SOURCES))
CORE_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/obj/%.o,$(CORE_SOURCES))
SOLVER_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/obj/%.o,$(SOURCES))
CHANNEL_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/channel/%.o,$(CORE_SOURCES))
BRINKMAN_OBJECTS = $(patsubst %.c,$(BUILD_DIR)/brinkman/%.o,$(CORE_SOURCES))
TEST_TARGETS = $(patsubst %.c,$(TEST_BIN_DIR)/%,$(notdir $(TEST_SOURCES)))
CHANNEL_CPPFLAGS = -DDEFAULT_LX=2.0 -DDEFAULT_LY=1.0 -DDEFAULT_LZ=1.0 \
	-DDEFAULT_WIDTH=192 -DDEFAULT_HEIGHT=96 -DDEFAULT_DEPTH=96
# A wide, flat channel: the flow depends on y only, and cells far apart along
# X and Z keep the exact profile on those faces from reaching the measurement.
BRINKMAN_CPPFLAGS = -DDEFAULT_LX=20.0 -DDEFAULT_LY=1.0 -DDEFAULT_LZ=20.0 \
	-DDEFAULT_WIDTH=6 -DDEFAULT_HEIGHT=64 -DDEFAULT_DEPTH=6

# Compile to a unique temporary file and publish atomically. Independent
# study jobs can build even the same variant without sharing partial output.
define compile
	@mkdir -p "$(@D)"
	@set -e; tmp=$$(mktemp "$@.XXXXXX"); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) $(1) -o "$$tmp" -lm; \
	chmod +x "$$tmp"; mv -f "$$tmp" "$@"
endef

define publish
	@mkdir -p "$(@D)"
	@set -e; tmp=$$(mktemp "$@.XXXXXX"); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	cp "$<" "$$tmp"; chmod +x "$$tmp"; mv -f "$$tmp" "$@"
endef

define compile_object
	@mkdir -p "$(@D)"
	@set -e; tmp=$$(mktemp "$@.XXXXXX"); trap 'rm -f "$$tmp"' EXIT HUP INT TERM; \
	$(CC) $(CPPFLAGS) $(CFLAGS) -c "$<" -o "$$tmp"; mv -f "$$tmp" "$@"
endef

$(BUILD_DIR)/obj/%.o: %.c $(HEADERS) Makefile
	$(compile_object)

# These tests change solver defaults, so their core objects must also carry
# those defaults. They cannot reuse the ordinary core object directory.
$(CHANNEL_OBJECTS): override CPPFLAGS += $(CHANNEL_CPPFLAGS)
$(BUILD_DIR)/channel/%.o: %.c $(HEADERS) Makefile
	$(compile_object)

$(BRINKMAN_OBJECTS): override CPPFLAGS += $(BRINKMAN_CPPFLAGS)
$(BUILD_DIR)/brinkman/%.o: %.c $(HEADERS) Makefile
	$(compile_object)

$(BUILD_DIR)/solver: $(SOLVER_OBJECTS) Makefile
	$(call compile,$(SOLVER_OBJECTS))

# These familiar paths are convenience copies. Studies and checks use the
# immutable variant path directly, so concurrent configurations cannot race.
$(TARGET): $(BUILD_DIR)/solver FORCE
	$(publish)

build/tests/%: $(TEST_BIN_DIR)/% FORCE
	$(publish)

tests: $(addprefix build/tests/,$(notdir $(TEST_TARGETS)))

test: tests

check:
	./scripts/check_pipeline.sh

$(TEST_BIN_DIR)/channel_obstacle: $(TEST_DIR)/channel_obstacle.c $(CHANNEL_OBJECTS) $(HEADERS) $(TEST_HEADERS) Makefile
	$(call compile,$(CHANNEL_CPPFLAGS) -Itest $< $(CHANNEL_OBJECTS))
$(TEST_BIN_DIR)/moving_sphere: $(TEST_DIR)/moving_sphere.c $(CHANNEL_OBJECTS) $(HEADERS) $(TEST_HEADERS) Makefile
	$(call compile,$(CHANNEL_CPPFLAGS) -Itest $< $(CHANNEL_OBJECTS))
$(TEST_BIN_DIR)/brinkman_channel: $(TEST_DIR)/brinkman_channel.c $(BRINKMAN_OBJECTS) $(HEADERS) $(TEST_HEADERS) Makefile
	$(call compile,$(BRINKMAN_CPPFLAGS) -Itest $< $(BRINKMAN_OBJECTS))

$(TEST_BIN_DIR)/%: $(TEST_DIR)/%.c $(CORE_OBJECTS) $(HEADERS) $(TEST_HEADERS) Makefile
	$(call compile,-Itest $< $(CORE_OBJECTS))

# The same recipe for the tests that live with their backend.  Two rules
# rather than a vpath: make picks whichever prerequisite actually exists, and
# a missing test fails loudly instead of being silently searched for elsewhere.
$(TEST_BIN_DIR)/%: $(TEST_DIR)/tridiag/$(TRIDIAG)/%.c $(CORE_OBJECTS) $(HEADERS) $(TEST_HEADERS) Makefile
	$(call compile,-Itest $< $(CORE_OBJECTS))

print-test-dir:
	@printf '%s\n' '$(TEST_BIN_DIR)'

print-build-id:
	@printf '%s\n' '$(BUILD_ID)'

FORCE:

clean:
	rm -f solver
	rm -rf build/tests $(BUILD_ROOT)

.SECONDARY:
.PHONY: solver check clean test tests print-test-dir print-build-id FORCE
