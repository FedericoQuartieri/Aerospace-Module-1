CC = mpicc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
SIMD ?= 0
PIPELINE_BATCH_LINES ?= 64
CFLAGS += -DPIPELINE_BATCH_LINES=$(PIPELINE_BATCH_LINES)

ifeq ($(SIMD),1)
CFLAGS += -DUSE_SIMD
HOST_ARCH := $(shell uname -m)
ifneq ($(filter x86_64 amd64,$(HOST_ARCH)),)
CFLAGS += -mavx2
endif
endif

TARGET = solver
SOURCES = $(filter-out src/simd_example.c,$(wildcard src/*.c))
HEADERS = $(wildcard include/*.h)

TEST_DIR = test
TEST_BIN_DIR = build/tests
TEST_SOURCES = $(wildcard $(TEST_DIR)/*.c)
TEST_HEADERS = $(wildcard $(TEST_DIR)/*.h)
CORE_SOURCES = $(filter-out src/main.c,$(SOURCES))
TEST_TARGETS = $(patsubst $(TEST_DIR)/%.c,$(TEST_BIN_DIR)/%,$(TEST_SOURCES))
STALE_TEST_TARGETS = $(wildcard $(TEST_BIN_DIR)/*)
SCALING_TARGETS = $(wildcard build/mpi_solver_* build/problem_size_solver_*)
CHANNEL_CFLAGS = -DLX=2.0 -DLY=1.0 -DLZ=1.0 \
	-DWIDTH=192 -DHEIGHT=96 -DDEPTH=96 \
	-DSTEPS=100

$(TARGET): $(SOURCES) $(HEADERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) -lm

tests: $(TEST_TARGETS)

test: tests

$(TEST_BIN_DIR)/channel_obstacle: CFLAGS += $(CHANNEL_CFLAGS)

$(TEST_BIN_DIR)/%: $(TEST_DIR)/%.c $(CORE_SOURCES) $(HEADERS) $(TEST_HEADERS) Makefile
	mkdir -p $(TEST_BIN_DIR)
	$(CC) $(CFLAGS) $< $(CORE_SOURCES) -o $@ -lm

clean:
	rm -f $(TARGET) $(TEST_TARGETS) $(STALE_TEST_TARGETS) $(SCALING_TARGETS)

.PHONY: clean test tests
