CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
SIMD ?= 0
ZETA_SIMD_VECTORS ?= 4
U_SIMD_VECTORS ?= 8

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
CHANNEL_CFLAGS = -DLX=2.0 -DLY=1.0 -DLZ=1.0 \
	-DWIDTH=96 -DHEIGHT=48 -DDEPTH=48

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
