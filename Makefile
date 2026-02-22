CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma -flto -g -Wno-stringop-truncation

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

FTYPE ?= DOUBLE
VEC ?= EXPL

DEFINE = -D$(FTYPE) -D$(VEC) -DTIMEITALL
INCLUDE = -I$(INC_DIR) -I$(SRC_DIR)
LIBS = -lm

SOLVER_OBJS = solver.o momentum.o pressure.o output.o thread-array.o
UNIT_TEST_OBJS = unit-test.o momentum-test.o pressure-test.o
CONVERGENCE_TEST_OBJS = $(SOLVER_OBJS) convergence-test.o

solver: mkdir-build $(BUILD_DIR)/solver
#tests: mkdir-build $(BUILD_DIR)/unit-test $(BUILD_DIR)/convergence-test $(BUILD_DIR)/convergence-pressure-test
tests: mkdir-build $(BUILD_DIR)/convergence-test

mkdir-build:
	mkdir -p $(BUILD_DIR)/objs

$(BUILD_DIR)/solver: $(addprefix $(BUILD_DIR)/objs/, $(SOLVER_OBJS) main.o)
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/unit-test: $(addprefix $(BUILD_DIR)/objs/, $(UNIT_TEST_OBJS))
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/convergence-test: $(addprefix $(BUILD_DIR)/objs/, $(CONVERGENCE_TEST_OBJS))
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/convergence-pressure-test: $(BUILD_DIR)/objs/convergence-pressure-test.o
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/objs/%.o: $(SRC_DIR)/%.c
	$(CC) -c $^ $(CFLAGS) $(INCLUDE) $(DEFINE) -o $@

$(BUILD_DIR)/objs/%.o: tests/%.c
	$(CC) -c $^ $(CFLAGS) $(INCLUDE) $(DEFINE) -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
