CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma -flto -g

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

FTYPE ?= DOUBLE
VEC ?= EXPL

DEFINE = -D$(FTYPE) -D$(VEC)
INCLUDE = -I$(INC_DIR) -I$(SRC_DIR) -Ibenchmarks
LIBS = -lm

SOLVER_OBJS = solver.o momentum.o pressure.o
UNIT_TEST_OBJS = unit-test.o momentum-test.o pressure-test.o
CONVERGENCE_TEST_OBJS = $(SOLVER_OBJS) convergence-test.o

solver: mkdir-build $(BUILD_DIR)/solver
tests: mkdir-build $(BUILD_DIR)/unit-test $(BUILD_DIR)/convergence-test $(BUILD_DIR)/convergence-1d-momentum-test $(BUILD_DIR)/convergence-1d-pressure-test

mkdir-build:
	mkdir -p $(BUILD_DIR)/objs

$(BUILD_DIR)/solver: $(addprefix $(BUILD_DIR)/objs/, $(SOLVER_OBJS) main.o)
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/unit-test: $(addprefix $(BUILD_DIR)/objs/, $(UNIT_TEST_OBJS))
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/convergence-test: $(addprefix $(BUILD_DIR)/objs/, $(CONVERGENCE_TEST_OBJS))
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/convergence-1d-momentum-test: $(addprefix $(BUILD_DIR)/objs/, convergence-1d-momentum-test.o)
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/convergence-1d-pressure-test: $(addprefix $(BUILD_DIR)/objs/, convergence-1d-pressure-test.o)
	$(CC) $^ $(LIBS) -o $@

$(BUILD_DIR)/objs/%.o: $(SRC_DIR)/%.c
	$(CC) -c $^ $(CFLAGS) $(INCLUDE) $(DEFINE) -o $@

$(BUILD_DIR)/objs/%.o: tests/%.c
	$(CC) -c $^ $(CFLAGS) $(INCLUDE) $(DEFINE) -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
