CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma -flto -g

SRC_DIR = src
INC_DIR = include
BUILD_DIR = build

FTYPE ?= DOUBLE

DEFINE = -D$(FTYPE)
INCLUDE = -I$(INC_DIR) -I$(SRC_DIR)

SOLVER_OBJS = main.o solver.o momentum.o pressure.o
TESTS_OBJS = tests.o momentum-test.o pressure-test.o

solver: mkdir-build $(BUILD_DIR)/solver
tests: mkdir-build $(BUILD_DIR)/tests

mkdir-build:
	mkdir -p $(BUILD_DIR)/objs

$(BUILD_DIR)/solver: $(addprefix $(BUILD_DIR)/objs/, $(SOLVER_OBJS))
	$(CC) $^ -o $@

$(BUILD_DIR)/tests: $(addprefix $(BUILD_DIR)/objs/, $(TESTS_OBJS))
	$(CC) $^ -o $@

$(BUILD_DIR)/objs/%.o: $(SRC_DIR)/%.c
	$(CC) -c $^ $(CFLAGS) $(INCLUDE) $(DEFINE) -o $@

$(BUILD_DIR)/objs/%.o: tests/%.c
	$(CC) -c $^ $(CFLAGS) $(INCLUDE) $(DEFINE) -o $@

.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)
