CC ?= cc
PRECISION ?= USE_DOUBLE
BUILD_DIR ?= build

CPPFLAGS = -Iinclude -Isrc -Itest -D_POSIX_C_SOURCE=200809L -D$(PRECISION)
CFLAGS ?= -O3
CFLAGS += -std=c99 -Wall -Wextra -Wpedantic
LDLIBS = -lm

COMMON_SOURCES = \
	src/field.c \
	src/grid.c \
	src/solver.c \
	src/momentum.c \
	src/pressure.c \
	src/physics.c \
	src/output.c

TEST_SUPPORT = test/manufactured_cases.c test/test_support.c

STANDARD = $(BUILD_DIR)/solver_standard
OPTIMIZED = $(BUILD_DIR)/solver_optimized
CORRECTNESS_STANDARD = $(BUILD_DIR)/test_correctness_standard
CORRECTNESS_OPTIMIZED = $(BUILD_DIR)/test_correctness_optimized
KERNEL_TEST = $(BUILD_DIR)/test_kernel_equivalence
CONVERGENCE_TEST = $(BUILD_DIR)/test_convergence
OUTPUT_TEST = $(BUILD_DIR)/test_output

all: $(STANDARD) $(OPTIMIZED)

$(BUILD_DIR):
	mkdir -p $@

$(STANDARD): $(COMMON_SOURCES) src/kernels_standard.c src/main.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSOLVER_BACKEND=SOLVER_BACKEND_STANDARD \
		$^ -o $@ $(LDLIBS)

$(OPTIMIZED): $(COMMON_SOURCES) src/kernels_optimized.c src/main.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSOLVER_BACKEND=SOLVER_BACKEND_OPTIMIZED \
		$^ -o $@ $(LDLIBS)

$(CORRECTNESS_STANDARD): $(COMMON_SOURCES) src/kernels_standard.c \
		test/test_correctness.c $(TEST_SUPPORT) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSOLVER_BACKEND=SOLVER_BACKEND_STANDARD \
		$^ -o $@ $(LDLIBS)

$(CORRECTNESS_OPTIMIZED): $(COMMON_SOURCES) src/kernels_optimized.c \
		test/test_correctness.c $(TEST_SUPPORT) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSOLVER_BACKEND=SOLVER_BACKEND_OPTIMIZED \
		$^ -o $@ $(LDLIBS)

$(KERNEL_TEST): src/field.c src/grid.c src/physics.c \
		src/kernels_standard.c src/kernels_optimized.c \
		test/test_kernel_equivalence.c test/manufactured_cases.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) $^ -o $@ $(LDLIBS)

$(CONVERGENCE_TEST): $(COMMON_SOURCES) src/kernels_standard.c \
		test/test_convergence.c $(TEST_SUPPORT) | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSOLVER_BACKEND=SOLVER_BACKEND_STANDARD \
		$^ -o $@ $(LDLIBS)

$(OUTPUT_TEST): $(COMMON_SOURCES) src/kernels_standard.c test/test_output.c | $(BUILD_DIR)
	$(CC) $(CPPFLAGS) $(CFLAGS) -DSOLVER_BACKEND=SOLVER_BACKEND_STANDARD \
		$^ -o $@ $(LDLIBS)

test: $(CORRECTNESS_STANDARD) $(CORRECTNESS_OPTIMIZED) $(KERNEL_TEST) $(OUTPUT_TEST)
	$(CORRECTNESS_STANDARD)
	$(CORRECTNESS_OPTIMIZED)
	$(KERNEL_TEST)
	$(OUTPUT_TEST)

test-convergence: $(CONVERGENCE_TEST)
	$(CONVERGENCE_TEST)

run-standard: $(STANDARD)
	$(STANDARD)

run-optimized: $(OPTIMIZED)
	$(OPTIMIZED)

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all test test-convergence run-standard run-optimized clean
