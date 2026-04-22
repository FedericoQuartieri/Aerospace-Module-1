CC = clang

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

PRECISION ?= USE_DOUBLE
PRECISION_CFLAGS = -D$(PRECISION)

ARCH_CFLAGS =
ifeq ($(UNAME_S),Linux)
ifneq (,$(filter $(UNAME_M),aarch64 arm64))
ARCH_CFLAGS += -march=armv8-a+simd
endif
endif

COMMON_CFLAGS = -Wall -Wextra -Iinclude $(PRECISION_CFLAGS) $(ARCH_CFLAGS)
RELEASE_CFLAGS = -O3
PROFILE_CFLAGS = -O3 -g -fno-omit-frame-pointer
DEPFLAGS = -MMD -MP
LDFLAGS = -lm

BUILDDIR = build
PROFILEDIR = build_profile

TARGET = $(BUILDDIR)/navier_stokes
PROFILE_TARGET = $(PROFILEDIR)/navier_stokes

SRCS = $(wildcard src/*.c)
OBJS = $(patsubst src/%.c,$(BUILDDIR)/%.o,$(SRCS))
PROFILE_OBJS = $(patsubst src/%.c,$(PROFILEDIR)/%.o,$(SRCS))

DEPS = $(OBJS:.o=.d)
PROFILE_DEPS = $(PROFILE_OBJS:.o=.d)

LIB_OBJS = $(filter-out $(BUILDDIR)/main.o,$(OBJS))
PROFILE_LIB_OBJS = $(filter-out $(PROFILEDIR)/main.o,$(PROFILE_OBJS))

TEST_COMMON = test/C_test/test_common.c
TESTS = \
	test_convergence \
	test_manufactured \
	test_paper_manufactured \
	test_zero_pressure_manufactured \
	test_tridiagonal \
	test_force_field \
	test_boundary_conditions \

TEST_BINS = $(addprefix $(BUILDDIR)/,$(TESTS))
PROFILE_TEST_BINS = $(addprefix $(PROFILEDIR)/,$(TESTS))

all: $(TARGET)
profile: $(PROFILE_TARGET)

$(TARGET): $(OBJS) | $(BUILDDIR)
	$(CC) $(COMMON_CFLAGS) $(RELEASE_CFLAGS) -o $@ $^ $(LDFLAGS)

$(PROFILE_TARGET): $(PROFILE_OBJS) | $(PROFILEDIR)
	$(CC) $(COMMON_CFLAGS) $(PROFILE_CFLAGS) -o $@ $^ $(LDFLAGS)

$(BUILDDIR)/%.o: src/%.c | $(BUILDDIR)
	$(CC) $(COMMON_CFLAGS) $(DEPFLAGS) $(RELEASE_CFLAGS) -c $< -o $@

$(PROFILEDIR)/%.o: src/%.c | $(PROFILEDIR)
	$(CC) $(COMMON_CFLAGS) $(DEPFLAGS) $(PROFILE_CFLAGS) -c $< -o $@

$(BUILDDIR)/test_%: test/C_test/test_%.c $(TEST_COMMON) $(LIB_OBJS) | $(BUILDDIR)
	$(CC) $(COMMON_CFLAGS) $(RELEASE_CFLAGS) -o $@ $^ $(LDFLAGS)

$(PROFILEDIR)/test_%: test/C_test/test_%.c $(TEST_COMMON) $(PROFILE_LIB_OBJS) | $(PROFILEDIR)
	$(CC) $(COMMON_CFLAGS) $(PROFILE_CFLAGS) -o $@ $^ $(LDFLAGS)

run: $(TARGET)
	./$(TARGET)

run-profile: $(PROFILE_TARGET)
	./$(PROFILE_TARGET)

test: $(TEST_BINS)
	@for t in $(TESTS); do \
		(cd test/C_test && ../../$(BUILDDIR)/$$t) || exit 1; \
	done

test-profile: $(PROFILE_TEST_BINS)
	@for t in $(TESTS); do \
		(cd test/C_test && ../../$(PROFILEDIR)/$$t) || exit 1; \
	done

run-test_%: $(BUILDDIR)/test_%
	cd test/C_test && ../../$(BUILDDIR)/test_$*

run-profile-test_%: $(PROFILEDIR)/test_%
	cd test/C_test && ../../$(PROFILEDIR)/test_$*

$(BUILDDIR):
	mkdir -p $@

$(PROFILEDIR):
	mkdir -p $@

clean:
	rm -rf $(BUILDDIR) $(PROFILEDIR)

.PHONY: all profile run run-profile test test-profile clean run-test_% run-profile-test_%

-include $(DEPS) $(PROFILE_DEPS)
