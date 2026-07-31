CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
TARGET = solver
SOURCES = $(wildcard src/*.c)
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
