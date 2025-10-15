# Compilatore e flag
CC = gcc
CFLAGS = -Wall -Wextra -O2 -MMD -MP -Iinclude
LDFLAGS = -lm

# Directory
SRCDIR = src
TESTDIR = test
BUILDDIR = build
TARGET = $(BUILDDIR)/navier_stokes
TESTTARGET = $(BUILDDIR)/tests

# File sorgenti e oggetti
SRCS = $(wildcard $(SRCDIR)/*.c)
TESTS = $(wildcard $(TESTDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))
TESTOBJS = $(patsubst $(TESTDIR)/%.c,$(BUILDDIR)/test_%.o,$(TESTS))

# Dipendenze generate automaticamente
DEPS = $(OBJS:.o=.d) $(TESTOBJS:.o=.d)

# Regola di default
all: $(TARGET)

# Link finale programma principale
$(TARGET): $(OBJS)
	@echo [LD] $@
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Link finale test
$(TESTTARGET): $(TESTOBJS) $(filter-out $(BUILDDIR)/main.o,$(OBJS))
	@echo [LD] $@
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Regole di compilazione generiche
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	@echo [CC] $<
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILDDIR)/test_%.o: $(TESTDIR)/%.c | $(BUILDDIR)
	@echo [CC] $<
	$(CC) $(CFLAGS) -c $< -o $@

# Includi le dipendenze
-include $(DEPS)

# Creazione directory build
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Pulizia
clean:
	rm -rf $(BUILDDIR)

# Debug mode
debug: CFLAGS += -g -DDEBUG
debug: clean all

# Esegui programma principale
run: all
	@echo "=== Running $(TARGET) ==="
	@./$(TARGET)

# Esegui i test
test: $(TESTTARGET)
	@echo "=== Running tests ==="
	@./$(TESTTARGET)

# Analisi con Valgrind
valgrind: debug
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

.PHONY: all clean debug run test valgrind
