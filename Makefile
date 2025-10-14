# Compilatore e flag
CC = gcc
CFLAGS = -Wall -Wextra -O2
LDFLAGS = -lm

# Directory e file
SRCDIR = .
BUILDDIR = build
TARGET = $(BUILDDIR)/navier_stokes

# File sorgenti e oggetti
SRCS = $(wildcard $(SRCDIR)/*.c)
OBJS = $(patsubst $(SRCDIR)/%.c,$(BUILDDIR)/%.o,$(SRCS))

# Regola di default
all: $(TARGET)

# Link finale
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)

# Compilazione in build/
$(BUILDDIR)/%.o: $(SRCDIR)/%.c | $(BUILDDIR)
	$(CC) $(CFLAGS) -c $< -o $@

# Creazione cartella build se non esiste
$(BUILDDIR):
	mkdir -p $(BUILDDIR)

# Pulizia
clean:
	rm -rf $(BUILDDIR)

# Modalità debug
debug: CFLAGS += -g -DDEBUG
debug: clean all

# Esecuzione del programma
run: all
	@echo "=== Running $(TARGET) ==="
	@./$(TARGET)

# Analisi con Valgrind
valgrind: debug
	valgrind --leak-check=full --show-leak-kinds=all ./$(TARGET)

.PHONY: all clean debug run valgrind
