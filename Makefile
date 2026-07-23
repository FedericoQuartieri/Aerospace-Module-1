CC = cc
CFLAGS = -std=c11 -O3 -Wall -Wextra -Iinclude
TARGET = solver
SOURCES = $(wildcard src/*.c)

$(TARGET): $(SOURCES)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) -lm

clean:
	rm -f $(TARGET)

.PHONY: clean
