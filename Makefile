CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma

DEFINES =

.PHONY: clean

all: benchmark test

float: DEFINES += -DFLOAT
float: all

auto-vec: DEFINES += -DAUTO_VEC
auto-vec: all

benchmark: finite-diff.c lin-solver.c benchmark.c
	$(CC) $(CFLAGS) -o benchmark $(DEFINES) $^

test: finite-diff.c lin-solver.c test.c
	$(CC) $(CFLAGS) -o test $(DEFINES) $^

clean:
	rm -f benchmark
	rm -f test
