CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma $(FLAGS)

all: benchmark test

benchmark: finite-diff.c lin-solver.c benchmark.c
	$(CC) $(CFLAGS) -o benchmark-double $^
	$(CC) $(CFLAGS) -o benchmark-float -DFLOAT $^

test: finite-diff.c lin-solver.c test.c
	$(CC) $(CFLAGS) -o test-double $^
	$(CC) $(CFLAGS) -o test-float -DFLOAT $^

clean:
	rm benchmark-float
	rm benchmark-double
	rm test-float
	rm test-double
