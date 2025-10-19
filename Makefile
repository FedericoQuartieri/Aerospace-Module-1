CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma $(FLAGS)

all: finite-diff lin-solver

finite-diff: finite-diff.c
	$(CC) $(CFLAGS) -o finite-diff-double $^
	$(CC) $(CFLAGS) -o finite-diff-float -DFLOAT $^

lin-solver: lin-solver.c main.c
	$(CC) $(CFLAGS) -o lin-solver-double $^
	$(CC) $(CFLAGS) -o lin-solver-float -DFLOAT $^

clean:
	rm finite-diff-double
	rm finite-diff-float
	rm lin-solver-float
	rm lin-solver-double
