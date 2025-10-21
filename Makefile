CC = gcc
CFLAGS = -O3 -Wall -mavx2 -mfma $(FLAGS)

all: benchmark

benchmark: finite-diff.c lin-solver.c main.c
	$(CC) $(CFLAGS) -o benchmark-double $^
	$(CC) $(CFLAGS) -o benchmark-float -DFLOAT $^

clean:
	rm benchmark-float
	rm benchmark-double
