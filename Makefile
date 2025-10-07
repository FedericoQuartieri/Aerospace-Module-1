CC = gcc
CFLAGS = -Wall

all: float double

double: finite-diff.c
	$(CC) $(CFLAGS) -o finite-diff-double $^

float: finite-diff.c
	$(CC) $(CFLAGS) -o finite-diff-float -DFLOAT $^

clean:
	rm finite-diff-double
	rm finite-diff-float
