CC = gcc
CFLAGS = -Wall

all: float double

double: main.c
	$(CC) $(CFLAGS) -o main-double $^

float: main.c
	$(CC) $(CFLAGS) -o main-float -DFLOAT $^

clean:
	rm main-double
	rm main-float