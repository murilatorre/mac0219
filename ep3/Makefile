CC = gcc
MPICC = mpicc
CFLAGS = -Wall -Wextra -O2

EXECUTABLES = sequential_julia 1D_parallel_julia

all: $(EXECUTABLES)

sequential_julia: sequential_julia.c
	$(CC) $(CFLAGS) -o $@ $^ -lm


1D_parallel_julia: 1D_parallel_julia.c
	$(MPICC) -o $@ $^ -lm

clean:
	rm -f $(EXECUTABLES) julia.bmp parallel_julia.bmp
