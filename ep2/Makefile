CC = gcc
NVCC = nvcc
CUDAPATH = /usr/local/cuda

CFLAGS = -Wall -Wextra -O2
NVCCFLAGS = -I$(CUDAPATH)/include 

TARGET = heat
CU_EXECUTABLES = $(TARGET)_parallel $(TARGET)_body
C_EXECUTABLE = $(TARGET)

all: $(C_EXECUTABLE) $(CU_EXECUTABLES)

$(C_EXECUTABLE): $(TARGET).o
	$(CC) $(CFLAGS) -o $@ $^

$(TARGET).o: $(TARGET).c
	$(CC) $(CFLAGS) -c $< -o $@

%: %.cu
	$(NVCC) $(NVCCFLAGS) $< -o $@

clean:
	rm -f $(TARGET).o $(C_EXECUTABLE) $(CU_EXECUTABLES)
