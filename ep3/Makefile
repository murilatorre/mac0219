CC = gcc
CFLAGS = -Wall -Wextra -O2

TARGET = sequential_julia
C_EXECUTABLE = $(TARGET)

all: $(C_EXECUTABLE)

$(C_EXECUTABLE): $(TARGET).c
	$(CC) $(CFLAGS) -o $@ $^ -lm

clean:
	rm -f $(C_EXECUTABLE) julia.bmp
