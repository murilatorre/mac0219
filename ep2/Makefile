TARGET = heat
CC = gcc
CFLAGS = -Wall -Wextra -O2

all: $(TARGET)

$(TARGET): heat.o
	$(CC) $(CFLAGS) -o $(TARGET) heat.o

heat.o: heat.cu
	$(CC) $(CFLAGS) -c heat.cu

clean:
	rm -f $(TARGET) heat.o
