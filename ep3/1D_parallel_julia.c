#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h> // (Message Passing Interface)

int main(int argc, char *argv[])  {
    int numtasks, rank;
    int n, height, width;
    float tint;
    
    // define dimensões e tint da imagem
    n = atoi(argv[1]);
    height = n;
    width = 2*n;
    tint = 1.0;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtém o rank do processo atual
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); // Obtém o número de processos no comunicador


    int rowsPerTask = height / numtasks, remainder = height % numtasks;
    if (rank != -1) {
        int start = rowsPerTask * rank  + (rank - 1 < remainder ? rank : remainder);
        int end   = start + rowsPerTask + (rank < remainder) - 1;
        printf("[Process %d out of %d]: I should compute pixel rows %d to %d, for a total of %d rows\n", rank, numtasks, start, end, (end-start + 1));
        
    } else
        printf("Must specify %d processors. Terminating.\n",SIZE);

    MPI_Finalize();
}