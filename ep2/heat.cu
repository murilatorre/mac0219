#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define WALL_TEMP 20.0
#define FIREPLACE_TEMP 100.0

#define FIREPLACE_START 3
#define FIREPLACE_END 7
#define ROOM_SIZE 10


__global__ void jacobi_iteration(double *h, double *g, int n, int iter_limit)
{
    for (int iter = 0; iter < iter_limit; iter++) {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                g[i*n + j] = 0.25 * (h[(i-1)*n + j] + h[(i+1)*n + j] + h[i*n + (j-1)] + h[i*n + (j+1)]);
            
    
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                h[i*n + j] = g[i*n + j];
    }
}

void c_jacobi_iteration(double *h, double *g, int n, int iter_limit)
{
    for (int iter = 0; iter < iter_limit; iter++) {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                g[i*n + j] = 0.25 * (h[(i-1)*n + j] + h[(i+1)*n + j] + h[i*n + (j-1)] + h[i*n + (j+1)]);
            
    
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                h[i*n + j] = g[i*n + j];
    }
}

void initialize(double *h, int n)
{
    int fireplace_start = (FIREPLACE_START * n) / ROOM_SIZE;
    int fireplace_end = (FIREPLACE_END * n) / ROOM_SIZE;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                h[i*n + j] = (i == n - 1 && j >= fireplace_start && j <= fireplace_end) ? FIREPLACE_TEMP : WALL_TEMP;
            else
                h[i*n + j] = 0.0;
        }
    }
}

double calculate_elapsed_time(struct timespec start, struct timespec end)
{
    double start_sec = (double)start.tv_sec * 1e9 + (double)start.tv_nsec;
    double end_sec = (double)end.tv_sec * 1e9 + (double)end.tv_nsec;
    return (end_sec - start_sec) / 1e9;
}

void save_to_file(double *h, int n)
{
    FILE *file = fopen("room.txt", "w");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            fprintf(file, "%lf ", h[i*n + j]);
        fprintf(file, "\n");
    }
    fclose(file);
}

int main(int argc, char *argv[])
{
    if (argc < 3) {
        fprintf(stderr, "Uso: %s <número de pontos> <limite de iterações>\n", argv[0]);
        return 1;
    }

    struct timespec start_host, end_host;
    struct timespec start_device, end_device;
    double *h, *g;
    double *d_h, *d_g; 

    int n = atoi(argv[1]);
    int iter_limit = atoi(argv[2]);

    h = (double *)malloc(n*n * sizeof(double));
    g = (double *)malloc(n*n * sizeof(double));
    if (h == NULL || g == NULL) {
        fprintf(stderr, "Erro ao alocar memória para h ou g\n");
        exit(EXIT_FAILURE);
    }

    initialize(h, n);

    // Allocate device memory 
    cudaMalloc((void**)&d_h, n*n * sizeof(double));
    cudaMalloc((void**)&d_g, n*n * sizeof(double));

    // Transfer data from host to device memory
    cudaMemcpy(d_h, h, n*n * sizeof(double), cudaMemcpyHostToDevice);

    // Executing kernel 
    int block_size = 256;
    int grid_size = ((n + block_size) / block_size);

    clock_gettime(CLOCK_MONOTONIC, &start_device);
    jacobi_iteration<<<1,1>>>(d_h, d_g, n, iter_limit);
    clock_gettime(CLOCK_MONOTONIC, &end_device);
    
    // Transfer data back to host memory
    cudaMemcpy(h, d_h, n*n * sizeof(double), cudaMemcpyDeviceToHost);
    save_to_file(h, n);

    // Verification

    // printf("Tempo de execução CPU: %.9f segundos\n", calculate_elapsed_time(start_host, end_host));
    printf("Tempo de execução GPU: %.9f segundos\n", calculate_elapsed_time(start_device, nd_device));

    // Deallocate device memory
    cudaFree(d_h);
    cudaFree(d_g);

    // Deallocate host memory
    free(h); 
    free(g); 

    return 0;
}