#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h> 
#include <cuda.h>
#include <cuda_runtime.h>

#define WALL_TEMP 20.0
#define BODY_TEMP 37.0

#define BODY_X 5
#define BODY_Y 5
#define ROOM_SIZE 10


__global__ void jacobi_iteration(double *h, double *g, int n, int iter_limit)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y;  // ROW
    int j = blockIdx.x * blockDim.x + threadIdx.x;  // COLUMN

    if (i > 0 && i < n - 1 && j > 0 && j < n - 1) {
        g[i * n + j] = (i == (BODY_X * n) / ROOM_SIZE && j == (BODY_Y * n) / ROOM_SIZE) ? 
                         h[i*n +j] : 0.25 * (h[(i - 1) * n + j] + h[(i + 1) * n + j] + h[i * n + (j - 1)] + h[i * n + (j + 1)]);
    } 
   
}


void c_jacobi_iteration(double *h, double *g, int n, int iter_limit)
{
    int body_x = (BODY_X * n) / ROOM_SIZE;
    int body_y = (BODY_Y * n) / ROOM_SIZE;

    for (int iter = 0; iter < iter_limit; iter++) {
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                g[i * n + j] = (i == body_x && j == body_y) ? 
                         h[i*n +j] : 0.25 * (h[(i - 1) * n + j] + h[(i + 1) * n + j] + h[i * n + (j - 1)] + h[i * n + (j + 1)]);
            
    
        for (int i = 1; i < n - 1; i++)
            for (int j = 1; j < n - 1; j++)
                h[i*n + j] = g[i*n + j];
    }
}

void initialize(double *h, int n)
{
    int body_x = (BODY_X * n) / ROOM_SIZE;
    int body_y = (BODY_Y * n) / ROOM_SIZE;

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == 0 || i == n - 1 || j == 0 || j == n - 1)
                h[i*n + j] = WALL_TEMP;
            else if (i == body_x && j == body_y) 
                h[i*n + j] = BODY_TEMP;
            else
                h[i*n + j] = 0.0;
        }
    }
}

bool equal_result(double *res_cpu, double *res_gpu, int n) {
     for (int i = 0; i < n*n; i++) {
            if (res_cpu[i] != res_gpu[i]) return false;
    }
    return true;
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
    if (argc < 4) {
        fprintf(stderr, "Uso: %s <número de pontos> <limite de iterações> <número de threads por bloco>\n", argv[0]);
        return 1;
    }

    struct timespec start_host, end_host;
    struct timespec start_device, end_device;
    struct timespec start_mov_hd, end_mov_hd;
    struct timespec start_mov_dh, end_mov_dh;
    
    double *h, *g;
    double *cpu_h, *cpu_g;
    double *d_h, *d_g; 

    int n = atoi(argv[1]);
    int iter_limit = atoi(argv[2]);
    int t = atoi(argv[3]);
    int b = (n + t) / t;

    dim3 block_dim(t, t);
    dim3 grid_dim(b, b);

    h = (double *)malloc(n*n * sizeof(double));
    g = (double *)malloc(n*n * sizeof(double));
    cpu_h = (double *)malloc(n*n * sizeof(double));
    cpu_g = (double *)malloc(n*n * sizeof(double));
    if (h == NULL || g == NULL || cpu_h == NULL || cpu_g == NULL) {
        fprintf(stderr, "Erro ao alocar memória para h ou g\n");
        exit(EXIT_FAILURE);
    }

    // Initialize host arrays
    initialize(h, n);
    initialize(cpu_h, n);

    // Allocate device memory 
    cudaMalloc((void**)&d_h, n*n * sizeof(double));
    cudaMalloc((void**)&d_g, n*n * sizeof(double));

    // Transfer data from host to device memory
    clock_gettime(CLOCK_MONOTONIC, &start_mov_hd);
    cudaMemcpy(d_h, h, n*n * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h, n*n * sizeof(double), cudaMemcpyHostToDevice);
    clock_gettime(CLOCK_MONOTONIC, &end_mov_hd);

    // Executing kernel 
    clock_gettime(CLOCK_MONOTONIC, &start_device);
    for (int iter = 0; iter < iter_limit; iter++) {
        jacobi_iteration<<<grid_dim, block_dim>>>(d_h, d_g, n, iter_limit);
        cudaDeviceSynchronize();
        double *temp = d_h;
        d_h = d_g;
        d_g = temp;
    }
    clock_gettime(CLOCK_MONOTONIC, &end_device);
    
    // Transfer data back to host memory
    clock_gettime(CLOCK_MONOTONIC, &start_mov_dh);
    cudaMemcpy(h, d_h, n*n * sizeof(double), cudaMemcpyDeviceToHost);
    clock_gettime(CLOCK_MONOTONIC, &end_mov_dh);
    save_to_file(h, n);

    clock_gettime(CLOCK_MONOTONIC, &start_host);
    c_jacobi_iteration(cpu_h, cpu_g, n, iter_limit);
    clock_gettime(CLOCK_MONOTONIC, &end_host);

    // Verification
    if (equal_result(cpu_h, h, n))
        printf("CORRETO\n");
    else printf("ERRADO\n");

    // Print results
    printf("Tempo de execução CPU: %.9f segundos\n", calculate_elapsed_time(start_host, end_host));
    printf("Tempo de execução GPU: %.9f segundos\n", calculate_elapsed_time(start_device, end_device));
    printf("Tempo de movimentação de dados entre device e host: %.9f segundos\n", calculate_elapsed_time(start_mov_dh, end_mov_dh));
    printf("Tempo de movimentação de dados entre host e device: %.9f segundos\n", calculate_elapsed_time(start_mov_hd, end_mov_hd));


    // Deallocate device memory
    cudaFree(d_h);
    cudaFree(d_g);

    // Deallocate host memory
    free(h); 
    free(g); 

    return 0;
}
