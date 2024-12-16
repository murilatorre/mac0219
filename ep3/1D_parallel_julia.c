#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h> // (Message Passing Interface)


/*
 * compute_julia_pixel(): calcula os valores RGB de um pixel em
 *                        uma imagem específica de um conjunto de Julia.
 *
 *  Entrada:
 *      (x, y):           coordenadas do pixel
 *      (width, height):  dimensões da imagem
 *      tint_bias:        um valor float para ajustar a tonalidade (1.0 é "sem ajuste")
 *  Saída:
 *      rgb: um array já alocado de 3 bytes onde serão escritos os valores
 *           R, G e B.
 *
 *  Retorno:
 *      0 em caso de sucesso, -1 em caso de falha
 *
 */

int compute_julia_pixel(int x, int y, int width, int height, float tint_bias, unsigned char *rgb) {

  // Verifica se as coordenadas são válidas
  if ((x < 0) || (x >= width) || (y < 0) || (y >= height)) {
    fprintf(stderr, "Coordenadas inválidas (%d,%d) para um pixel em uma imagem de %d x %d\n", x, y, width, height);
    return -1;
  }
 
  // "Amplia" a visualização para mostrar uma área agradável do conjunto de Julia
  float X_MIN = -1.6, X_MAX = 1.6, Y_MIN = -0.9, Y_MAX = +0.9;
  float float_y = (Y_MAX - Y_MIN) * (float)y / height + Y_MIN;
  float float_x = (X_MAX - X_MIN) * (float)x / width + X_MIN;

  // Ponto que define o conjunto de Julia
  float julia_real = -.79;
  float julia_img = .15;

  // Número máximo de iterações
  int max_iter = 300;

  // Calcula a convergência da série complexa
  float real = float_y, img = float_x;
  int num_iter = max_iter;
  while ((img * img + real * real < 2 * 2) && (num_iter > 0)) {
    float xtemp = img * img - real * real + julia_real;
    real = 2 * img * real + julia_img;
    img = xtemp;
    num_iter--;
  }

  // Pinta o pixel com base no número de iterações usando uma coloração estilizada
  float color_bias = (float) num_iter / max_iter;
  rgb[0] = (num_iter == 0 ? 200 : -500.0 * pow(tint_bias, 1.2) * pow(color_bias, 1.6));
  rgb[1] = (num_iter == 0 ? 100 : -255.0 * pow(color_bias, 0.3));
  rgb[2] = (num_iter == 0 ? 100 : 255 - 255.0 * pow(tint_bias, 1.2) * pow(color_bias, 3.0));

  return 0;
}

/* write_bmp_header():
 *
 *   Entrada:
 *      f: Um arquivo aberto para escrita ('w') 
 *      (width, height): dimensões da imagem
 *   
 *   Retorno:
 *      0 em caso de sucesso, -1 em caso de falha
 *
 */

int write_bmp_header(FILE *f, int width, int height) {

  unsigned int row_size_in_bytes = width * 3 + 
	  ((width * 3) % 4 == 0 ? 0 : (4 - (width * 3) % 4));

  // Define todos os campos no cabeçalho do BMP
  char id[3] = "BM";
  unsigned int filesize = 54 + (int)(row_size_in_bytes * height * sizeof(char));
  short reserved[2] = {0,0};
  unsigned int offset = 54;

  unsigned int size = 40;
  unsigned short planes = 1;
  unsigned short bits = 24;
  unsigned int compression = 0;
  unsigned int image_size = width * height * 3 * sizeof(char);
  int x_res = 0;
  int y_res = 0;
  unsigned int ncolors = 0;
  unsigned int importantcolors = 0;

  // Escreve os bytes no arquivo, mantendo o controle do
  // número de "objetos" escritos
  size_t ret = 0;
  ret += fwrite(id, sizeof(char), 2, f);
  ret += fwrite(&filesize, sizeof(int), 1, f);
  ret += fwrite(reserved, sizeof(short), 2, f);
  ret += fwrite(&offset, sizeof(int), 1, f);
  ret += fwrite(&size, sizeof(int), 1, f);
  ret += fwrite(&width, sizeof(int), 1, f);
  ret += fwrite(&height, sizeof(int), 1, f);
  ret += fwrite(&planes, sizeof(short), 1, f);
  ret += fwrite(&bits, sizeof(short), 1, f);
  ret += fwrite(&compression, sizeof(int), 1, f);
  ret += fwrite(&image_size, sizeof(int), 1, f);
  ret += fwrite(&x_res, sizeof(int), 1, f);
  ret += fwrite(&y_res, sizeof(int), 1, f);
  ret += fwrite(&ncolors, sizeof(int), 1, f);
  ret += fwrite(&importantcolors, sizeof(int), 1, f);

  // Sucesso significa que escrevemos 17 "objetos" com êxito
  return (ret != 17);
}


void write_bmp_lines(FILE *file, unsigned char *pixels, int width, int numrows) {
    for (int i = 0; i < numrows; i++) {
        fwrite(&pixels[i * width * 3], sizeof(unsigned char), width * 3, file); 
        // padding no caso de um número par de pixels por linha
        unsigned char padding[3] = {0,0,0};
        fwrite(padding, sizeof(char), ((width * 3) % 4), file);
    }

}


int main(int argc, char *argv[])  {
    int numtasks, rank;
    int n, height, width;
    float tint;
    
    if (argc < 2) {
        fprintf(stderr, "Uso: %s <tamanho da imagem>\n", argv[0]);
        return 1;
    }
    
    // define dimensões e tint da imagem
    n = atoi(argv[1]);
    height = n;
    width = 2*n;
    tint = 1.0;
    
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Obtém o rank do processo atual
    MPI_Comm_size(MPI_COMM_WORLD, &numtasks); // Obtém o número de processos no comunicador

    // essa parte é comum a todos os processos
    char filename[] = "parallel_julia.bmp";
    int rowsPerTask = height / numtasks, remainder = height % numtasks;
    int start   = rowsPerTask * rank  + (rank - 1 < remainder ? rank : remainder);
    int end     = start + rowsPerTask + (rank < remainder) - 1;
    int numrows = end - start + 1;


    // Cada pixel é representado por 3 bytes (RGB)
    // Só alocamos os pixels necessários
    unsigned char *pixels = (unsigned char *)malloc(3*numrows*width * sizeof(unsigned char));
    if (pixels == NULL) {
        fprintf(stderr, "[Process %d]: Erro ao alocar memória para pixels.\n", rank);
        MPI_Finalize();
        return -1;
    }

    printf("[Process %d out of %d]: I should compute pixel rows %d to %d, for a total of %d rows\n", rank, numtasks, start, end, numrows);
    
    double start_time = MPI_Wtime();

    // Computar os pixels para as linhas atribuídas
    for (int y = 0; y < numrows; y++) {
        for (int x = 0; x < width; x++) {
            compute_julia_pixel(x, start + y, width, height, tint, &pixels[3*(y*width + x)]);
        }
    }

    if (rank == 0) {
        FILE *file = fopen(filename, "w");
        if (write_bmp_header(file, width, height)) {
            fprintf(stderr, "Erro ao escrever o cabeçalho BMP.\n");
            fclose(file); MPI_Finalize();
            return -1;
        }

        write_bmp_lines(file, pixels, width, numrows);
        fclose(file);

        int auth = 1;
        MPI_Send(&auth, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    } else {
        int auth_recv;
        MPI_Recv(&auth_recv, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        FILE *file = fopen(filename, "a");
        write_bmp_lines(file, pixels, width, numrows);
        fclose(file);

        if (rank < numtasks - 1) {
            MPI_Send(&auth_recv, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
        }
    }

    double end_time = MPI_Wtime();
    printf("[Process %d out of %d]: Time elapsed during the job: %.2fs.\n", rank, numtasks, (end_time-start_time));  

    free(pixels);
    MPI_Finalize();

    return 0;
}