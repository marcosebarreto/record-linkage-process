/*
@(#)File:           $linkage.cu$
@(#)Version:        $v2$
@(#)Last changed:   $Date: 2017/01/31 09:05:00 $
@(#)Purpose:        Probabilistic linkage for 1GPU
@(#)Author:         Pedro Marcelino Mendes Novaes Melo
                    Clicia Santos Pinto
                    Murilo Boratto
@(#)Usage:
 (*) Hotocompile:   make clean; make
 (*) Hotoexecute:  ./object <threads_per_block> <larger_file>
 (*) Hotoexecute:  ./linkage 16 10
@(#)Comment:
 (*) Pass arguments (name of file *.bloom) for command-line interface
 (*) Get time with omp_get_wtime() in seconds
 (*) Inaccurate Divide dimGrid
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>
#include <time.h>
#include <omp.h>

#define NCOL 101

__device__ int contador = 0;

void fill_matrix(int *matrix, int pos, char *line);
void process_file(FILE *fp, int *matrix);
void print_matrix(int *matrix, int nlines);
int get_num_of_lines(FILE *fp);
int *divide(int *source_matrix, int lower_threshold, int upper_threshold);
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b);
__device__ float dice(int *bloomA, int *bloomB);


int main(int argc, char const *argv[]) {
    double t1, t2;
    t1 = omp_get_wtime();

    FILE *base_a, *base_b;
    char file1[30];
    strcpy(file1, "base_");
    strcat(file1, argv[2]);
    strcat(file1, "K.bloom");

    int nlines_a = 0, nlines_b = 0;
    int threads_per_block = atoi(argv[1]);

    // opening large base (base_a) and small base (base_b)
    //printf("[LOADING DATABASES ... ]\n");
    base_a = fopen(file1, "r");
    base_b = fopen("base_10000.bloom", "r");

    // --------------------- OPERATIONS WITH BASE A --------------------- //
    // getting line quantity
    //printf("[GETTING NUMBER LINES FOR BASE A ... ]\n");
    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));

    // processing base_a to fill matrixA
    printf("[PROCESSING BASE A ... ]\n");
    process_file(base_a, matrixA);
    // print_matrix(matrixA, nlines_a);

    // testing divide function
    int *test;
    int lower_threshold = 0;
    int upper_threshold = 5;
    test = divide(matrixA, lower_threshold, upper_threshold);
    //print_matrix(test, (upper_threshold - lower_threshold));


    // --------------------- OPERATIONS WITH BASE B --------------------- //
    // getting line quantity
    //printf("[GETTING NUMBER LINES FOR BASE B ... ]\n");
    nlines_b = get_num_of_lines(base_b);
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));

    // processing base_b to fill matrixB
    //printf("[PROCESSING BASE B ... ]\n");
    process_file(base_b, matrixB);
    // print_matrix(matrixB, nlines_b);


    // ------------------------- CUDA OPERATIONS ------------------------ //
    int *matrixA_d, *matrixB_d;

    // allocating device memory using a cuda function
    cudaMalloc((int **)&matrixA_d, nlines_a * NCOL * sizeof(int));
    cudaMalloc((int **)&matrixB_d, nlines_b * NCOL * sizeof(int));

    // copying host memory to device
    cudaMemcpy(matrixA_d, matrixA, nlines_a * NCOL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB, nlines_b * NCOL * sizeof(int), cudaMemcpyHostToDevice);

    // kernel operations
    //printf("[OPERATING AT KERNEL CUDA ... ]\n");
    dim3 dimGrid = (int) ceil( (int) nlines_a / (int) threads_per_block);
    dim3 dimBlock = threads_per_block;
    kernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, nlines_a, nlines_b);

    cudaDeviceSynchronize();

    // deallocating device memory
    cudaFree(matrixA_d);
    cudaFree(matrixB_d);

    free(matrixA);
    free(matrixB);

    // close files
    fclose(base_a);
    fclose(base_b);

    t2 = omp_get_wtime();

    int length_problem = atoi(argv[2]);
    printf("%d\t%f\n", (length_problem ), (t2-t1));

    return 0;
}


// function to get the number of lines of the file
int get_num_of_lines(FILE *fp) {
    int lines = 0;
    char line[256];
    if(!fgets(line, 255, fp))
        printf("fp = NULL");

    while (!feof(fp)) {
        lines++;
        if(!fgets(line, 255, fp))
            break;
    }

    return lines;
}


// function to get line by line of the file
void process_file(FILE *fp, int *matrix) {
    char line[256];
    int pos_to_insert = 0;

    rewind(fp);

    // getting line by line to insert into matrix
    if(!fgets(line, 255, fp))
        printf("fp = NULL");
    while (!feof(fp)) {
        line[strlen(line) - 1] = '\0';
        fill_matrix(matrix, pos_to_insert, line);

        pos_to_insert++;
        if(!fgets(line, 255, fp))
            break;
    }
}


// function to split a line and to insert the elements in matrix
void fill_matrix(int *matrix, int pos, char *line) {
    int i = 0, j = 0;
    char c, id[10];

    do {
        c = line[j];
        id[j] = c;
        j++;
    } while (c != ';');
    id[j-1] = '\0';
    // printf("ncol * pos: %d\n", NCOL * pos);
    matrix[NCOL * pos] = atoi(id);

    for (i = 1; i < NCOL; i++) {
        matrix[NCOL * pos + i] = line[j] - '0';
        j++;
    }
}


// function to divide matrixA into a smaller matrix, given a lower threshold
// and a upper threshold. Each one will be executed on a GPU
int *divide(int *source_matrix, int lower_threshold, int upper_threshold) {
    static int *destination_matrix;
    destination_matrix = (int *)malloc((upper_threshold - lower_threshold) * NCOL * sizeof(int));

    int i, j = 0;

    for (i = (lower_threshold * NCOL); i < (upper_threshold * NCOL); i++) {
        destination_matrix[j] = source_matrix[i];
        j++;
    }

    return destination_matrix;
}


void print_matrix(int *matrix, int nlines) {
    int i, j;

    // for (i = 0; i < NCOL * nlines; i += 101) {
    //     printf("%d | ", matrix[i]);
    // }
    // printf("\n");

    for (i = 0; i < nlines; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrix[i * NCOL + j]);
        }
        printf("\n");
    }
    printf("\n");
}


// Kernel CUDA to compute linkage between matrixA and matrixB using a dice
// function as similarity measure
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // printf("I = %d - blockID.x= %d e blockId.y = %d -- blockDim.x = %d e blockDim.y = %d -- threadIdx.x = %d e threadIdx.y = %d\n", i, blockIdx.x, blockIdx.y, blockDim.x, blockDim.y, threadIdx.x, threadIdx.y);
    // int j = blockIdx.y * blockDim.y + threadIdx.y;

    int bloomA[100], bloomB[100];

    if (i < nlines_a) {
        // printf("%d ", matrixA[i * NCOL]);

        // getting bloom filter for each matrixA register
        for (int j = 1; j < 101; j++) {
            bloomA[j - 1] = matrixA[i * NCOL + j];
        }

        // getting bloom filter for each matrixB register
        for (int k = 0; k < nlines_b; k++) {
            for (int l = 1; l < 101; l++) {
                bloomB[l - 1] = matrixB[k * NCOL + l];
            }
            dice(bloomA, bloomB);
        }
    }

    // printf("num de comparacoes para thread %d: %d\n", i, contador);
}


// device function to calculate dice coefficient using bloom filter
__device__ float dice(int *bloomA, int *bloomB) {
    float a = 0, b = 0, h = 0;
    int i;

    for (i = 0; i < 100; i++) {
        if (bloomA[i] == 1) {
            a++;
            if (bloomB[i] == 1)
                h++;
        }
        if (bloomB[i] == 1) {
            b++;
        }
    }
    float dice = ((h * 2.0) / (a + b)) * 10000;
    // printf("%.1f\n", dice);
    // contador++;

    return dice;
}
