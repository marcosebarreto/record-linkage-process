// Linkage Process using CUDA C API
// Authors: Clicia Santos Pinto and Pedro Marcelino Mendes Novaes Melo

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda.h>

#define NCOL 101


void fill_matrix(int *matrix, int pos, char *line);
int get_num_of_lines(FILE *fp);
void process_file(FILE *fp, int *matrix);
void print_matrix(int *matrix, int nlines);
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b);
__device__ float dice(int *bloomA, int *bloomB);


int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    int nlines_a = 0, nlines_b = 0;

    // opening large base (base_a) and small base (base_b)
    printf("[LOADING DATABASES ... ]\n");
    base_a = fopen("base_a10.bloom", "r");
    base_b = fopen("base_b10.bloom", "r");

    // --------------------- OPERATIONS WITH BASE A --------------------- //
    // getting line quantity
    printf("[GETTING NUMBER LINES FOR BASE A ... ]\n");
    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));

    // processing base_a to fill matrixA
    printf("[PROCESSING BASE A ... ]\n");
    process_file(base_a, matrixA);
    print_matrix(matrixA, nlines_a);

    // --------------------- OPERATIONS WITH BASE B --------------------- //
    // getting line quantity
    printf("[GETTING NUMBER LINES FOR BASE B ... ]\n");
    nlines_b = get_num_of_lines(base_b);
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));

    // processing base_b to fill matrixB
    printf("[PROCESSING BASE B ... ]\n");
    process_file(base_b, matrixB);
    print_matrix(matrixB, nlines_b);

    // ------------------------- CUDA OPERATIONS ------------------------ //
    int *matrixA_d, *matrixB_d;

    // allocating device memory using a cuda function
    cudaMalloc((int **)&matrixA_d, nlines_a * NCOL * sizeof(int));
    cudaMalloc((int **)&matrixB_d, nlines_b * NCOL * sizeof(int));

    // copying host memory to device
    cudaMemcpy(matrixA_d, matrixA, nlines_a * NCOL * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(matrixB_d, matrixB, nlines_b * NCOL * sizeof(int), cudaMemcpyHostToDevice);

    // kernel operations
    printf("[OPERATING AT KERNEL CUDA ... ]\n");
    kernel<<<nlines_a, 1>>>(matrixA_d, matrixB_d, nlines_a, nlines_b);

    // deallocating device memory
    cudaFree(matrixA_d);
    cudaFree(matrixB_d);

    fclose(base_a);
    fclose(base_b);

    return 0;
}


// function to get the number of lines of the file
int get_num_of_lines(FILE *fp) {
    int lines = 0;
    char line[256];

    fgets(line, 255, fp);
    while (!feof(fp)) {
        lines++;
        fgets(line, 255, fp);
    }

    // printf("num lines: %d\n", lines);
    return lines;
}


// function to get line by line of the file
void process_file(FILE *fp, int *matrix) {
    char line[256];
    int pos_to_insert = 0;

    rewind(fp);

    // getting line by line to insert into matrix
    fgets(line, 255, fp);
    while (!feof(fp)) {
        line[strlen(line) - 1] = '\0';
        // printf("%s\n", line);
        fill_matrix(matrix, pos_to_insert, line);

        pos_to_insert++;
        fgets(line, 255, fp);
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


__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
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
}


// device function to calculate dice coefficient using bloom filter
__device__ float dice(int *bloomA, int *bloomB) {
    printf("teste\n");
}
