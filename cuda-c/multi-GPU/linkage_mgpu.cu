/*
@(#)File:           $linkage.cu$
@(#)Version:        $v3$
@(#)Last changed:   $Date: 2017/02/03 09:05:00 $
@(#)Purpose:        Probabilistic linkage for multi-GPU
@(#)Author:         Pedro Marcelino Mendes Novaes Melo
                    Clicia Santos Pinto
                    Murilo Boratto
@(#)Usage:
 (*) Hotocompile:   make clean; make
 (*) Hotoexecute:  ./object <num_threads_per_block> <file1> <threads_openmp> <percentage_each_gpu> <qtd_gpu>
 (*) Hotoexecute:  ./linkage 16 1000 32 45 2
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
int get_num_of_lines(FILE *fp);
void process_file(FILE *fp, int *matrix);
void print_matrix(int *matrix, int nlines);
int *divide(int *source_matrix, int lower_threshold, int upper_threshold);
int *get_pu_threshold(int lines, int qtd_gpu, int percentage_each_gpu);
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b);
__device__ float dice(int *bloomA, int *bloomB);


int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    double t1, t2;
    t1 = omp_get_wtime();

    int nlines_a = 0, nlines_b = 0;
    char file1[30];

    // reading arguments
    int threads_per_block = atoi(argv[1]);
    strcpy(file1, "base_");
    strcat(file1, argv[2]);
    strcat(file1, "K.bloom");
    int threads_openmp = atoi(argv[3]);
    int percentage_each_gpu = atoi(argv[4]);
    int qtd_gpu = atoi(argv[5]);

    // printf("[LOADING DATABASES ... ]\n");
    base_a = fopen(file1, "r");
    base_b = fopen("base_1000K.bloom", "r");

    // --------------------- OPERATIONS WITH BASE A --------------------- //
    // getting line quantity
    // printf("[GETTING NUMBER LINES FOR BASE A ... ]\n");
    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));

    // processing base_a to fill matrixA
    // printf("[PROCESSING BASE A ... ]\n");
    process_file(base_a, matrixA);
    // print_matrix(matrixA, nlines_a);

    // --------------------- OPERATIONS WITH BASE B --------------------- //
    // getting line quantity
    // printf("[GETTING NUMBER LINES FOR BASE B ... ]\n");
    nlines_b = get_num_of_lines(base_b);
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));

    // processing base_b to fill matrixB
    // printf("[PROCESSING BASE B ... ]\n");
    process_file(base_b, matrixB);
    // print_matrix(matrixB, nlines_b);

    // pre-processing
    int *pu_threshold;
    pu_threshold = get_pu_threshold(nlines_a, qtd_gpu, percentage_each_gpu);


    // ------------------------- CUDA OPERATIONS ------------------------ //

    // pragma directive to create 2 threads to execute cuda code in parallel
    #pragma omp parallel num_threads(2)
    {

        int id;
        int gpu_id = -1;
        id = omp_get_thread_num();
        cudaSetDevice(id);
        cudaGetDevice(&gpu_id);

        int *matrixA_d, *matrixB_d;
        int lower_threshold, upper_threshold;

        // splitting matrixA into 2 GPUs
        if(id == 0){
    	    int *matrixA_tmp;
    	    // lower_threshold = 0;
    	    // upper_threshold = (nlines_a/2);
            lower_threshold = pu_threshold[2];
            upper_threshold = pu_threshold[3];
            printf("GPU1 = %d -- %d\n", lower_threshold, upper_threshold);
    	    matrixA_tmp = divide(matrixA, lower_threshold, upper_threshold);

    	    cudaMalloc((int **)&matrixA_d, (upper_threshold - lower_threshold) * NCOL * sizeof(int));
            cudaMemcpy(matrixA_d, matrixA_tmp, (upper_threshold - lower_threshold) * NCOL * sizeof(int), cudaMemcpyHostToDevice);
        }
        else{
            int *matrixA_tmp;
            // lower_threshold = (nlines_a / 2);
            // upper_threshold = (nlines_a);
            lower_threshold = pu_threshold[4];
            upper_threshold = pu_threshold[5];
            printf("GPU2 = %d -- %d\n", lower_threshold, upper_threshold);
            matrixA_tmp = divide(matrixA, lower_threshold, upper_threshold);
            cudaMalloc((int **)&matrixA_d, (upper_threshold - lower_threshold) * NCOL * sizeof(int));
            cudaMemcpy(matrixA_d, matrixA_tmp, (upper_threshold - lower_threshold) * NCOL * sizeof(int), cudaMemcpyHostToDevice);
        }

        // allocating device memory using a cuda function
        cudaMalloc((int **)&matrixB_d, nlines_b * NCOL * sizeof(int));

        // copying host memory to device
        cudaMemcpy(matrixB_d, matrixB, nlines_b * NCOL * sizeof(int), cudaMemcpyHostToDevice);

        // kernel operations
        // printf("[OPERATING AT KERNEL CUDA ... ]\n");
        dim3 dimGrid = (int) ceil( (int) (upper_threshold - lower_threshold) / (int) threads_per_block);
        dim3 dimBlock = threads_per_block;
        kernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, (upper_threshold - lower_threshold), nlines_b);

        cudaDeviceSynchronize();

        // deallocating device memory
        cudaFree(matrixA_d);
        cudaFree(matrixB_d);

    } // end pragma openmp

    free(matrixA);
    free(matrixB);

    // closing files
    fclose(base_a);
    fclose(base_b);

    t2 = omp_get_wtime();

    int length_problem = atoi(argv[2]);
    printf("%d\t%f\n", (length_problem * 1000), (t2-t1));

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


// function to indicate the upper and lower threshold for each gpu and cpu
// according to number of lines in matrixA
int *get_pu_threshold(int lines, int qtd_gpu, int percentage_each_gpu) {
    static int *threshold_vector;
    threshold_vector = (int *)malloc((2 + (qtd_gpu * 2)) * sizeof(int));

    int percentage_gpu = percentage_each_gpu * qtd_gpu;
    int percentage_cpu = 100 - percentage_gpu;

    int i;
    int init_line = 0;

    if (percentage_cpu == 0) {
        threshold_vector[0] = -1;
        threshold_vector[1] = -1;
    }
    else {
        threshold_vector[0] = init_line;
        init_line = (lines * percentage_cpu)/100;
        threshold_vector[1] = init_line;
    }

    for (i = 2; i < (2 + qtd_gpu * 2); i = i + 2) {
        threshold_vector[i] = init_line;
        init_line = init_line + (lines * percentage_each_gpu)/100;
        if ((lines * percentage_each_gpu)/100 % 2 != 0) {
            init_line++;
        }
        threshold_vector[i + 1] = init_line;
    }

    threshold_vector[1 + qtd_gpu * 2] = lines;

    return threshold_vector;
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
    double a = 0, b = 0, h = 0;
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
    double dice = ((h * 2.0) / (a + b)) * 10000;
    //   printf("%.1f\n", dice);
    // contador++;

    return dice;
}
