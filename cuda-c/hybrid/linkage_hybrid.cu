/*
@(#)File:           $linkage.cu$
@(#)Version:        $v3$
@(#)Last changed:   $Date: 2017/02/09 09:05:00 $
@(#)Purpose:        Probabilistic linkage for multi-GPU
@(#)Author:         Pedro Marcelino Mendes Novaes Melo
                    Clicia Santos Pinto
                    Murilo Boratto
@(#)Usage:
 (*) Hotocompile:   make clean; make
 (*) Hotoexecute:  ./object <num_threads_per_block> <file1> <threads_openmp> <percentage_each_gpu> <num_gpus>
 (*) Hotoexecute:  ./linkage 64 100 32 40 2
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

void fill_matrix(int *, int , char *);
int get_num_of_lines(FILE *);
void process_file(FILE *, int *);
void print_matrix(int *, int );
int *divide_matrix(int *, int , int );
int *get_pu_edges(int , int , int );
void multicore_execution(int *, int *, int , int , int , int );
float dice_multicore(int *, int *);
//void multicore_execution(int nlines_b, int id_nested, int quantum, int leftover);
__global__ void kernel(int *, int *, int , int );
__device__ float dice(int *, int *);


int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    char file1[30];
    double t1, t2;
    int nlines_a = 0, nlines_b = 0;
    int *pu_edges;
    int threads_per_block = atoi(argv[1]);
 	int threads_openmp = atoi(argv[3]);
    int percentage_each_gpu = atoi(argv[4]);
    int num_gpus = atoi(argv[5]);

    t1 = omp_get_wtime();
    
    strcpy(file1, "base_");
    strcat(file1, argv[2]);
    strcat(file1, "K.bloom");

    // printf("[LOADING DATABASES ... ]\n");
    base_a = fopen(file1, "r");
    base_b = fopen("base_1000K.bloom", "r");

    // --------------------- INIT: Reading A and B Files --------------------- //
    // printf("[GETTING NUMBER LINES FOR BASE A ... ]\n");
    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));

    // Filling matrixA with records from original file
    process_file(base_a, matrixA);
    // print_matrix(matrixA, nlines_a); //TODO: apagar após testar

    // printf("[GETTING NUMBER LINES FOR BASE B ... ]\n");
    nlines_b = get_num_of_lines(base_b);
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));

    // Filling matrixB with records from original file
    process_file(base_b, matrixB);
    // print_matrix(matrixB, nlines_b);
    // --------------------- END: Reading A and B Files --------------------- //

    // Getting initial and final indexes in which each PU will execute
    pu_edges = get_pu_edges(nlines_a, num_gpus, percentage_each_gpu);

//TODO:Apagar após testar
/*   printf("Imprimindo o vetor de índices: ");
    for(int i=0;i<6;i++){
    	printf("%d ", pu_edges[i]);
	}
	printf("\n");
*/	

    omp_set_nested(1);
    #pragma omp parallel num_threads(num_gpus+1)
    {
		int *matrixA_d, *matrixB_d;
		int lower_edge, upper_edge, thread_id, gpu_id = -1;

		thread_id = omp_get_thread_num();
		cudaSetDevice(thread_id);
		cudaGetDevice(&gpu_id);

		// Thread 0 will perform multicore execution
		if(thread_id == 0){
			int *matrixA_tmp;
			lower_edge = pu_edges[thread_id * 2];
			upper_edge = pu_edges[(thread_id * 2)+1];
			// matrixA_tmp will store a part of matrixA equivalent the percentage chosen.
			matrixA_tmp = divide_matrix(matrixA, lower_edge, upper_edge);

			double quantum, leftover;
			quantum = (float)upper_edge / threads_openmp;
			leftover = (quantum - (int)quantum) * threads_openmp;
			//leftover = ceilf(leftover);
			//printf("Mensagem da thread multicore mãe: Quantum: %f, Leftover: %f\n",quantum, leftover);

			#pragma omp parallel num_threads(threads_openmp)
			{
				int id_nested;
				id_nested = omp_get_thread_num();
				//printf("Thread mãe: %d executando a thread interna %d\n",thread_id, id_nested); //TODO: apagar após teste
				multicore_execution(matrixA_tmp, matrixB, nlines_b, id_nested, (int) quantum, (int) leftover);				
			}
		}
		// Thread 1 (and higher) will perform multicore execution.
        else{
//			printf("Executando a thread de numero: %d\n", thread_id);
			int *matrixA_tmp;
			lower_edge = pu_edges[thread_id * 2];
			upper_edge = pu_edges[(thread_id * 2)+1];
			matrixA_tmp = divide_matrix(matrixA, lower_edge, upper_edge);
			//print_matrix(matrixA_tmp,upper_edge-lower_edge); //TODO: apagar após teste

			cudaMalloc((int **)&matrixA_d, (upper_edge - lower_edge) * NCOL * sizeof(int));
			cudaMemcpy(matrixA_d, matrixA_tmp, (upper_edge - lower_edge) * NCOL * sizeof(int), cudaMemcpyHostToDevice);
			cudaMalloc((int **)&matrixB_d, nlines_b * NCOL * sizeof(int));
			cudaMemcpy(matrixB_d, matrixB, nlines_b * NCOL * sizeof(int), cudaMemcpyHostToDevice);

			// kernel operations
			// printf("[OPERATING AT KERNEL CUDA ... ]\n"); // TODO: apagar apos teste
			dim3 dimGrid = (int) ceil( (int) (upper_edge - lower_edge) / (int) threads_per_block);
			dim3 dimBlock = threads_per_block;
			kernel<<<dimGrid, dimBlock>>>(matrixA_d, matrixB_d, (upper_edge - lower_edge), nlines_b);

			cudaDeviceSynchronize();

			// deallocating device memory
			cudaFree(matrixA_d);
			cudaFree(matrixB_d);
		}

    } // end pragma openmp

    free(matrixA);
    free(matrixB);
    
    fclose(base_a);
    fclose(base_b);

    t2 = omp_get_wtime();

    int length_problem = atoi(argv[2]);
    printf("%d\t%f\n", (length_problem * 1000), (t2-t1));

    return 0;
}


// function to get the number of rows in a file
// input: reference for a open file || output: Number of rows in a file
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


// function to put every row in a file into a vector (that can be read as a matrix)
// input: reference for a open file; pointer to include the elements of file || output: None. The vector will be changed by reference
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
// input: TODO
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

// function to divide a matrix into a smaller matrix, given a lower edge and a upper edge.
// input: TODO
int *divide_matrix(int *original_matrix, int lower_edge, int upper_edge) {
    static int *final_matrix;
	int i, j = 0;

    final_matrix = (int *)malloc((upper_edge - lower_edge) * NCOL * sizeof(int));
    for (i = (lower_edge * NCOL); i < (upper_edge * NCOL); i++) {
        final_matrix[j] = original_matrix[i];
        j++;
    }
    return final_matrix;
}

// function to indicate edges for each PU according to the problem size
// input: TODO
int *get_pu_edges(int problem_size, int num_gpus, int percentage_each_gpu) {
    static int *edge_vector;
    int i, border = 0;
    float quantum_cpu, quantum_gpu, leftover = 0.0;   
    int percentage_cpu = 100 - (percentage_each_gpu * num_gpus);
    
    edge_vector = (int *)malloc((2 + (num_gpus * 2)) * sizeof(int));


    quantum_cpu = (percentage_cpu/100.0)*problem_size;
    quantum_gpu = (percentage_each_gpu/100.0)*problem_size;
    leftover = quantum_cpu - ((int)quantum_cpu);
    leftover += (quantum_gpu - ((int)quantum_gpu))*(num_gpus);

    if (percentage_cpu == 0) {
        edge_vector[0] = -1;
        edge_vector[1] = -1;
    }
    else {
        edge_vector[0] = border;
        border = border+(int)quantum_cpu;
	if ((int)leftover != 0){
		border++;
		leftover-= 1.0;
	}
        edge_vector[1] = border;
    }
	

    for (i = 2; i < (2 + num_gpus * 2); i = i + 2) {
        if(percentage_each_gpu == 0){
                border = -1;
        }

        edge_vector[i] = border;
	if ((int)leftover != 0){
                border++;
                leftover-= 1.0;
	}
        if(percentage_each_gpu == 0){
                border = -1;
        }

        border = border+(int)quantum_gpu;
        edge_vector[i + 1] = border;
    }

    return edge_vector;
}


void print_matrix(int *matrix, int nlines) {
    int i, j;

    for (i = 0; i < nlines; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrix[i * NCOL + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void multicore_execution(int *matrixA, int *matrixB, int nlines_b, int id_nested, int quantum, int leftover){
	int bloomA[100], bloomB[100], inicio, fim;
	if(id_nested < leftover){
		inicio = id_nested * (quantum +1);
		fim = quantum + 1 + inicio;
	}
	else{
		if(id_nested ==leftover){
			if(leftover == 0){
				inicio = id_nested * quantum;
				fim = inicio + quantum;
			}
			else{
				inicio = id_nested * (quantum +1);
				fim = inicio + quantum;
			}
		}
		else{
			if(leftover==0){
				inicio = id_nested * (quantum);
				fim = inicio + quantum;
			}
			else{
				inicio = id_nested * (quantum+1) - (id_nested-leftover);
				fim = inicio + quantum;
			}
		}
	}
//	printf("Thread aninhada de id: %d vai executar da linha %d até a linha %d\n",id_nested, inicio, fim);
	int i = inicio;
	while (i < fim) {
	        for (int j = 1; j < 101; j++) {
	            bloomA[j - 1] = matrixA[i * NCOL + j];
        	}
	        // getting bloom filter for each matrixB register
        	for (int k = 0; k < nlines_b; k++) {
        		for (int l = 1; l < 101; l++) {
		                bloomB[l - 1] = matrixB[k * NCOL + l];
            		}
	            //x = dice(bloomA, bloomB);
            		dice_multicore(bloomA, bloomB);
//	                if(x > y){
//	        	        vetor[i] = x;
//              		y = x;
//		        }
        	}	

		
		i++;
	}
	
}

// Kernel CUDA to compute linkage between matrixA and matrixB using a dice
// function as similarity measure
__global__ void kernel(int *matrixA, int *matrixB, int nlines_a, int nlines_b){
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int bloomA[100], bloomB[100];

    if (i < nlines_a) {
//    	y = 0;
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
            //x = dice(bloomA, bloomB);
            dice(bloomA, bloomB);
//            if(x > y){
//            	vetor[i] = x;
//            	y = x;
//            }
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
float dice_multicore(int *bloomA, int *bloomB) {
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
  
