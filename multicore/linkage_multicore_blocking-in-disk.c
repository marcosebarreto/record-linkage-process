/*
@(#)File:           $linkage_multicore_blocking_in_disk.c$
@(#)Last changed:   $Date: 2017/07/01 17:54:00 $
@(#)Purpose:        Probabilistic linkage on multicore using standard blocking

@(#)Authors:        Clicia Pinto
					Pedro Melo                    
                    Murilo Boratto 
@(#)Usage:         
 (*) Hotocompile:   gcc linkage_multicore_blocking_in_disk.c -o linkage_multicore_blocking_in_disk -fopenmp
 (*) Hotoexecute:  ./object						               <size file1>   <cores>
                   ./linkage_multicore_blocking_in_disk       1000           32


@(#)Comment: 


 (*)  INPUT BLOOM FILES SHOULD BE NAMED LIKE THE PATTER 
       base_10000BL1.bloom = First block from the original 10.000-line dataset
       
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define TAM_BLOOM 100
#define BLOCKS 10
#define SECONDARY_BASE_SIZE "100"

void divide(char[][256], char[], int); 
float getDice (char[], char[]);
int get_number_lines(FILE *);

int main(int argc, char *argv[]){

   	if( argc < 3 ) {
    	 printf("./%s [SIZE FILE 1] [CORES]\n", argv[0]);
    	 exit(-1);
   	}
    

	char buffer_line_p[255], lineSplitedP[2][256], buffer_line_q[255], lineSplitedQ[2][256];
	int id, nt, lines_p, lines_q, block_size;
	float dice;
	int NUM_THREADS = atoi(argv[2]);
	FILE *p[NUM_THREADS], *q[NUM_THREADS];
	char file1[30], file2[30], result_file_name[30], str_thread_id[10];

	double t1, t2;

	t1 = omp_get_wtime();
	block_size = atoi(argv[1]) / (int)BLOCKS;
	
	#pragma omp parallel num_threads(NUM_THREADS) private(id, nt, dice, lines_p, lines_q, buffer_line_p, lineSplitedP, buffer_line_q, lineSplitedQ, file1, file2, result_file_name, str_thread_id)
	{
		
		
		int count = 0, countq = 0;
		int inicio, fim, fracao, resto, aux=0;

		nt = omp_get_num_threads();
		id = omp_get_thread_num();
		FILE *result[id];

	    sprintf(str_thread_id, "%d", id);
	    strcpy(result_file_name, "result_dice_thread_");
	    strcat(result_file_name, str_thread_id);
	    strcat(result_file_name, ".dice");
//	    result[id] = fopen(result_file_name, "a");
	    

		for(int k=0; k<BLOCKS; k++){		

			char block_num[8];
			sprintf(block_num, "%d", k+1);

		    strcpy(file1, "blocks/base_");
			strcat(file1, argv[1]);
			strcat(file1, "_BL");
			strcat(file1, block_num);
			strcat(file1, ".bloom");

			strcpy(file2, "blocks/base_");
			strcat(file2, SECONDARY_BASE_SIZE);
			strcat(file2, "_BL");
			strcat(file2, block_num);
			strcat(file2, ".bloom");			

			p[id] = fopen(file1, "r");
			q[id] = fopen(file2, "r");

			lines_p = get_number_lines(p[id]);
			lines_q = get_number_lines(q[id]);

			fseek ( p[id] , 0 , SEEK_SET);
			fseek ( q[id] , 0 , SEEK_SET);


			resto = lines_p % NUM_THREADS;
			fracao = lines_p / NUM_THREADS;

			if(id < resto){
				inicio = id * (fracao+1);
				fim = inicio + (fracao+1);
			}
			else{
				inicio = (id * (fracao+1)) - (id - resto);
				fim = inicio + fracao;
			}
			while(aux < inicio){
				fgets(buffer_line_p, 255, p[id]);
				aux++;
			}

			int i;
			for( i=inicio; i<fim; i++){

				fgets(buffer_line_p, 255, p[id]);
				aux++;
				divide(lineSplitedP, buffer_line_p, id);				
				count++;
				countq = 0;

				while(fgets(buffer_line_q, 255, q[id]) != NULL){
					divide(lineSplitedQ, buffer_line_q, id);
					countq++;
					dice = getDice(lineSplitedP[1], lineSplitedQ[1]);
					//fprintf(result[id], "%f\n", dice);
					//printf("Dice%d\n", );

				}
				fseek ( q[id] , 0 , SEEK_SET);

			}
			fclose(p[id]);
			fclose(q[id]);
			
		}
	}
	t2 = omp_get_wtime();

	printf("%d\t%f\n",atoi(argv[1]), t2-t1);

	return 0;

}/*main*/


void divide(char m[][256], char v[],  int ident){
	char c;
	int i = 0, j=0;
	c = v[0];
	do{
		m[0][i] = (char) c;
		i++;
		c = v[i];
	}while(c != ';');
	m[0][i] = '\0';

	//skipping ";""
	i++;
	c = v[i];

	do{
		m[1][j] = (char) c;
		i++;
		c = v[i];
		j++;
	}while(c!='\n');
	m[1][j] = '\0';
}

float getDice (char vP[], char vQ[]){
	float a=0, b=0, h=0;
	int i;
	float dice;

	for(i=0; i<TAM_BLOOM; i++){
		if (vP[i] == '1'){
			a++;
			if(vQ[i] == '1') h++;
		}
		if(vQ[i] == '1'){
			b++;
		}
	}
	dice = ((h*2.0) / (a+b)) *10000;
	//printf("Dice: %f\n", dice);
	return dice;

}

int get_number_lines(FILE * arquivo){
  char buffer[255];
  int count=0;
  while(fgets(buffer,255,arquivo) != NULL){
    count++;
  }
  return count;

}