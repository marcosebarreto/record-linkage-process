#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define TAM_BLOOM 100
//#define NUM_THREADS 12


void divide(char[][256], char[], int); //eliminar o parametro int que representa a id, pois não é usado

float getDice (char[], char[]);

int get_number_lines(FILE *);

int main(int argc, char *argv[]){
	char buffer_line_p[255], lineSplitedP[2][256], buffer_line_q[255], lineSplitedQ[2][256];
	int id, nt, lines_p, lines_q;
	float dice;
	double t1, t2;
	int NUM_THREADS = atoi(argv[2]);
	FILE *p[NUM_THREADS], *q[NUM_THREADS];


	t1 = omp_get_wtime();
	#pragma omp parallel num_threads(NUM_THREADS) private(id, nt, dice, lines_p, lines_q, buffer_line_p, lineSplitedP, buffer_line_q, lineSplitedQ)
	{
		int count = 0, countq = 0;
		int inicio, fim, fracao, resto, aux=0;

		nt = omp_get_num_threads();
		id = omp_get_thread_num();


		p[id] = fopen("base_1M.bloom", "r");
		q[id] = fopen("base_100k.bloom", "r");

		//Isolar esse procedimento para só ser feito por uma thread: a que chegar primeiro
//		lines_p = get_number_lines(p[id]);
		lines_p = atoi(argv[1]);
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
/*		printf("Thread %d prendeu a aux até aux = %d\n", id, aux);
		printf("THREAD %d -- Arquivo com %d linhas -- inicio na linha %d e fim na linha %d \n", id, lines_p, inicio, fim);
*/
		int i;
		for( i=inicio; i<fim; i++){

			fgets(buffer_line_p, 255, p[id]);
			aux++;
			divide(lineSplitedP, buffer_line_p, id);

			//printf("Iteração externa %d da thread %d -- Tamanho: %ld e %ld -- aux: %d\n", count, id, strlen(lineSplitedP[0]), strlen(lineSplitedP[1]), aux-1);
			count++;
			countq = 0;

			while(fgets(buffer_line_q, 255, q[id]) != NULL){
				divide(lineSplitedQ, buffer_line_q, id);
				countq++;
				dice = getDice(lineSplitedP[1], lineSplitedQ[1]);


			}
			fseek ( q[id] , 0 , SEEK_SET);

		}
		fclose(p[id]);
		fclose(q[id]);
	}
	t2 = omp_get_wtime();
	printf("%d\t%f\n",atoi(argv[1]), t2-t1);


	return 0;

}

//Função testada
void divide(char m[][256], char v[],  int ident){
	char c;
	int i = 0, j=0;
	//printf("Mensagem da Função reordena: Recebido da thread: %d a matriz no endereço: %p e o vetor no endereço %p\n", ident, m, v);
	c = v[0];
	do{
		m[0][i] = (char) c;
		i++;
		c = v[i];
	}while(c != ';');
	m[0][i] = '\0';

	//pulando o ;
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
