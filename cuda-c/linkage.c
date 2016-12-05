// Linkage Process using CUDA C API
// Authors: Clicia Santos Pinto and Pedro Marcelino Mendes Novaes Melo

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCOL 101


void fill_matrix(int *matrix, int pos, char *line);
int get_num_of_lines(FILE *fp);
void process_file(FILE *fp, int *matrix);
void print_matrix(int *matrix, int nlines);


int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    int nlines_a = 0, nlines_b = 0, i, j;

    // opening large base (base_a) and small base (base_b)
    base_a = fopen("base_a10.bloom", "r");
    base_b = fopen("base_b10.bloom", "r");

    // ------- OPERATIONS WITH BASE A ------- //
    // getting line quantity
    nlines_a = get_num_of_lines(base_a);
    int *matrixA = (int *)malloc(nlines_a * NCOL * sizeof(int));

    // processing base_a to fill matrixA
    printf("base_a\n");
    process_file(base_a, matrixA);
    print_matrix(matrixA, nlines_a);

    // ------- OPERATIONS WITH BASE B ------- //
    // getting line quantity
    nlines_b = get_num_of_lines(base_b);
    // int matrixB[nlines_b][NCOL];
    int *matrixB = (int *)malloc(nlines_b * NCOL * sizeof(int));

    printf("base_b\n");
    process_file(base_b, matrixB);
    print_matrix(matrixB, nlines_b);


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

    for (i = 0; i < nlines; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrix[i * NCOL + j]);
        }
        printf("\n");
    }
    printf("\n");
}
