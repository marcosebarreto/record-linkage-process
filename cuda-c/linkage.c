// Linkage Process using CUDA C API
// Authors: Clicia Santos Pinto and Pedro Marcelino Mendes Novaes Melo

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCOL 101


void fill_matrix(int matrix[][NCOL], int pos, char *line);
int get_num_of_lines(FILE *fp);
void process_file(FILE *fp, int matrix[][NCOL]);
void print_matrix(int matrix[][NCOL], int nlines);


int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    int nlines_a = 0, nlines_b = 0, i, j;

    // opening large base (base_a) and small base (base_b)
    base_a = fopen("base_a10.bloom", "r");
    base_b = fopen("base_b10.bloom", "r");

    // ------- OPERATIONS WITH BASE A ------- //
    // getting line quantity
    nlines_a = get_num_of_lines(base_a);
    int matrixA[nlines_a][NCOL];

    // processing base_a to fill matrixA
    printf("base_a\n");
    process_file(base_a, matrixA);
    print_matrix(matrixA, nlines_a);

    // ------- OPERATIONS WITH BASE B ------- //
    // getting line quantity
    nlines_b = get_num_of_lines(base_b);
    int matrixB[nlines_b][NCOL];

    printf("base_b\n");
    process_file(base_b, matrixB);
    print_matrix(matrixB, nlines_b);

    fclose(base_a);
    fclose(base_b);

    return 0;
}


// function to split a line and to insert the elements in matrix
void fill_matrix(int matrix[][NCOL], int pos, char *line) {
    int i, j = 0, before_v = 1;
    char c, id[10];

    for (i = 0; i < strlen(line); i++) {
        c = line[i];
        if (before_v == 1) {
            if (c == ';') {
                id[i] = '\0';
                matrix[pos][j] = atoi(id);
                j++;
                before_v = 0;
            }
            else {
                id[i] = c;
            }
        }
        else {
            matrix[pos][j] = line[i] - '0';
            j++;
        }
    }
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
void process_file(FILE *fp, int matrix[][NCOL]) {
    char line[256];
    int pos_to_insert = 0;

    rewind(fp);

    // getting line by line to insert into matrix
    fgets(line, 255, fp);
    while (!feof(fp)) {
        line[strlen(line) - 1] = '\0';
        fill_matrix(matrix, pos_to_insert, line);
        // printf("%s\n", line);
        pos_to_insert++;
        fgets(line, 255, fp);
    }
}


void print_matrix(int matrix[][NCOL], int nlines) {
    int i, j;

    for (i = 0; i < nlines; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrix[i][j]);
        }
        printf("\n");
    }
    printf("\n");
}
