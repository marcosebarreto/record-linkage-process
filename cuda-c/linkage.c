// Linkage Process using CUDA C API
// Authors: Clicia Santos Pinto and Pedro Marcelino Mendes Novaes Melo

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCOL 101

void fill_matrix(int matrix[][NCOL], int pos, char *line);
int get_num_of_lines(FILE *fp);

int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    char line[256];
    int nlines_a = 0, nlines_b = 0, pos_to_insert = 0, i, j;

    // opening large base (base_a) and small base (base_b)
    base_a = fopen("base_a10.bloom", "r");
    base_b = fopen("base_b10.bloom", "r");

    // ------------------------------ OPERATIONS WITH BASE A ------------------------------ //
    // getting line quantity
    nlines_a = get_num_of_lines(base_a);
    int matrixA[nlines_a][NCOL];

    // getting line by line to insert into matrixA
    printf("base_a\n");
    rewind(base_a);
    fgets(line, 255, base_a);
    while (!feof(base_a)) {
        line[strlen(line) - 1] = '\0';
        fill_matrix(matrixA, pos_to_insert, line);
        // printf("%s\n", line);
        pos_to_insert++;
        fgets(line, 255, base_a);
    }

    for (i = 0; i < nlines_a; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrixA[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    // ------------------------------ OPERATIONS WITH BASE B ------------------------------ //
    pos_to_insert = 0;

    // getting line quantity
    nlines_b = get_num_of_lines(base_b);
    int matrixB[nlines_b][NCOL];

    // getting line by line to inset into matrixB
    printf("base_b\n");
    rewind(base_b);
    fgets(line, 255, base_b);
    while (!feof(base_b)) {
        line[strlen(line) - 1] = '\0';
        fill_matrix(matrixB, pos_to_insert, line);
        // printf("%s\n", line);
        pos_to_insert++;
        fgets(line, 255, base_b);
    }

    for (i = 0; i < nlines_b; i++) {
        for (j = 0; j < NCOL; j++) {
            printf("%d", matrixB[i][j]);
        }
        printf("\n");
    }
    printf("\n");

    fclose(base_a);
    fclose(base_b);

    return 0;
}

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

int get_num_of_lines(FILE *fp) {
    int lines = 0;
    char line[256];

    fgets(line, 255, fp);
    while (!feof(fp)) {
        lines++;
        fgets(line, 255, fp);
    }

    printf("num lines: %d\n", lines);
    return lines;
}
