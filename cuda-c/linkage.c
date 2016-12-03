// Linkage Process using CUDA C API
// Authors: Clicia Santos Pinto and Pedro Marcelino Mendes Novaes Melo

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define NCOL 101

int main(int argc, char const *argv[]) {
    FILE *base_a, *base_b;
    char line[256];
    int nlines_a = 0, nlines_b = 0, pos_to_insert = 0;

    // opening large base (base_a) and small base (base_b)
    base_a = fopen("base_a10.bloom", "r");
    base_b = fopen("base_b10.bloom", "r");

    // ------------------------------ OPERATIONS WITH BASE A ------------------------------ //
    // getting line quantity
    fgets(line, 255, base_a);
    while (!feof(base_a)) {
        nlines_a++;
        fgets(line, 255, base_a);
    }
    int matrixA[nlines_a][NCOL];

    // getting line by line to insert into matrixA
    printf("base_a\n");
    rewind(base_a);
    fgets(line, 255, base_a);
    while (!feof(base_a)) {
        line[strlen(line) - 1] = '\0';
        // fill_matrix(matrixA, pos_to_insert, line);
        printf("%s\n", line);
        pos_to_insert++;
        fgets(line, 255, base_a);
    }

    // ------------------------------ OPERATIONS WITH BASE B ------------------------------ //
    pos_to_insert = 0;

    // getting line quantity
    fgets(line, 255, base_b);
    while (!feof(base_b)) {
        nlines_b++;
        fgets(line, 255, base_b);
    }
    int matrixB[nlines_b][NCOL];

    // getting line by line to inset into matrixB
    printf("base_b\n");
    rewind(base_b);
    fgets(line, 255, base_b);
    while (!feof(base_b)) {
        line[strlen(line) - 1] = '\0';
        // fill_matrix(matrixB, pos_to_insert, line);
        printf("%s\n", line);
        pos_to_insert++;
        fgets(line, 255, base_b);
    }

    fclose(base_a);
    fclose(base_b);

    return 0;
}
