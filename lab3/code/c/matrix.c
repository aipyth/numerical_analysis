#include "matrix.h"


matrix* matrix_init(uint64_t isize, uint32_t jsize) {
    matrix* M = malloc(sizeof(matrix));
    M->isize = isize; M->jsize = jsize;
    M->data = (long double**)malloc(sizeof(long double*) * isize);
    printf("allocated matrix\n");
    for (uint32_t i = 0; i < isize; i++) {
        printf("allocating %i\n", i);
        M->data[i] = (long double*)malloc(sizeof(long double) * jsize);
    }
    return M;
}

matrix* matrix_create_from_const(long double*** m, uint32_t isize, uint32_t jsize) {
    matrix* M = matrix_init(isize, jsize);
    printf("matrix initialized\n");
    for (uint32_t i = 0; i < isize; i++) {
        printf("%x\n", m[i]);
        for (uint32_t j = 0; j < jsize; j++) {
            M->data[i][j] = (*m)[i][j];
        }
    }
    return M;
}

void matrix_print(matrix const* M) {
    printf("--\n");
    for (uint16_t i = 0; i < M->isize; i++) {
        printf("| ");
        for (uint16_t j = 0; j < M->jsize; j++) {
            printf("%.6Lf  ", M->data[i][j]);
        }
        printf(" |\n");
    }
    printf("--\n");
}
