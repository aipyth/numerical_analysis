#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    uint32_t isize;
    uint32_t jsize;
    long double** data;
} matrix;

matrix* matrix_init(uint64_t isize, uint32_t jsize);

matrix* matrix_create_from_const(long double*** m, uint32_t isize, uint32_t jsize);

void matrix_print(matrix const* M);
