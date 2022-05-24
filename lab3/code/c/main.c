#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#include "matrix.h"

double* sqrt_method(double** A, double* b, uint16_t size) {

}




int main(void) {
    // {
    //     {2.12, 0.42, 1.34, 0.88},
    //     {0.42, 3.95, 1.87, 0.43},
    //     {1.34, 1.87, 2.98, 0.46},
    //     {0.88, 0.43, 0.46, 4.44},
    // };
    // { 11.172, 0.115, 0.009, 9.349 };

    long double Araw[4][4] = {
        {2.12, 0.42, 1.34, 0.88},
        {0.42, 3.95, 1.87, 0.43},
        {1.34, 1.87, 2.98, 0.46},
        {0.88, 0.43, 0.46, 4.44},
    };

    printf("araw %x\n", Araw);
    printf("araw0 %x\n", Araw[0]);
    printf("araw00 %i\n", Araw[0][0]);

    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            printf("%Lf ", Araw[i][j]);
        }
        printf("\n");
    }
    
    matrix* M = matrix_create_from_const(&Araw, 4, 4);

    printf("created matrix\n");

    matrix_print(M);

    return 0;
}
