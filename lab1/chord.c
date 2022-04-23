#include <stdio.h>
#include <math.h>

double chord(double func(double), double a, double b, double eps) {
    unsigned int iteration = 0;
    double *to_change;
    double c_prev = a;
    double c;
    while (1) {
        double a_value = func(a);
        double b_value = func(b);

        c = (a * b_value - b * a_value) / (b_value - a_value);
        double c_value = func(c);
        if (a_value * c_value < 0) {
            to_change = &b;
        } else if (b_value * c_value < 0) {
            to_change = &a;
        }
        iteration++;
        printf("%2d: a = %.5f -> %.5f | b = %.5f -> %.5f | c = %.5f -> %.5f | abs(c - c_prev) < eps || |f(c)| < eps => %i \n", iteration,
                a, a_value,
                b, b_value,
                c, c_value,
                (fabs(c - c_prev) < eps || fabs(c_value) < eps));
        *to_change = c;
        if (fabs(c - c_prev) < eps || fabs(c_value) < eps) {
            break;
        }
    }
    printf("result %f in %d iterations \n", (a + b)/2, iteration);
    return c;
}

typedef double func(double);

double target(double x) {
    /* return -pow(x, 4) + 3 * pow(x, 3) - 2 * x + 4; */
    return x*(-2 + x*x*(3 - x)) + 4;
}

int main(void) {
    printf("chord result %f\n", chord(target, 1, 4, pow(10, -5)));
    return 0;
}
