#include <stdio.h>
#include <math.h>

double bisection(double func(double), double a, double b, double eps) {
    unsigned int iteration = 0;
    double* to_change;
    while (fabs(a - b) > eps) {
        double c = (a + b) / 2;
        double a_value = func(a);
        double b_value = func(b);
        double c_value = func(c);
        if (a_value * c_value < 0) {
            to_change = &b;
        } else if (b_value * c_value < 0) {
            to_change = &a;
        }
        iteration++;
        printf("%2d: a = %.5f -> %.5f | b = %.5f -> %.5f | c = %.5f -> %.5f | abs(a - b) <= eps => %i \n", iteration,
                a, a_value,
                b, b_value,
                c, c_value,
                fabs(a - b) <= eps);
        *to_change = c;
    }
    printf("result %f in %d iterations \n", (a + b)/2, iteration);
    return (b + a)/2;
}

typedef double func(double);

double target(double x) {
    /* return -pow(x, 4) + 3 * pow(x, 3) - 2 * x + 4; */
    return x*(-2 + x*x*(3 - x)) + 4;
}

int main(void) {
    printf("bisection result %f\n", bisection(target, -2.58, -0.52, pow(10, -5)));
    return 0;
}
