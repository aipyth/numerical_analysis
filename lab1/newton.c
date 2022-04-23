#include <stdio.h>
#include <math.h>

double newton(double func(double), double der(double), double a, double b, double eps) {
    unsigned int iteration = 0;
    double a_value = func(a);
    double b_value = func(b);
    double c_prev = a;
    double c;
    while (1) {
        double c_prev_value = func(c_prev);

        c = c_prev - c_prev_value / der(c_prev);
        if (c > b) {
            printf("%f out of bounds\n", c);
            c_prev = b;
            c_prev_value = func(c_prev);
            c = c_prev - c_prev_value / der(c_prev);
        }
        double c_value = func(c);
        iteration++;
        printf("%2d: a = %.5f -> %.5f | b = %.5f -> %.5f | c = %.5f -> %.5f | c_prev = %.5f -> %.5f| abs(c - c_prev) < eps || |f(c)| < eps => %i \n", iteration,
                a, a_value,
                b, b_value,
                c, c_value,
                c_prev, c_prev_value,
                (fabs(c - c_prev) < eps || fabs(c_value) < eps));
        if (fabs(c - c_prev) < eps || fabs(c_value) < eps) {
            break;
        } else {
            c_prev = c;
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

double derivative(double x) {
    return pow(x, 2) * (9 - 4 * x) - 2;
}

int main(void) {
    printf("newton result %f\n", newton(target, derivative, -2.58, -0.52, pow(10, -5)));
    return 0;
}
