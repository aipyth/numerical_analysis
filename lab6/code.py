import numpy as np
import pandas
import matplotlib.pyplot as plt


def solve_runge_kutta(f, x, y, h, n):
    y_arr = [y]
    x_arr = [x]
    for _ in range(0, n):
        k_1 = h*f(x, y)
        k_2 = h*f(x + 1/2*h, y + 1/2*k_1)
        k_3 = h*f(x + 1/2*h, y + 1/2*k_2)
        k_4 = h*f(x+h, y + k_3)
        y = y + 1/6*(k_1 + 2*k_2 + 2*k_3 + k_4)
        x = x + h
        y_arr.append(y)
        x_arr.append(x)
    return x_arr, y_arr


def solve_adams_bushforth(f, x_0, y_0, h, n):
    x, y = solve_runge_kutta(f, x_0, y_0, h, 3)
    for i in range(3, n):
        y.append(
            y[i] + h*(55*f(x[i], y[i]) - 59*f(x[i-1], y[i-1]) +
                37*f(x[i-2], y[i-2]) - 9*f(x[i-3], y[i-3]))/24
        )
        x.append(x[i] + h)
    return x, y


def main():
    def f(x, y):
        return (1 - x**2) * y + (x**3 - x + 1)*np.cos(x) - x * np.sin(x)
    h = 0.1
    n = 30
    exact = lambda x: x * np.cos(x)

    x, y1 = solve_runge_kutta(f, 0, 0, h, n)
    x, y2 = solve_adams_bushforth(f, 0, 0, h, n)
    f_x = list(map(exact, x))

    print("Function values")
    table = [
        [x[i], f_x[i], y1[i], y2[i]] for i in range(0, n)]
    print(pandas.DataFrame(table, columns=['x', 'f(x)', 'Runge-Kutta',
                           'Adams-Bushforth']), '\n')

    err1 = [np.absolute(y1[i] - f_x[i]) for i in range(0, n+1)]
    err2 = [np.absolute(y2[i] - f_x[i]) for i in range(0, n+1)]

    fig, ax = plt.subplots(figsize=(10, 10))
    xx = [0.01 * k for k in range(1, 10*n)]
    ax.plot(xx, list(map(exact, xx)))
    ax.plot(x, y1)
    ax.plot(x, y2)
    ax.set_title("Numerical solution and exact solution plot")
    fig.savefig('solution.png')

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.plot(x, err1, label="Runge-Kutta")
    ax.plot(x, err2, label="Adams-Bushforth")
    ax.set_title("Error plot of two numerical solutions")
    ax.legend()
    fig.savefig('errors.png')


if __name__ == "__main__":
    main()
