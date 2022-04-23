import numpy as np
import matplotlib.pyplot as plt

a = -2
b = 3.2
x = np.linspace(a, b, num=1000)
y = -x**4 + 3*x**3 - 2*x + 4

plt.plot(x, y, 'r')
plt.grid()
plt.axhline()
plt.show()
