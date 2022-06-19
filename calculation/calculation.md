::: {#e5f47033-3a0f-4ad4-9498-63905b79a45d .cell .markdown}
# Методи обчислень. Розрахунково-графічна робота. {#методи-обчислень-розрахунково-графічна-робота}

**Варіант 4** *Змоделювати процес розповсюдження газоподібної
забруднюючої домішки в атмосфері за умови наявності ефекту самоочищення
середовища.\
Самоочищення виникає при досягненні концентрацією $u$ деякого критичного
значення.*

*Рівняння еволюції концентрації домішки має вигляд:*

$$\frac{\partial u}{\partial t} = k_1 \frac{\partial^2 u}{\partial x^2} + k_2 \frac{\partial^2 u}{\partial y^2} + \mathcal{X} + \varphi$$

$$\mathcal{X} = \begin{cases} 0 &\quad u < u_{cr} \\ -du &\quad u \geq u_{cr} \end{cases}$$

$$u(0) = 0$$

$$u \mid_{\Gamma} = 0$$

*Прийняти, що джерела забруднення розміщені в двох несусідніх
центральних точках області, і діють з інтенсивністю
$\varphi = 5\text{мкг/хв}$. Розміри області завдати самостійно (але
розбиття області повинно включати не менше 10 вузлів за кожним
напрямком). Знайти поле концентрації домішки на протязі часу, за який
концентрація домішки встигне досягти свого критичного значення хоча б в
деяких точках області.*
:::

::: {#b66050a3-f8da-4e6f-83c7-5d95b2a6ff36 .cell .markdown}
$$t \in [0, \infty]$$

Нехай маємо границі по осі $x$: $[a_x, b_x]$, та по осі $y$:
$[a_y, b_y]$. Тоді:

$$u\mid_{\Gamma} = 0 \; \Rightarrow \; u(a, b) = 0 \quad \forall a \in \{a_x, a_y\}, \forall b \in \{b_x, b_y\}$$
:::

::: {#3beccbd6-52c9-44b1-b568-a843033e194a .cell .markdown tags="[]"}
## Дискретизація

Рівняння має параболічний тип. Будемо використовувати неявну схему
Кранка-Ніколсона:

-   розбиваємо відрізки $[a_x, b_x]$ та $[a_y, b_y]$ на $n+1$ вузлів
-   $i$ позначатимемо номер вузла по $[a_x, b_x]$, а $j$ по $[a_y, b_y]$
-   відстань між сусідніми вузлами $\Delta x$ та $\Delta y$
-   дискретизуємо час $t$: $t_h \in [0, m]$ та індекс
    $h \in \mathbb{N}_0, h = \overline{0, l}$. крок $\Delta t$
-   джерело забруднення: нехай маємо дві пари $(s_1, d_1)$ та
    $(s_2, d_2)$, такі що
    -   $s_2 - s_1 > 1 \;\land\; d_2 - d_1 > 1$ - умова несусідства тоді
        $$
          \varphi = \begin{cases}
          0 &\; (i,j) \not= (s_1, d_1) \;\land\; (i,j) \not= (s_2, d_2) \\
          5 &\; (i,j) =     (s_1, d_1) \;\lor\;  (i,j) =     (s_2, d_2)
          \end{cases}
          $$

Беремо скінченно-різницеві апроксимації:

$$\frac{\partial u}{\partial t} = \frac{u_{i,j}^{h + 1} - u_{i,j}^{h}}{\Delta t}$$

$$\frac{\partial^2 u}{\partial x^2} = 
\lambda \frac{u_{i+1,j}^{h+1} - 2 u_{i,j}^{h+1} + u_{i-1,j}^{h+1}}{\Delta x^2} +
(1-\lambda) \frac{u_{i+1,j}^{h} - 2 u_{i,j}^{h} + u_{i-1,j}^{h}}{\Delta x^2}$$

$$
\frac{\partial^2 u}{\partial y^2} = 
\lambda \frac{u_{i,j+1}^{h+1} - 2 u_{i,j}^{h+1} + u_{i,j-1}^{h+1}}{\Delta x^2} +
(1-\lambda) \frac{u_{i,j+1}^{h} - 2 u_{i,j}^{h} + u_{i,j-1}^{h}}{\Delta y^2}
$$

$$
\frac{\partial u}{\partial x} = \lambda \frac{u_{i+1,j}^{h+1} - u_{i,j}^{h+1}}{\Delta x} +
(1-\lambda) \frac{u_{i+1,j}^{h} - u_{i,j}^{h}}{\Delta x} 
$$ $$
\frac{\partial u}{\partial y} = \lambda \frac{u_{i,j+1}^{h+1} - u_{i,j}^{h+1}}{\Delta y} +
(1-\lambda) \frac{u_{i,j+1}^{h} - u_{i,j}^{h}}{\Delta y}
$$ $$\lambda \in (0,1)$$
:::

::: {#a542fc35-2c59-41b2-b9ff-a37d7d321585 .cell .markdown}
### Різницева задача

$$\frac{u_{i,j}^{h + 1} - u_{i,j}^{h}}{\Delta t} = 
k_1 \left( \lambda \frac{u_{i+1,j}^{h+1} - 2 u_{i,j}^{h+1} + u_{i-1,j}^{h+1}}{\Delta x^2} +
(1-\lambda) \frac{u_{i+1,j}^{h} - 2 u_{i,j}^{h} + u_{i-1,j}^{h}}{\Delta x^2} \right) +$$
$$+ k_2 \left( \lambda \frac{u_{i,j+1}^{h+1} - 2 u_{i,j}^{h+1} + u_{i,j-1}^{h+1}}{\Delta y^2} +
(1-\lambda) \frac{u_{i,j+1}^{h} - 2 u_{i,j}^{h} + u_{i,j-1}^{h}}{\Delta y^2} \right) +
\mathcal{X} + \varphi$$

$$\mathcal{X} = \begin{cases}
0 &\quad u < u_{cr} \\
- \left(
  \lambda\left( u_{i+1,j}^{h+1} + u_{i,j+1}^{h+1} -2 u_{i,j}^{h+1} \right) +
  (1-\lambda)\left( u_{i+1,j}^{h} + u_{i,j+1}^{h} -2 u_{i,j}^{h} \right) +
  u_{i,j}^{h+1} - u_{i,j}^{h}
  \right) &\quad u \geq u_{cr}
\end{cases}$$

$\exists (s_1,d_1) \text{ and } \exists (s_2,d_2) : \quad s_2 - s_1 > 1 \;\land\; d_2 - d_1 > 1$

$$\varphi = \begin{cases}
    0 &\; (i,j) \not= (s_1, d_1) \; \land \; (i,j) \not= (s_2, d_2) \\
    5 &\; (i,j) = (s_1, d_1) \; \lor  \;  (i,j) =     (s_2, d_2)
\end{cases}$$

З початковими: $$\forall i,j \quad  u_{i,j}^{0} = 0$$

З граничними:
$$\forall h :\; \forall i : u_{i,0}^{h} = u_{i,n}^{h} = 0$$
$$\forall h :\; \forall j : u_{0,j}^{h} = u_{n,j}^{h} = 0$$
:::

::: {#22b0445f-1566-4e51-b8b7-47e278bd82a5 .cell .markdown}
Перенесемо всі значення $u$ у момент $h+1$ у ліву частину рівняння:
$$\frac{1}{\Delta t}u_{i,j}^{h+1} - k_1\frac{\lambda}{\Delta x^2} \left(
    u_{i+1,j}^{h+1} - 2u_{i,j}^{h+1} + u_{i-1,j}^{h+1}
\right) - k_2 \frac{\lambda}{\Delta y^2} \left(
    u_{i,j+1}^{h+1} - 2u_{i,j}^{h+1} + u_{i,j-1}^{h+1}
\right)
=$$

$$ = \frac{1}{\Delta t}u_{i,j}^{h} +
k_1 \frac{(1-\lambda)}{\Delta x^2} \left(
    u_{i+1,j}^{h} - 2u_{i,j}^{h} + u_{i-1,j}^{h}
\right) +
k_2 \frac{(1-\lambda)}{\Delta y^2} \left(
    u_{i,j+1}^{h} - 2u_{i,j}^{h} + u_{i,j-1}^{h}
\right) + \mathcal{X} + \varphi$$
:::

::: {#b6e467dc-1b75-4987-beed-7e6d99befd11 .cell .markdown}
Але маємо два випадки у залежності від $u$:

якщо $u < u_{cr}$, то
$$\frac{1}{\Delta t}u_{i,j}^{h+1} - k_1\frac{\lambda}{\Delta x^2} \left(
    u_{i+1,j}^{h+1} - 2u_{i,j}^{h+1} + u_{i-1,j}^{h+1}
\right) - k_2 \frac{\lambda}{\Delta y^2} \left(
    u_{i,j+1}^{h+1} - 2u_{i,j}^{h+1} + u_{i,j-1}^{h+1}
\right)
=$$

$$ = \frac{1}{\Delta t}u_{i,j}^{h} + k_1 \frac{(1-\lambda)}{\Delta x^2} \left(
    u_{i+1,j}^{h} - 2u_{i,j}^{h} + u_{i-1,j}^{h}
\right) + k_2 \frac{(1-\lambda)}{\Delta y^2} \left(
    u_{i,j+1}^{h} - 2u_{i,j}^{h} + u_{i,j-1}^{h}
\right)
+ \varphi$$
:::

::: {#4fd9816b-e195-4a26-ba86-3bee4c4a9c45 .cell .markdown}
$$\left( - \frac{k_1\lambda}{\Delta x^2} \right)u_{i+1,j}^{h+1} +
  \left( - \frac{k_2\lambda}{\Delta y^2} \right)u_{i,j+1}^{h+1} +
  \left( \frac{1}{\Delta t} + \frac{2 k_1 \lambda}{\Delta x^2} +
         \frac{2 k_2 \lambda}{\Delta y^2}\right) u_{i,j}^{h+1} +
  \left( - \frac{k_1\lambda}{\Delta x^2} \right) u_{i-1,j}^{h+1} +
  \left( - \frac{k_2\lambda}{\Delta y^2} \right) u_{i,j-1}^{h+1}
  =$$

$$=
  \left( \frac{k_1(1-\lambda)}{\Delta x^2} \right) u_{i+1,j}^{h} +
  \left( \frac{k_2(1-\lambda)}{\Delta y^2} \right) u_{i,j+1}^{h} +
  \left( \frac{1}{\Delta t} -
         \frac{2 k_1 (1-\lambda)}{\Delta x^2} -
         \frac{2 k_2 (1-\lambda)}{\Delta y^2} \right) u_{i,j}^{h} +
$$

$$ + \left( \frac{k_1 (1-\lambda)}{\Delta x^2} \right) u_{i-1,j}^{h} +
  \left( \frac{k_2 (1-\lambda)}{\Delta y^2} \right) u_{i,j-1}^{h}
  + \varphi$$
:::

::: {#55c2b2b4-e5cf-4b40-b7d4-46ea7de9ca9c .cell .markdown}
а якщо $u \geq u_{cr}$:

$$\frac{1}{\Delta t}u_{i,j}^{h+1} - k_1\frac{\lambda}{\Delta x^2} \left(
    u_{i+1,j}^{h+1} - 2u_{i,j}^{h+1} + u_{i-1,j}^{h+1}
\right) - k_2 \frac{\lambda}{\Delta y^2} \left(
    u_{i,j+1}^{h+1} - 2u_{i,j}^{h+1} + u_{i,j-1}^{h+1}
\right)
=$$

$$=
\frac{1}{\Delta t}u_{i,j}^{h} +
k_1 \frac{(1-\lambda)}{\Delta x^2} \left(
    u_{i+1,j}^{h} - 2u_{i,j}^{h} + u_{i-1,j}^{h}
\right) +
k_2 \frac{(1-\lambda)}{\Delta y^2} \left(
    u_{i,j+1}^{h} - 2u_{i,j}^{h} + u_{i,j-1}^{h}
\right) - $$

$$
- \left(
  \lambda\left( u_{i+1,j}^{h+1} + u_{i,j+1}^{h+1} -2 u_{i,j}^{h+1} \right) +
  (1-\lambda)\left( u_{i+1,j}^{h} + u_{i,j+1}^{h} -2 u_{i,j}^{h} \right) +
  u_{i,j}^{h+1} - u_{i,j}^{h}
  \right) $$
:::

::: {#c35637a0-b1ba-47ad-a46b-01cd09095a6b .cell .markdown}
$$\left( - \frac{k_1\lambda}{\Delta x^2} + \lambda \right)            u_{i+1,j}^{h+1} +
  \left( - \frac{k_2\lambda}{\Delta y^2} + \lambda \right)            u_{i,j+1}^{h+1} +
  \left( \frac{1}{\Delta t} + \frac{2 k_1 \lambda}{\Delta x^2} +
         \frac{2 k_2 \lambda}{\Delta y^2} - 2 \lambda + 1 \right)     u_{i,j}^{h+1} +
  \left( - \frac{k_1\lambda}{\Delta x^2} \right)                      u_{i-1,j}^{h+1} +
  \left( - \frac{k_2\lambda}{\Delta y^2} \right)                      u_{i,j-1}^{h+1}
  =$$

$$=
  \left( \frac{k_1(1-\lambda)}{\Delta x^2} - 1 + \lambda \right)      u_{i+1,j}^{h} +
  \left( \frac{k_2(1-\lambda)}{\Delta y^2} - 1 + \lambda \right)      u_{i,j+1}^{h} +
  \left( \frac{1}{\Delta t} -
         \frac{2 k_1 (1-\lambda)}{\Delta x^2} -
         \frac{2 k_2 (1-\lambda)}{\Delta y^2} + 2 - 2 \lambda + 1 \right) u_{i,j}^{h} +
$$

$$ + \left( \frac{k_1 (1-\lambda)}{\Delta x^2} \right)                       u_{i-1,j}^{h} +
  \left( \frac{k_2 (1-\lambda)}{\Delta y^2} \right)                       u_{i,j-1}^{h}
  + \varphi$$
:::

::: {#7fc8febe-b8e9-42d2-b6a4-77b90c139328 .cell .code execution_count="17"}
``` python
import numpy as np
import itertools
```
:::

::: {#24afb13b-4877-4261-923f-05fd2edc2929 .cell .code execution_count="29"}
``` python
class DiffusionDiffEquationSolver:
    emitters = []
    area = {}
    history = {}
    current_time = 0
    lattice = None
    
    
    def __init__(self, k_1, k_2, l,
                 n, x, y, t_step,
                 u_crit,
                 emitters, phi,
                 initial):
        self.k_1 = k_1
        self.k_2 = k_2
        self.l = l
        self.n = n

        self.area['x'] = x
        self.area['y'] = y 
        self.nodesx = np.linspace(self.area['x'][0], self.area['x'][1], n)
        self.nodesy = np.linspace(self.area['y'][0], self.area['y'][1], n)
        
        self.nodes = np.array(list(itertools.product([i for i in range(n)], repeat=2)))

        self.x_step = self.nodesx[1] - self.nodesx[0]
        self.y_step = self.nodesy[1] - self.nodesy[0]
        self.t_step = t_step

        self.u_crit = u_crit
        self.phi = phi
        self.emitters = emitters
        # unitialize lattice
        self.lattice = np.zeros((n, n))
        for i in range(self.lattice.shape[0]):
            for j in range(self.lattice.shape[1]):
                self.lattice[i, j] = initial(i, j)
                

    def _build_discrete_eq_coefs_next(self, i, j):
        coefs = np.array([
            [- self.k_1 * self.l / np.power(self.x_step, 2)],
            [- self.k_2 * self.l / np.power(self.y_step, 2)],
            [1/self.t_step +
             2 * self.k_1 * (1-self.l)/np.power(self.x_step, 2) -
             2 * self.k_2 * (1-self.l)/np.power(self.y_step, 2)],
            [- self.k_1 * self.l / np.power(self.x_step, 2)],
            [- self.k_2 * self.l / np.power(self.y_step, 2)],
        ])
        
        if self.lattice[i, j] >= self.u_crit:
            coefs[0, 0] += self.l
            coefs[1, 0] += self.l
            coefs[2, 0] += + 1 - 2 * self.l
        return coefs
    

    def _build_discrete_coefs_bias(self, i, j):
        coefs = np.array([
            [self.k_1 * (1 - self.l) / np.power(self.x_step, 2)],
            [self.k_2 * (1 - self.l) / np.power(self.y_step, 2)],
            [1/self.t_step -
             2 * self.k_1 * self.l / np.power(self.x_step, 2) +
             2 * self.k_2 * self.l / np.power(self.y_step, 2)],
            [self.k_1 * (1 - self.l) / np.power(self.x_step, 2)],
            [self.k_2 * (1 - self.l) / np.power(self.y_step, 2)],
        ])
        
        if self.lattice[i, j] >= self.u_crit:
            coefs[0, 0] += - 1 + self.l
            coefs[1, 0] += - 1 + self.l
            coefs[2, 0] += + 3 + 2 * self.l
        return coefs

    def _emition_at(self, i, j):
        if (i,j) in self.emitters:
            return self.phi
        return 0
    
    def _compute_bias_at(self, i, j):
        coefs = self._build_discrete_coefs_bias(i, j)
        concentration = np.array([
            [self.lattice[i+1, j]],
            [self.lattice[i, j+1]],
            [self.lattice[i, j]],
            [self.lattice[i-1, j]],
            [self.lattice[i, j-1]],
        ])
        return np.sum(coefs * concentration) + self._emition_at(i, j)

    
    def _build_next_lattice_coefs(self):        
        n_sq = np.power(self.n, 2)
        next_lattice_coefs = np.zeros((n_sq, n_sq))
        bias = np.zeros((n_sq, 1))
        for node_idx in self.nodes:
            b = self._compute_bias_at(node_idx[0], node_idx[1])
            row_idx = self.n * node_idx[0] + node_idx[1]
            bias[row_idx, 0] = b
            left_coefs = self._build_discrete_eq_coefs_next(node_idx[0], node_idx[1])
            next_lattice_coefs[
                               row_idx,
                               self.n * (node_idx[0] + 1) + node_idx[1]
                ] = left_coefs[0, 0]
            next_lattice_coefs[
                               row_idx,
                               self.n * (node_idx[0]) + node_idx[1] + 1
                ] = left_coefs[1, 0]
            next_lattice_coefs[
                               row_idx,
                               self.n * (node_idx[0]) + node_idx[1]
                ] = left_coefs[2, 0]
            next_lattice_coefs[
                               row_idx,
                               self.n * (node_idx[0]) + node_idx[1]
                ] = left_coefs[3, 0]
            next_lattice_coefs[
                               row_idx,
                               self.n * (node_idx[0]) + node_idx[1] - 1
                ] = left_coefs[4, 0]
        return next_lattice_coefs
    
    
    def step(self):
        pass
```
:::

::: {#cc46a4f4-066f-401d-affe-c73ddb71dd4c .cell .code execution_count="30"}
``` python
k_1, k_2 = [0.8, 0.7]
l = 0.5                      # lambda
n = 4                        # number of nodes
x, y = [(0, 1), (0, 1)]      # observed area
t_step = 1                   # time step
u_crit = 40                  # critical concentration
phi = 5                      # emitters strength
emitters = [(3, 3), (9, 9)]  # emitters location at node
initial = lambda x, y: 0

solver = DiffusionDiffEquationSolver(k_1, k_2, l,
                                     n, x, y,
                                     t_step, u_crit,
                                     emitters, phi,
                                     initial)

print(vars(solver))
print(solver.lattice)
```

::: {.output .stream .stdout}
    {'k_1': 0.8, 'k_2': 0.7, 'l': 0.5, 'n': 4, 'nodesx': array([0.        , 0.33333333, 0.66666667, 1.        ]), 'nodesy': array([0.        , 0.33333333, 0.66666667, 1.        ]), 'nodes': array([[0, 0],
           [0, 1],
           [0, 2],
           [0, 3],
           [1, 0],
           [1, 1],
           [1, 2],
           [1, 3],
           [2, 0],
           [2, 1],
           [2, 2],
           [2, 3],
           [3, 0],
           [3, 1],
           [3, 2],
           [3, 3]]), 'x_step': 0.3333333333333333, 'y_step': 0.3333333333333333, 't_step': 1, 'u_crit': 40, 'phi': 5, 'emitters': [(3, 3), (9, 9)], 'lattice': array([[0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]])}
    [[0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]
     [0. 0. 0. 0.]]
:::
:::

::: {#78d2a32d-16f5-4aed-8962-2e018327b9a5 .cell .code execution_count="31"}
``` python
solver._build_next_lattice_coefs()
```

::: {.output .error ename="IndexError" evalue="index 4 is out of bounds for axis 1 with size 4"}
    ---------------------------------------------------------------------------
    IndexError                                Traceback (most recent call last)
    Input In [31], in <cell line: 1>()
    ----> 1 solver._build_next_lattice_coefs()

    Input In [29], in DiffusionDiffEquationSolver._build_next_lattice_coefs(self)
         95 bias = np.zeros((n_sq, 1))
         96 for node_idx in self.nodes:
    ---> 97     b = self._compute_bias_at(node_idx[0], node_idx[1])
         98     row_idx = self.n * node_idx[0] + node_idx[1]
         99     bias[row_idx, 0] = b

    Input In [29], in DiffusionDiffEquationSolver._compute_bias_at(self, i, j)
         80 def _compute_bias_at(self, i, j):
         81     coefs = self._build_discrete_coefs_bias(i, j)
         82     concentration = np.array([
         83         [self.lattice[i+1, j]],
    ---> 84         [self.lattice[i, j+1]],
         85         [self.lattice[i, j]],
         86         [self.lattice[i-1, j]],
         87         [self.lattice[i, j-1]],
         88     ])
         89     return np.sum(coefs * concentration) + self._emition_at(i, j)

    IndexError: index 4 is out of bounds for axis 1 with size 4
:::
:::

::: {#3c213e3a-f707-49b8-a5ab-72426f610561 .cell .code}
``` python
```
:::

::: {#90429432-db82-4a8c-a068-b84e191c599e .cell .code execution_count="35"}
``` python
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
main.py
"""
__author__      = "take-iwiw"
__copyright__   = "Copyright 2017, take-iwiw"
__date__        = "18 Oct 2017"

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pandas as pd
import time
import matplotlib.animation as animation

NUMBER_X: int = 10
NUMBER_Y: int = 10

CANVAS_WIDTH:  int = 10
CANVAS_HEIGHT: int = 10

def heatmap_animation1():
    fig, ax_lst = plt.subplots(NUMBER_X, NUMBER_Y)
    ax_lst = ax_lst.ravel()

    def plot(data):
        data = np.random.rand(CANVAS_WIDTH, CANVAS_HEIGHT)
        heatmap = ax_lst[0].pcolor(data)

    ani = animation.FuncAnimation(fig, plot, interval=1)
    ani.save('animation.gif')
    plt.show()

def heatmap_animation2():
    fig, ax_lst = plt.subplots(NUMBER_X, NUMBER_Y)
    ax_lst = ax_lst.ravel()

    data = np.random.rand(CANVAS_WIDTH, CANVAS_HEIGHT)
    im = ax_lst[0].imshow(data)

    while True:
        t_start = time.time()
        data = np.random.rand(CANVAS_WIDTH, CANVAS_HEIGHT)
        im.set_data(data) 
        plt.pause(0.001)
        t_end = time.time()
        print("fps = {0}".format(999 if t_end == t_start else 1/(t_end-t_start)))

def heatmap_animation3():
    fig, ax_lst = plt.subplots(NUMBER_X, NUMBER_Y)
    ax_lst = ax_lst.ravel()

    data = np.random.rand(CANVAS_WIDTH, CANVAS_HEIGHT)
    heatmap = ax_lst[0].pcolor(data)
    fig.canvas.draw()
    fig.show()

    while True:
        data = np.random.rand(CANVAS_WIDTH, CANVAS_HEIGHT)
        t_start = time.time()
        heatmap = ax_lst[0].pcolor(data)
        ax_lst[0].draw_artist(ax_lst[0].patch)
        ax_lst[0].draw_artist(heatmap)
        fig.canvas.blit(ax_lst[0].bbox)
        fig.canvas.flush_events()
        t_end = time.time()
        print("fps = {0}".format(999 if t_end == t_start else 1/(t_end-t_start)))


def main():
    """
    Entry function
    :called when: the program starts
    :param none: no parameter
    :return: none
    :rtype: none
    """
    heatmap_animation1()



if __name__ == '__main__':
    main()
```

::: {.output .stream .stderr}
    MovieWriter stderr:
    Error writing trailer of animation.gif: Invalid argument


    KeyboardInterrupt
:::

::: {.output .stream .stdout}
    Error in callback <function flush_figures at 0x7f3eef8fe3b0> (for post_execute):
:::

::: {.output .stream .stderr}

    KeyboardInterrupt
:::
:::

::: {#22a5d8cb-54dc-4368-87d7-763a46910454 .cell .code}
``` python
```
:::
