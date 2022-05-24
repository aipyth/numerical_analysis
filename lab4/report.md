---
title: Numerical analysis laboratory work №4
abstract: Finding eigenvalues of a matrix
author: Ivan Zhytkevych FI-91
language: en
header-includes: |
  \usepackage[utf8]{inputenc}
  \usepackage{fancyhdr}
  \pagestyle{fancy}
  \usepackage{hyperref}
  \usepackage{xcolor}
  \hypersetup{%
    colorlinks=true,
    linkcolor=blue,
    filecolor=magenta,      
    urlcolor=cyan,
    linkbordercolor={0 0 1}
  }
---

## System
$$ A = \begin{bmatrix}
    6.3 & 1.07 & 0.99 & 1.2 \\
    1.07 & 4.12 & 1.3 &  0.16 \\
    0.99 & 1.3 &  5.48 & 2.1 \\
    1.2 &  0.16 & 2.1 &  6.06
\end{bmatrix} $$

## Program output
```bash
$ go build && ./lab4
A = ⎡ 6.3  1.07  0.99   1.2⎤
    ⎢1.07  4.12   1.3  0.16⎥
    ⎢0.99   1.3  5.48   2.1⎥
    ⎣ 1.2  0.16   2.1  6.06⎦

Eigenvalues of A:
[(9.231213674637974+0i) (2.805857676530958+0i) (5.450954011402351+0i) (4.471974637428714+0i)]
initial vector y0: _______________
|             |
|   0.000000  |
|   0.000000  |
|   0.000000  |
|   1.000000  |
|_____________|
Full Krylov system
________________________________________________________________________
|                                                                      |
|   190.148396    17.082200     1.200000     0.000000 |  -1936.979267  |
|    81.661162     5.642800     0.160000     0.000000 |   -923.647933  |
|   254.157738    25.630000     2.100000     0.000000 |  -2387.279671  |
|   333.375640    42.599200     6.060000     1.000000 |  -2795.231489  |
|______________________________________________________________________|
Coefficients
_________________
|               |
|     1.000000  |
|   -21.960000  |
|   169.721000  |
|  -550.440464  |
|   631.387954  |
|_______________|
Eigenvalues:
_______________
|             |
|   2.805858  |
|   4.471975  |
|   5.450954  |
|   9.231214  |
|_____________|

```
The solution coincides with the one we got using `gonum` library
$$
\begin{pmatrix}
   2.805858 \\
   4.471975 \\
   5.450954 \\
   9.231214
\end{pmatrix}
\begin{pmatrix}
9.231213674637974+0i \\
2.805857676530958+0i \\ 
5.450954011402351+0i \\
4.471974637428714+0i
\end{pmatrix}
$$

## Code

Listed on <u>[github](https://github.com/aipyth/numerical_analysis/tree/master/lab4/code)</u>
