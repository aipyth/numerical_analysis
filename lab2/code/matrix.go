package main

import (
	"fmt"
	"math"
)

type Matrix struct {
    data [][]float64
    m uint
    n uint
}

func MatrixInit(m, n uint) *Matrix {
    M := new(Matrix)
    M.m = m
    M.n = n
    M.data = make([][]float64, m)
    for mi := range M.data {
        M.data[mi] = make([]float64, n)
    }
    return M
}

func NewMatrix(m [][]float64) *Matrix {
    if len(m) == 0 {
        return MatrixInit(0, 0)
    }
    n := len(m[0])
    M := MatrixInit(uint(len(m)), uint(n))
    for i := range m {
        copy(M.data[i], m[i])
    }
    return M;
}

func (m Matrix) String() (s string) {
    // s += fmt.Sprintf("--\n")
    s += "\n"
    for _, row := range m.data {
        s += fmt.Sprintf("| ")
        for _, item := range row {
            // s += fmt.Sprintf("%+02.6f  ", item)
            s += fmt.Sprint(item)
            s += "   "
        }
        s += fmt.Sprintf(" |\n")
    }
    // s += fmt.Sprintf("--\n")
    return
}


func (M Matrix) LUDecompose() *Matrix {
    // cannot decompose not square matrixes
    if M.m != M.n { return nil }

    C := MatrixInit(M.m, M.n)

    // c_{ii} = sqrt{ a_{ii} - sum_{k=1}^{i-1} c_{ki}^2 }     i = \overline{2, n}
    calc_diag := func(i uint) float64 {
        var cSum float64 = 0
        for k := uint(0); k < i; k++ {
            cSum += math.Pow(C.data[k][i], 2)
        }
        return math.Sqrt(M.data[i][i] - cSum)
    }

    calc_other := func(i, j uint) float64 {
        var cSum float64
        for k := uint(0); k < i; k++ {
            cSum += C.data[k][i] * C.data[k][j]
        }
        return (M.data[i][j] - cSum) / C.data[i][i]
    }

    // c_{11} = \sqrt{a_{11}}
    C.data[0][0] = math.Sqrt(M.data[0][0])

    // c_{1j} = \frac {a_{1j}} {c_{11}}     j = \overline{2, n}
    for j := uint(1); j < M.n; j++ {
        if C.data[0][0] == 0 { return nil }
        C.data[0][j] = M.data[0][j] / C.data[0][0]
    }

    fmt.Println("C after c_11 and c_1j inits:", C)

    for i := uint(1); i < C.m; i++ {
        for j := uint(0); j < C.n; j++ {
            fmt.Printf("C on %d,%d step", i, j)
            fmt.Println(C)
            if j < i {
                C.data[i][j] = 0
            } else if j == i {
                C.data[i][j] = calc_diag(i)
            } else if j > i {
                C.data[i][j] = calc_other(i, j)
            }
        }
    }
    
    // var cSum float64
    // c_{ii} = sqrt{ a_{ii} - sum_{k=1}^{i-1} c_{ki}^2 }     i = \overline{2, n}
    // for i := uint(1); i < M.n; i++ {
    //     cSum = 0
    //     for k := uint(0); k < i; k++ {
    //         cSum += math.Pow(C.data[k][i], 2)
    //     }
    //     C.data[i][i] = math.Sqrt(M.data[i][i] - cSum)
    // }

    // c_{ij} = \frac{ a_{ij} - sum_{k=1}^{i-1} c_{ki} c_{kj} } {c_{ii}}   (2 \leq i < j)
    // for j := uint(2); j < M.n; j++ {
    //     for i := uint(1); i < j; i++ {
    //         cSum = 0
    //         for k := uint(0); k < i; k++ {
    //             cSum += C.data[k][i] * C.data[k][j]
    //         }
    //         C.data[i][j] = (M.data[i][j] - cSum) / C.data[i][i]
    //     }
    // }

    return C
}

func (M *Matrix) SolveSQRTMethod(b *Matrix) *Matrix {
    if b.m != M.m { return nil }
    if b.n != 1   { return nil }
    if M.m != M.n { return nil }

    n := M.m

    C := M.LUDecompose()
    if C == nil { return nil }

    y := MatrixInit(n, 1)
    // y_1 = b_1 / c_11
    y.data[0][0] = b.data[0][0] / C.data[0][0]

    var cSum float64;
    for i := uint(1); i < n; i++ {
        cSum = 0
        for k := uint(0); k < i; k++ {
            cSum += C.data[k][i] * y.data[k][0]
        }
        y.data[i][0] = (b.data[i][0] - cSum) / C.data[i][i]
    }

    x := MatrixInit(n, 1)
    x.data[n-1][0] = y.data[n-1][0] / C.data[n-1][n-1]

    for i := uint(n-2); i < n-1; i-- {
        cSum = 0
        for k := uint(i+1); k < n; k++ {
            cSum += C.data[i][k] * x.data[k][0]
        }
        x.data[i][0] = (y.data[i][0] - cSum) / C.data[i][i]
    }

    return x
}

func (M Matrix) Dot(A *Matrix) *Matrix {
    if M.n != A.m { return nil }
    R := MatrixInit(M.m, A.n)
    for i := uint(0); i < M.m; i++ {
        for j := uint(0); j < A.n; j++ {
            for k := uint(0); k < M.n; k++ {
                R.data[i][j] += M.data[i][k] * A.data[k][j]
            }
        }
    }
    return R
}

func (M Matrix) Minus(A *Matrix) *Matrix {
    if M.m != A.m { return nil }
    if M.n != A.n { return nil }

    R := MatrixInit(M.m, A.n)
    for i := uint(0); i < M.m; i++ {
        for j := uint(0); j < A.n; j++ {
            R.data[i][j] = M.data[i][j] - A.data[i][j]
        }
    }
    return R
}
