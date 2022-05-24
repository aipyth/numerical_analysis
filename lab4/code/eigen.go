package main

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type Eigen struct {
  Matrix
}

type EigenKrylovOpts struct {
  SLESolve string  // solve using `sqrt`, `simpleiteration`, `gonum`
  eps      float64 // for simple iteration method
}

var DefaultEigenKrylovOpts = &EigenKrylovOpts{
  SLESolve: "gonum",
  eps: 0,
}

func (e *Eigen) FactorizeKrylov(a *Matrix, s *Matrix, opts *EigenKrylovOpts) {
  if s.n != 1 {
    panic("invalid matrix s")
  }

  n := int(a.m)

  // precompute y = a^k * s
  aPows := make(map[int]*Matrix)
  aPows[0] = NewIMatrix(n)
  aPows[1] = a
  for i := 2; i <= n; i++ {
    aPows[i] = aPows[i-1].Dot(a)
  }

  y := MatrixInit(4, 4)
  for j := 0; j < n; j++ {
    k := aPows[j].Dot(s)
    for i, row := range k.data {
      y.data[i][n - j - 1] = row[0]
    }
  }

  y.AddExtention(
    aPows[n].
      Dot(s).
      Mult(math.Pow(-1, float64(n + 1))),
  )

  fmt.Println("Full Krylov system")
  fmt.Println(y)
  
  var coefficients *Matrix
  if opts == nil { opts = DefaultEigenKrylovOpts }
  switch opts.SLESolve {
  case "sqrt":
    // coefficients = y.SolveSQRTMethod(y.Extention)
  case "simpleiteration":
    // if opts.eps == 0 { opts.eps = DefaultEigenKrylovOpts.eps }
    // coefficients = y.SolveSimpleIteration(opts.eps)
  default:
    d := new(mat.Dense)
    if err := d.Solve(y, y.Extention); err != nil {
      panic(err)
    }
    r, c := d.Dims()
    coefficients = MatrixInit(uint(r)+1, uint(c))
    coefficients.data[0][0] = math.Pow(-1, float64(n))
    for i := 1; i < r+1; i++ {
      for j := 0; j < c; j++ {
        coefficients.data[i][j] = d.At(i-1, j)
      }
    }
  }

  fmt.Println("Coefficients")
  fmt.Println(coefficients)

  roots := MatrixInit(uint(n), 1)
  roots.SolveNonlinear(coefficients.T(), nil)

  // e = &Eigen{*MatrixInit(uint(n), uint(n))}
  e.Restart(n, n)
  e.AddExtention(roots)
}
