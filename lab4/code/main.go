package main

import (
  "fmt"
  "log"
  "gonum.org/v1/gonum/mat"
  )

func main() {
  M := NewMatrix([][]float64{
    {6.3,  1.07, 0.99, 1.2 },
    {1.07, 4.12, 1.3,  0.16},
    {0.99, 1.3,  5.48, 2.1 },
    {1.2,  0.16, 2.1,  6.06},
  })

  a := mat.NewDense(4, 4, []float64{
    6.3,  1.07, 0.99, 1.2,
    1.07, 4.12, 1.3,  0.16,
    0.99, 1.3,  5.48, 2.1,
    1.2,  0.16, 2.1,  6.06,
	})
	fmt.Printf("A = %v\n\n", mat.Formatted(a, mat.Prefix("    ")))

	var eig mat.Eigen
	ok := eig.Factorize(a, mat.EigenLeft)
	if !ok {
		log.Fatal("Eigendecomposition failed")
	}
	fmt.Printf("Eigenvalues of A:\n%v\n", eig.Values(nil))

  y0 := NewMatrix([][]float64{
    {0},
    {0},
    {0},
    {1},
  })
  fmt.Printf("initial vector y0: %v\n", y0)
  mEig := &Eigen{}
  mEig.FactorizeKrylov(M, y0, &EigenKrylovOpts{
      SLESolve: "gonum",
    },
  )
  fmt.Printf("Eigenvalues: \n%v\n", mEig.Extention)
}
