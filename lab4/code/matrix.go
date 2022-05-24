package main

import (
	"fmt"
	"math"
	"strings"

	"gonum.org/v1/gonum/mat"
)

type Matrix struct {
  data [][]float64
  m uint // rows
  n uint // columns

  Extention *Matrix
}

var OutputPrecision = 6

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

func NewIMatrix(m int) *Matrix {
  if m == 0 {
    return MatrixInit(0, 0)
  }
  n := m
  M := MatrixInit(uint(m), uint(n))
  for i := 0; i < m; i++ {
    M.data[i][i] = 1
  }
  return M;

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

func (M *Matrix) Restart(r, c int) {
  M.m = uint(r)
  M.n = uint(c)
  M.data = make([][]float64, r)
  for mi := range M.data {
    M.data[mi] = make([]float64, c)
  }
}

func (m *Matrix) AddExtention(ext *Matrix) *Matrix {
  m.Extention = ext
  return m
}

func (m Matrix) AbsMax() (float64) {
  if len(m.data) == 0 { return 0 }
  if len(m.data[0]) == 0 { return 0 }
  max := math.Abs(m.data[0][0])
  for _, row := range m.data {
    for _, item := range row {
      if math.Abs(item) > max {
        max = math.Abs(item)
      }
    }
  }
  return max
}

func (m Matrix) String() (s string) {
  mainLength := len(strings.Split(fmt.Sprint(m.AbsMax()), ".")[0]) + OutputPrecision + 3
  mainLengthStr := fmt.Sprint(mainLength)
  outputPrecisionStr := fmt.Sprint(OutputPrecision)
  format := "% " + mainLengthStr + "." + outputPrecisionStr + "f"
  s += "\n"
  for i, row := range m.data {
    s += fmt.Sprintf("| ")
    for _, item := range row {
      s += fmt.Sprintf(format, item)
      s += " "
    }
    if m.Extention != nil {
      mainLength := len(strings.Split(fmt.Sprint(m.Extention.AbsMax()), ".")[0]) + OutputPrecision + 3
      mainLengthStr := fmt.Sprint(mainLength)
      outputPrecisionStr := fmt.Sprint(OutputPrecision)
      format := "% " + mainLengthStr + "." + outputPrecisionStr + "f"
      s += "| "
      for _, item := range m.Extention.data[i] {
        s += fmt.Sprintf(format, item)
        s += " "
      }
    }
    s += fmt.Sprintf(" |\n")
  }
  headingLen := len(strings.Split(s, "\n")[1])
  s = strings.Repeat("_", headingLen) + "\n|" + strings.Repeat(" ", headingLen - 2) + "|" + s
  s += "|" + strings.Repeat("_", headingLen - 2) + "|"
  return s
}

func (m Matrix) ApplyPermutations(permutations *Matrix) *Matrix {
  result := NewMatrix(m.data)
  if m.Extention != nil {
    result.AddExtention(m.Extention)
  }
  for i, row := range permutations.data {
    for _, elem := range row {
      result.RearrangeRows(i, int(elem))
    }
  }
  return result
}

func (m Matrix) RowAbsMax(row int) (float64, int) {
  ind := 0
  max := math.Abs(m.data[row][ind])
  for i, item := range m.data[row] {
    if math.Abs(item) > max {
      max = math.Abs(item)
      ind = i
    }
  }
  return max, ind
}

func (m Matrix) RearrangeColumns(i, j int) {
  for _, row := range m.data {
    row[i], row[j] = row[j], row[i]
  }
}

func (m Matrix) RearrangeRows(i, j int) {
  m.data[i], m.data[j] = m.data[j], m.data[i]
}

func (m Matrix) ReorderOnlyToDiagonal() (*Matrix, *Matrix) {
  mt := NewMatrix(m.data)
  if m.Extention != nil {
    mt.Extention = NewMatrix(m.Extention.data)
  }

  pertts := make([][]float64, mt.n)
  for i := range pertts {
    pertts[i] = []float64{float64(i)}
  }
  permutations := NewMatrix(pertts)

  var rowMaxIdx int
  var elem float64
  var matrixMaxElem float64
  for i := range mt.data {
    elem, rowMaxIdx = mt.RowAbsMax(i)
    if rowMaxIdx <= i && elem > matrixMaxElem {
      matrixMaxElem = elem
    } else if rowMaxIdx <= i {
      continue
    }
    mt.RearrangeColumns(i, rowMaxIdx)
    permutations.RearrangeRows(i, rowMaxIdx)
  }
  return mt, permutations
}

func (m Matrix) IsWithDiagonalDominance() bool {
  absSumExcept := func (except int, row []float64) float64 {
    s := float64(0)
    for i, item := range row {
      if i == except { continue }
      s += math.Abs(item)
    }
    return s
  }
  for i, row := range m.data {
    if row[i] <= absSumExcept(i, row) {
      return false
    }
  }
  return true
}

func (m Matrix) computeLinearRowsSum(coef float64, targetRowIdx, sourceRowIdx int) []float64 {
  resultRow := make([]float64, m.n)
  copy(resultRow, m.data[targetRowIdx])
  for i, item := range resultRow {
    resultRow[i] = coef * m.data[sourceRowIdx][i] + item
  }
  return resultRow
}

func (m *Matrix) LinearRowSum(coef float64, targetIdx, sourceIdx int) {
  m.data[targetIdx] = m.computeLinearRowsSum(coef, targetIdx, sourceIdx)
  if m.Extention != nil {
    m.Extention.data[targetIdx] = m.Extention.computeLinearRowsSum(coef, targetIdx, sourceIdx)
  }
}

func (m Matrix) ToDiagonalDominance() (*Matrix, *Matrix) {
  mt, permutations := m.ReorderOnlyToDiagonal()

  closeness := func (i int, row []float64) float64 {
    main := math.Abs(row[i])
    sum := float64(0)
    for j := range row {
      if j != i {
        sum += math.Abs(row[j])
      }
    }
    return main - sum
  }


  closenessAfterMerge := func (coef float64, targetRowIdx, sourceRowIdx int) float64 {
    resultRow := mt.computeLinearRowsSum(coef, targetRowIdx, sourceRowIdx)
    return closeness(targetRowIdx, resultRow)
  }


  mergeBestToRow := func (i int) bool {
    closenessValues := make([]float64, mt.n)
    var maxClosenessIdx int
    var coefficient float64

    var coef float64
    for j := 0; uint(j) < mt.m; j++ {
      if j == i { continue }

      // calculate coefficient (using optimization methods)
      coef = divideByHalfOptimization(func (x float64) float64 {
        return -closenessAfterMerge(x, i, j)
      }, -10, 10, 0.001)
      // fmt.Println("Coef after optimization:", coef)

      closenessValues[j] = closenessAfterMerge(coef, i, j)
      if closenessValues[j] > closenessValues[maxClosenessIdx] {
        maxClosenessIdx = j
        coefficient = coef
      }
    }

    // fmt.Printf("Closeness values after merging i-th rows to %d row: %v\n", i, closenessValues)
    // fmt.Printf("Closeness with maximum value is %d-th closeness: %f\n", maxClosenessIdx, closenessValues[maxClosenessIdx])
    // fmt.Printf("Coefficient: %f\n", coefficient)
    // linearSum := mt.computeLinearRowsSum(coefficient, i, maxClosenessIdx)
    // fmt.Printf("Linear rows sum: %v\n", linearSum)
    mt.LinearRowSum(coefficient, i, maxClosenessIdx)

    // fmt.Print()

    return false
  }

  // avgCloseness := func () float64 {
  //   c := float64(0)
  //   for i, row := range mt.data {
  //     c += closeness(i, row)
  //   }
  //   return c / float64(mt.n)
  // }

  for !mt.IsWithDiagonalDominance() {
    // fmt.Println("Matrix is not with diagonal dominance")
    for i, row := range mt.data {
      // fmt.Printf("Row %d closeness %f\n", i, closeness(i, row))
      if (closeness(i, row) <= 0) {
        mergeBestToRow(i)
      }
    }
    // fmt.Printf("Average closeness: %f\n", avgCloseness())
    // fmt.Println("Current matrix:")
    // fmt.Println(mt)
  }

  return mt, permutations
}

func (m *Matrix) SolveSimpleIteration(eps float64) *Matrix {
  pertts := make([][]float64, m.n)
  for i := range pertts {
    pertts[i] = []float64{float64(i)}
  }
  permutations := NewMatrix(pertts)

  if !m.IsWithDiagonalDominance() {
    m, permutations = m.ToDiagonalDominance()
    // return nil
  }

  computeMatrixC := func () *Matrix {
    c := MatrixInit(m.m, m.n)
    for i, row := range c.data {
      for j := range row {
        if i == j { continue }
        c.data[i][j] = - m.data[i][j] / m.data[i][i]
      }
    }
    return c
  }

  computeMatrixD := func () *Matrix {
    d := MatrixInit(m.m, 1)
    for i, row := range m.Extention.data {
      for j := range row {
        d.data[i][j] = m.Extention.data[i][j] / m.data[i][i]
      }
    }
    return d
  }

  computeQ := func (c *Matrix) float64 {
    var q float64
    var sum float64
    for _, row := range c.data {
      sum = 0
      for _, elem := range row {
        sum += math.Abs(elem)
      }
      if sum > q {
        q = sum
      }
    }
    return q
  }


  C := computeMatrixC()
  D := computeMatrixD()
  Q := computeQ(C)
  
  convergenceCriteria := func (x *Matrix, xPrev *Matrix) bool {
    var maxDiff float64
    var diff float64
    for i, row := range x.data {
      for j := range row {
        diff = math.Abs(x.data[i][j] - xPrev.data[i][j])
        if diff > maxDiff {
          maxDiff = diff
        }
      }
    }
    return (Q / (1 - Q)) * maxDiff < eps
  }

  computeNextX := func (x *Matrix) *Matrix {
    return C.Dot(x).Plus(D)
  }

  solution := MatrixInit(m.m, 1)
  var previousSolution *Matrix

  iteration := 0

  for {
    previousSolution = solution
    solution = computeNextX(solution)

    // fmt.Printf("Iteration %d with solution:\n", iteration)
    // fmt.Println(solution)
    //
    // fmt.Println("And residual:")
    // fmt.Println(m.Residual(solution))

    if convergenceCriteria(solution, previousSolution) {
      break
    }

    iteration++
  }

  solution = solution.ApplyPermutations(permutations)
  return solution
}

func (m Matrix) Residual(solution *Matrix) *Matrix {
  if m.Extention == nil {
    return nil
  }
  return m.Extention.Minus(m.Dot(solution))
}

// -----------------------------------------
//       PART OF 2nd LAB
// -----------------------------------------

func (M Matrix) Factorization() *Matrix {
  if M.m != M.n { return nil }

  C := MatrixInit(M.m, M.n)

  // c_{ii} = sqrt{ a_{ii} - sum_{k=1}^{i-1} c_{ki}^2 }   i = \overline{2, n}
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

  // c_{1j} = \frac {a_{1j}} {c_{11}}   j = \overline{2, n}
  for j := uint(1); j < M.n; j++ {
    if C.data[0][0] == 0 { return nil }
    C.data[0][j] = M.data[0][j] / C.data[0][0]
  }

  // fmt.Println("C after c_11 and c_1j inits:", C)

  for i := uint(1); i < C.m; i++ {
    for j := uint(0); j < C.n; j++ {
      // fmt.Printf("C on %d,%d step", i, j)
      // fmt.Println(C)
      if j < i {
        C.data[i][j] = 0
      } else if j == i {
        C.data[i][j] = calc_diag(i)
      } else if j > i {
        C.data[i][j] = calc_other(i, j)
      }
    }
  }
  return C
}

func (M *Matrix) SolveSQRTMethod(b *Matrix) *Matrix {
  if b.m != M.m { return nil }
  if b.n != 1   { return nil }
  if M.m != M.n { return nil }

  n := M.m

  C := M.Factorization()
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

func (M Matrix) Mult(c float64) *Matrix {
  R := MatrixInit(M.m, M.n)
  for i := uint(0); i < M.m; i++ {
    for j := uint(0); j < M.n; j++ {
      R.data[i][j] = M.data[i][j] * c
    }
  }
  return R
}

func (M *Matrix) Pow(k int) *Matrix {
  if k == 1 { return M }
  return M.Dot(M).Pow(k-1)
}

func (M Matrix) Plus(A *Matrix) *Matrix {
  if M.m != A.m { return nil }
  if M.n != A.n { return nil }

  R := MatrixInit(M.m, A.n)
  for i := uint(0); i < M.m; i++ {
    for j := uint(0); j < A.n; j++ {
      R.data[i][j] = M.data[i][j] + A.data[i][j]
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

func (M Matrix) T() mat.Matrix {
  R := *MatrixInit(M.n, M.m)
  for i := uint(0); i < M.m; i++ {
    for j := uint(0); j < M.n; j++ {
      R.data[j][i] = M.data[i][j]
    }
  }
  return R
}

func (M Matrix) Dims() (r, c int) {
  return int(M.m), int(M.n)
}

func (M Matrix) At(i, j int) float64 {
  return M.data[i][j]
}
