package main

import (
	// "time"
	// "image/color"
	"math"

	"gonum.org/v1/gonum/mat"
	// "gonum.org/v1/plot"
	// "gonum.org/v1/plot/plotter"
	// "gonum.org/v1/plot/vg"
)

type SolveNonlinearOpts struct {
  Method string // "bisection"
}

// Solutions' bounds for all complex roots
func SolutionsBounds(m mat.Matrix) (float64, float64) {
  k, n := m.Dims()
  if k != 1 { return 0, 0 }

  a := m.At(0, 0)
  for i := 1; i < n - 1; i++ {
     if math.Abs(m.At(0, i)) > a {
      a = math.Abs(m.At(0, i))
    }
  }

  b := m.At(0, 1)
  for i := 2; i < n; i++ {
     if math.Abs(m.At(0, i)) > b {
      b = math.Abs(m.At(0, i))
    }
  }

  lower := math.Abs(m.At(0, n-1)) / (b + math.Abs(m.At(0, n-1)))
  upper := (math.Abs(m.At(0, 0)) + a) / math.Abs(m.At(0, 0))
  return lower, upper
}

func cutIntoIntervals(a, b, step float64, f func(float64) float64) [][]float64 {
  nPieces := math.Ceil(math.Abs(b - a) / step)
  pieces := make([][]float64, 0, int(nPieces))
  for ai := a; ai < b; ai += step {
    pieces = append(pieces, []float64{ai, ai + step})
  }
  intervals := make([][]float64, 0)
  var intervalStart float64 = pieces[0][0]
  var intervalStartVal float64 = f(intervalStart)
  var pieceValue float64
  for _, piece := range pieces {
    pieceValue = f(piece[1])
    if intervalStartVal * pieceValue < 0 {

      intervals = append(intervals, []float64{intervalStart, piece[1]})
      intervalStartVal = pieceValue
      intervalStart = piece[1]
    }
  }
  return intervals
}

func evaluatePolynomial(coefs mat.Matrix) func(float64) float64 {
  _, n := coefs.Dims()
  return func (x float64) float64 {
    res := float64(0)
    for i := 0; i < int(n); i++ {
      res += coefs.At(0, i) * math.Pow(x, float64(int(n) - i - 1))
    }
    return res
  }
}

func dichotomy(a, b float64, f func(float64) float64, eps float64) float64 {
  left := a
  right := b
  var c float64
  for math.Abs(right - left) >= eps {
    c = (right + left) / 2
    if f(left) * f(c) <= 0 { right = c } else { left = c }
  }
  return c
}

// Solve nonlinear equation
func (m *Matrix) SolveNonlinear(a mat.Matrix, opts *SolveNonlinearOpts) {
  f := evaluatePolynomial(a)

  lower, upper := SolutionsBounds(a)

  intervalsStep := 1e-2
  intervals := make([][]float64, 0)
  intervals = append(intervals, cutIntoIntervals(-upper, -lower, intervalsStep, f)...)
  intervals = append(intervals, cutIntoIntervals(lower, upper, intervalsStep, f)...)

  roots := make([]float64, 0, len(intervals))
  for _, interval := range intervals {
    roots = append(roots, dichotomy(interval[0], interval[1], f, 1e-8))
  }
  for i, root := range roots {
    m.data[i][0] = root
  }
}


 //  p := plot.New()
 //  p.Title.Text = "Characteristics Polynom"
	// p.X.Label.Text = "x"
	// p.Y.Label.Text = "P(x)"
 //  pl := plotter.NewFunction(f)
 //  pl.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
 //  p.Add(pl)
	// p.Legend.Add("P(x)", pl)
	// p.Legend.ThumbnailWidth = 0.5 * vg.Inch
	//
	// // Set the axis ranges.  Unlike other data sets,
	// // functions don't set the axis ranges automatically
	// // since functions don't necessarily have a
	// // finite range of x and y values.
	// p.X.Min = 0
	// p.X.Max = 10
	// p.Y.Min = -5
	// p.Y.Max = 5
	//
 //  // Save the plot to a PNG file.
	// if err := p.Save(8*vg.Inch, 8*vg.Inch, "polynom.png"); err != nil {
	// 	panic(err)
	// }
