package main

import "fmt"

func main() {
    M := NewMatrix([][]float64{
        {2.12, 0.42, 1.34, 0.88},
        {0.42, 3.95, 1.87, 0.43},
        {1.34, 1.87, 2.98, 0.46},
        {0.88, 0.43, 0.46, 4.44},
    })

    b := NewMatrix([][]float64{
        { 11.172 },
        {  0.115 },
        {  0.009 },
        {  9.349 },
    })

    M.AddExtention(b)

    fmt.Println(M)


    OutputPrecision = 8

    var permutations *Matrix
    M, permutations = M.ToDiagonalDominance()
    fmt.Println(M)

    fmt.Println("Using Simple iteration...")

    solution := M.SolveSimpleIteration(0.000001)

    solution = solution.ApplyPermutations(permutations)

    fmt.Println("Solution:")
    fmt.Println(solution)
}
