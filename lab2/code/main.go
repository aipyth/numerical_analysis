package main

import "fmt"

func main() {
    // {
    //     {2.12, 0.42, 1.34, 0.88},
    //     {0.42, 3.95, 1.87, 0.43},
    //     {1.34, 1.87, 2.98, 0.46},
    //     {0.88, 0.43, 0.46, 4.44},
    // };
    // { 11.172, 0.115, 0.009, 9.349 };

    A := NewMatrix([][]float64{
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

    solution := A.SolveSQRTMethod(b)
    fmt.Println("Solution = ", solution)

    b0 := A.Dot(solution)
    fmt.Println("A \\cdot solution = ", b0)
    fmt.Println("нев'язок b = ", b.Minus(b0))

    eps := A.SolveSQRTMethod(b.Minus(b0))
    fmt.Println("x eps", eps)
}
