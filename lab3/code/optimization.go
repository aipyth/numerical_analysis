package main

func divideByHalfOptimization(f func(float64) float64, a, b, eps float64) float64 {
    l := b - a
    x := (a + b) / 2
    var x1, x2, xValue, x1Value, x2Value float64
    for {
        x1 = a + l/4
        x2 = b - l/4
        xValue = f(x)
        x1Value = f(x1)
        x2Value = f(x2)

        if x1Value < xValue {
            b = x
            x = x1
        } else if x2Value < xValue {
            a = x
            x = x2
        } else {
            a = x1
            b = x2
        }

        l = b - a
        if l <= eps {
            break
        }
    }
    return x
}
