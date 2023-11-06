package neural

import "math"

func HardLimit(x float64) float64 {
	if x >= 0 {
		return 1
	} else {
		return 0
	}
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidDerivative(x float64) float64 {
	return x * (1 - x)
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func ReLUDerivative(x float64) float64 {
	if x > 0 {
		return 1
	} else {
		return 0
	}
}
