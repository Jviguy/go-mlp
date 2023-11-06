package neural

type Pattern struct {
	Features             []float64
	SingleRawExpectation string
	SingleExpectation    float64
	MultipleExpectation  []float64
}
