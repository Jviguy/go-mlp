package data

import (
	"go-preceptron/neural"
	"strconv"
)

func CSVRecordsToPatterns(records [][]string) []neural.Pattern {
	ps := make([]neural.Pattern, 0)
	for _, record := range records {
		p := neural.Pattern{
			Features:             make([]float64, 0),
			SingleRawExpectation: "",
			SingleExpectation:    0,
			MultipleExpectation:  nil,
		}
		for i, item := range record {
			// if last
			if i == len(record)-1 {
				switch item {
				case "Iris-setosa":
					p.MultipleExpectation = []float64{1, 0, 0}
				case "Iris-versicolor":
					p.MultipleExpectation = []float64{0, 1, 0}
				case "Iris-virginica":
					p.MultipleExpectation = []float64{0, 0, 1}
				}
				break
			}
			f, err := strconv.ParseFloat(item, 64)
			if err != nil {
				panic("INVALID INP DATA.")
			}
			p.Features = append(p.Features, f)
		}
		ps = append(ps, p)
	}
	return ps
}
